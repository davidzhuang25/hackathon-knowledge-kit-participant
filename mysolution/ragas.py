#!/usr/bin/env python3
"""
Evaluate aisolution on the 40 public questions with Ragas and write a CSV.

Steps:
    1. Build aisolution and ingest all snapshots in date order (skipped when already indexed)
    2. Call answer() for each question while capturing the retrieved context
    3. Score each answer with the harness scorer (harness/scoring.py)
    4. Compute 7 Ragas metrics and write results/ragas-<MMDDYYYY>-<n>.csv

Usage (from the project root):
    python aisolution/ragas.py                                   # harness scores only, no Ragas calls
    python aisolution/ragas.py --metrics context_recall,faithfulness
    python aisolution/ragas.py --metrics all --model gpt-4.1-mini --concurrency 4

Ragas metrics are off by default because they are expensive (about 49 LLM calls per question
with all 7 on and 12 retrieved chunks). Turn them on with --metrics (comma-separated names,
"all" or "none") or the RAGAS_METRICS environment variable. Disabled metrics leave empty cells.

Dependencies (local evaluation only; not submission dependencies, do not add to aisolution/requirements.txt):
    pip install ragas "langchain-community==0.4.1"
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# This file shares its name with the ragas package; remove this folder from sys.path first, or `import ragas` imports this file itself
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "tools"))

import argparse  # noqa: E402
import asyncio  # noqa: E402
import csv  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from dataclasses import asdict  # noqa: E402
from datetime import date  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

from make_eval_report import explain, join_list  # noqa: E402
from run_eval import load_snapshots, to_answer  # noqa: E402
from scoring import score_item  # noqa: E402

load_dotenv(ROOT / ".env", override=False)

# Replaced in main() with solution.EVIDENCE_CHARS, the truncation applied before text reaches the model
CONTEXT_CHARS = 1400

METRIC_COLUMNS = [
    "context_recall",
    "context_entity_recall",
    "context_precision",
    "faithfulness",
    "noise_sensitivity",
    "factual_correctness",
    "aspect_critic",
]

# Approximate LLM calls per question for each metric, where k is the number of retrieved chunks
METRIC_CALLS = {
    "context_recall": lambda k: 1,
    "context_entity_recall": lambda k: 2,
    "context_precision": lambda k: k,
    "faithfulness": lambda k: 2,
    "noise_sensitivity": lambda k: 2 * k + 3,
    "factual_correctness": lambda k: 4,
    "aspect_critic": lambda k: 1,
}


def parse_metrics(value: str) -> list[str]:
    """Turn 'all', 'none' or a comma-separated list into metric names, in METRIC_COLUMNS order."""
    names = {name.strip().lower() for name in (value or "").split(",") if name.strip()}
    if not names or names == {"none"}:
        return []
    if "all" in names:
        return list(METRIC_COLUMNS)
    unknown = sorted(names - set(METRIC_COLUMNS))
    if unknown:
        raise SystemExit(
            f"Unknown Ragas metric(s): {', '.join(unknown)}. "
            f"Choose from: {', '.join(METRIC_COLUMNS)}, all, none."
        )
    return [name for name in METRIC_COLUMNS if name in names]


COLUMNS = [
    "No.",
    "id",
    "correct",
    "score",
    *METRIC_COLUMNS,
    "how_score_calculated",
    "question",
    "as_of",
    "expected_answer",
    "my_answer",
    "category",
    "must_include",
    "must_not_include",
    "difficulty",
    "weight",
    "expected_sources",
    "group",
]

ASPECT_DEFINITION = (
    "Return 1 only if the response follows the credit-union assistant rules: it answers with the "
    "specific values the question asks for (every relevant tier or list item when the question is "
    "broad), it does not hedge between old and new values, and if the information is not available "
    "it clearly declines without stating any numbers. Otherwise return 0."
)


def load_solution_module():
    spec = importlib.util.spec_from_file_location("solution", HERE / "solution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_capturing_assistant(solution):
    """Capture each retrieval result through a subclass, without modifying solution.py."""

    class CapturingAssistant(solution.KnowledgeAssistant):
        def __init__(self, **kwargs):
            self.last_retrieval = None
            super().__init__(**kwargs)

        def _retrieve_and_prepare(self, request: dict) -> dict:
            prepared = super()._retrieve_and_prepare(request)
            self.last_retrieval = prepared
            return prepared

    return CapturingAssistant()


def collect_answers(assistant, items: list[dict]) -> list[dict]:
    records = []
    for number, item in enumerate(items, start=1):
        assistant.last_retrieval = None
        answer = to_answer(assistant.answer(item["question"], item.get("as_of")))
        documents = assistant.last_retrieval["documents"] if assistant.last_retrieval else []
        # Use the context exactly as the model saw it (same truncation) so faithfulness etc. reflect the real input
        contexts = [doc.page_content[:CONTEXT_CHARS] for doc in documents]

        result = score_item(item, answer)
        result["answer"] = asdict(answer)
        how, _ = explain(item, result)
        records.append({
            "number": number,
            "item": item,
            "answer": answer,
            "contexts": contexts,
            "result": result,
            "how": how,
        })
        print(f"[{number:2d}/{len(items)}] {item['id']} score={result['score']:.4f}", flush=True)
    return records


async def safe_value(coro) -> float | int | str:
    try:
        result = await coro
    except Exception as exc:  # record a failing metric in its cell and keep evaluating the others
        return f"ERROR: {type(exc).__name__}: {exc}"[:300]
    value = getattr(result, "value", result)
    return round(float(value), 4)


def metric_call(name: str, metric, question: str, reference: str, response: str, contexts: list[str]):
    """Build the scoring coroutine for one metric."""
    if name == "context_recall":
        return metric.ascore(user_input=question, retrieved_contexts=contexts, reference=reference)
    if name == "context_entity_recall":
        return metric.ascore(reference=reference, retrieved_contexts=contexts)
    if name == "context_precision":
        return metric.ascore(user_input=question, reference=reference, retrieved_contexts=contexts)
    if name == "faithfulness":
        return metric.ascore(user_input=question, response=response, retrieved_contexts=contexts)
    if name == "noise_sensitivity":
        return metric.ascore(
            user_input=question, response=response, reference=reference, retrieved_contexts=contexts)
    if name == "factual_correctness":
        return metric.ascore(response=response, reference=reference)
    if name == "aspect_critic":
        from ragas import SingleTurnSample
        return metric.single_turn_ascore(SingleTurnSample(user_input=question, response=response))
    raise ValueError(f"unknown metric {name}")


async def evaluate_record(record: dict, metrics: dict, semaphore: asyncio.Semaphore) -> dict:
    item = record["item"]
    question = item["question"]
    reference = item.get("expected_answer", "")
    response = record["answer"].text
    contexts = record["contexts"]

    names = list(metrics)
    async with semaphore:
        values = await asyncio.gather(*(
            safe_value(metric_call(name, metrics[name], question, reference, response, contexts))
            for name in names
        ))
    print(f"  ragas done: {item['id']}", flush=True)
    row = {name: "" for name in METRIC_COLUMNS}
    row.update(zip(names, values))
    return row


def build_metrics(enabled: list[str], model: str) -> dict:
    """Create only the enabled Ragas metric objects; ragas is imported only when needed."""
    from langchain_openai import ChatOpenAI
    from openai import AsyncOpenAI
    from ragas.llms import LangchainLLMWrapper, llm_factory
    from ragas.metrics import AspectCritic
    from ragas.metrics.collections import (
        ContextEntityRecall,
        ContextPrecision,
        ContextRecall,
        FactualCorrectness,
        Faithfulness,
        NoiseSensitivity,
    )

    modern = {
        "context_recall": ContextRecall,
        "context_entity_recall": ContextEntityRecall,
        "context_precision": ContextPrecision,
        "faithfulness": Faithfulness,
        "noise_sensitivity": NoiseSensitivity,
        "factual_correctness": FactualCorrectness,
    }
    metrics = {}
    llm = None
    for name in enabled:
        if name == "aspect_critic":
            # AspectCritic is still on the legacy API and needs a LangChain-wrapped model
            legacy_llm = LangchainLLMWrapper(ChatOpenAI(model=model, temperature=0))
            metrics[name] = AspectCritic(name="answer_rules", definition=ASPECT_DEFINITION, llm=legacy_llm)
            continue
        if llm is None:
            # The default max_tokens=1024 truncates claim extraction for long answers (e.g. full rate lists)
            llm = llm_factory(model, client=AsyncOpenAI(), max_tokens=4096)
        metrics[name] = modern[name](llm=llm)
    return metrics


async def evaluate_all(records: list[dict], enabled: list[str], model: str, concurrency: int) -> list[dict]:
    if not enabled:
        return [{name: "" for name in METRIC_COLUMNS} for _ in records]
    metrics = build_metrics(enabled, model)
    semaphore = asyncio.Semaphore(concurrency)
    return await asyncio.gather(*(evaluate_record(r, metrics, semaphore) for r in records))


def estimate_calls(records: list[dict], enabled: list[str]) -> int:
    return sum(METRIC_CALLS[name](len(r["contexts"])) for r in records for name in enabled)


def next_report_path(folder: Path) -> Path:
    stem = f"ragas-{date.today().strftime('%m%d%Y')}"
    n = 1
    while (folder / f"{stem}-{n}.csv").exists():
        n += 1
    return folder / f"{stem}-{n}.csv"


def write_csv(records: list[dict], metric_rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig lets Excel detect the encoding correctly
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for record, metric_values in zip(records, metric_rows):
            item, result = record["item"], record["result"]
            correctness = result["correctness"]
            writer.writerow({
                "No.": record["number"],
                "id": item["id"],
                "correct": "yes" if correctness >= 0.999 else ("partial" if correctness > 0 else "no"),
                "score": result["score"],
                **metric_values,
                "how_score_calculated": record["how"],
                "question": item["question"],
                "as_of": item.get("as_of") or "",
                "expected_answer": item.get("expected_answer", ""),
                "my_answer": record["answer"].text,
                "category": item.get("category", ""),
                "must_include": join_list(item.get("must_include")),
                "must_not_include": join_list(item.get("must_not_include")),
                "difficulty": item.get("difficulty", ""),
                "weight": item.get("weight", 1.0),
                "expected_sources": join_list(item.get("expected_sources")),
                "group": item.get("group") or "",
            })


def print_summary(records: list[dict], metric_rows: list[dict], enabled: list[str]) -> None:
    weights = sum(r["item"].get("weight", 1.0) for r in records)
    qa = sum(r["result"]["score"] * r["item"].get("weight", 1.0) for r in records) / weights
    print(f"\nqa_score (weighted) = {qa:.4f}")
    if not enabled:
        print("  Ragas metrics: off (use --metrics to turn them on)")
    for column in enabled:
        values = [row[column] for row in metric_rows if not isinstance(row[column], str)]
        errors = len(metric_rows) - len(values)
        mean = sum(values) / len(values) if values else float("nan")
        print(f"  {column:22s} mean={mean:.4f}  (n={len(values)}, errors={errors})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate aisolution with Ragas")
    ap.add_argument("--eval", type=Path, default=ROOT / "eval" / "public_eval.jsonl")
    ap.add_argument("--snapshots", type=Path, default=ROOT / "snapshots")
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--model", default=os.getenv("RAGAS_LLM_MODEL", "gpt-4.1-mini"), help="model used by Ragas for grading")
    ap.add_argument("--concurrency", type=int, default=4, help="number of questions evaluated at once")
    ap.add_argument(
        "--metrics",
        default=os.getenv("RAGAS_METRICS", "none"),
        help=f"Ragas metrics to compute: comma-separated names from {', '.join(METRIC_COLUMNS)}, "
             "or 'all' / 'none' (default: none, or the RAGAS_METRICS environment variable)",
    )
    args = ap.parse_args()
    enabled = parse_metrics(args.metrics)

    items = [json.loads(line) for line in args.eval.read_text(encoding="utf-8").splitlines() if line.strip()]

    solution = load_solution_module()
    global CONTEXT_CHARS
    CONTEXT_CHARS = solution.EVIDENCE_CHARS
    assistant = build_capturing_assistant(solution)
    for snapshot in load_snapshots(args.snapshots):
        print(f"ingest {snapshot.snapshot_id} ({len(snapshot.pages)} pages)", flush=True)
        assistant.ingest(snapshot)

    records = collect_answers(assistant, items)
    if enabled:
        print(f"Ragas metrics on: {', '.join(enabled)} "
              f"(about {estimate_calls(records, enabled)} LLM calls with {args.model})", flush=True)
    else:
        print("Ragas metrics off: no Ragas LLM calls", flush=True)
    metric_rows = asyncio.run(evaluate_all(records, enabled, args.model, args.concurrency))

    out = next_report_path(args.results)
    write_csv(records, metric_rows, out)
    print_summary(records, metric_rows, enabled)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
