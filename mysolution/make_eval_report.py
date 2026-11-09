#!/usr/bin/env python3
"""
Generate a per-question CSV report from the evaluation results.

Reads:
    eval/public_eval.jsonl       questions and expected answers
    results/per_question.jsonl   per-question results written by run_eval.py

Writes:
    results/<MMDDYYYY>-<n>.csv   n starts at 1 and increments if the file already exists

Usage:
    python mysolution/make_eval_report.py
    python mysolution/make_eval_report.py --eval eval/public_eval.jsonl --results results
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))

from scoring import REFUSAL_RE, _numbers  # noqa: E402

COLUMNS = [
    "No.",
    "id",
    "correct",
    "score",
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


def join_list(values) -> str:
    return "; ".join(str(v) for v in values or [])


def explain(item: dict, result: dict) -> tuple[str, float]:
    """Rebuild the calculation using the rules in harness/scoring.py; return (explanation, recomputed score)."""
    answer = result["answer"]
    text = answer.get("text") or ""
    parts: list[str] = []

    if item.get("category") == "should_refuse":
        # Refusal questions: must decline, and the answer may not contain numbers absent from the question
        refused = (not answer.get("confident", True)) or bool(REFUSAL_RE.search(text))
        extra = sorted({f"${v:g}" if u == "$" else f"{v:g}{u}" for u, v in _numbers(text) - _numbers(item["question"])})
        trap = result["trap_hits"]
        correctness = 1.0 if refused and not trap and not extra else 0.0
        parts.append(f"should_refuse: refused={'yes' if refused else 'no'}")
        parts.append(f"numbers not in question={extra if extra else 'none'}")
        if trap:
            parts.append(f"trap hit={trap}")
        parts.append(f"correctness={correctness:g} (needs refusal AND no extra numbers)")
    else:
        must = item.get("must_include", [])
        matched, missed, trap = result["matched"], result["missed"], result["trap_hits"]
        if not must:
            correctness = 0.0
            parts.append("must_include is empty -> scorer always gives correctness=0")
        else:
            correctness = len(matched) / len(must)
            parts.append(f"must_include matched {len(matched)}/{len(must)} -> correctness={correctness:.4g}")
            if missed:
                parts.append(f"missed={missed}")
        if trap:
            correctness = 0.0
            parts.append(f"must_not_include hit={trap} -> correctness=0")

    if item.get("expected_sources"):
        src = result["source_hit"]
        parts.append(f"source={src:g} ({'cited an expected source' if src else 'no expected source cited'})")
        score = 0.85 * correctness + 0.15 * src
        parts.append(f"score = 0.85*{correctness:.4g} + 0.15*{src:g} = {score:.4f}")
    else:
        score = correctness
        parts.append(f"no expected_sources -> score = correctness = {score:.4f}")

    return "; ".join(parts), score


def next_report_path(folder: Path) -> Path:
    stem = date.today().strftime("%m%d%Y")
    n = 1
    while (folder / f"{stem}-{n}.csv").exists():
        n += 1
    return folder / f"{stem}-{n}.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a per-question evaluation CSV report")
    ap.add_argument("--eval", type=Path, default=ROOT / "eval" / "public_eval.jsonl")
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    items = [json.loads(line) for line in args.eval.read_text(encoding="utf-8").splitlines() if line.strip()]
    results_file = args.results / "per_question.jsonl"
    by_id = {
        r["id"]: r
        for r in (json.loads(line) for line in results_file.read_text(encoding="utf-8").splitlines() if line.strip())
    }

    rows = []
    for number, item in enumerate(items, start=1):
        result = by_id[item["id"]]
        how, recomputed = explain(item, result)
        # The recomputed score must match the scorer; otherwise the rules were misread
        if abs(recomputed - result["score"]) > 1e-3:
            raise SystemExit(f"{item['id']}: recomputed {recomputed:.4f} != scorer {result['score']}")

        correctness = result["correctness"]
        correct = "yes" if correctness >= 0.999 else ("partial" if correctness > 0 else "no")
        rows.append({
            "No.": number,
            "id": item["id"],
            "correct": correct,
            "score": result["score"],
            "how_score_calculated": how,
            "question": item["question"],
            "as_of": item.get("as_of") or "",
            "expected_answer": item.get("expected_answer", ""),
            "my_answer": result["answer"].get("text", ""),
            "category": item.get("category", ""),
            "must_include": join_list(item.get("must_include")),
            "must_not_include": join_list(item.get("must_not_include")),
            "difficulty": item.get("difficulty", ""),
            "weight": item.get("weight", 1.0),
            "expected_sources": join_list(item.get("expected_sources")),
            "group": item.get("group") or "",
        })

    out = next_report_path(args.results)
    # utf-8-sig lets Excel detect the encoding correctly
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    total = sum(1 for r in rows if r["correct"] == "yes")
    print(f"Wrote {out} ({len(rows)} questions, {total} fully correct)")


if __name__ == "__main__":
    main()
