#!/usr/bin/env python3
"""
Trace how aisolution/solution.py handles one question from the eval set.

Usage (from the project root):
    python tools/debug.py 7            # the 7th question in eval/public_eval.jsonl (1-based)
    python tools/debug.py dup-073      # the question with id "dup-073"
    python tools/debug.py dup-049 --repeat 3 --deep-k 80

The trace is printed to the console and written to results/trace-<id>-MMDDYYYY-n.md.
solution.py is not modified; retrieval and the model call are captured through a subclass.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "tools"))

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict  # noqa: E402
from datetime import date  # noqa: E402

from dotenv import load_dotenv  # noqa: E402
from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

from make_eval_report import explain, join_list  # noqa: E402
from run_eval import load_snapshots, to_answer  # noqa: E402
from scoring import REFUSAL_RE, _numbers, _url_key, contains, score_item  # noqa: E402

load_dotenv(ROOT / ".env", override=False)

# Must match the truncation applied in solution.KnowledgeAssistant._retrieve_and_prepare
# Replaced in main() with solution.EVIDENCE_CHARS, the truncation applied before text reaches the model
CONTEXT_CHARS = 1400
SPEED_TARGET_SECONDS = 2.0


def cut_marker() -> str:
    return f"\n----- MODEL VIEW ENDS HERE ({CONTEXT_CHARS:,} chars); text below is NOT sent to the model -----\n"
FENCE = "````"


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

class Report:
    """Collect Markdown lines, echoing each one to the console as it is added."""

    def __init__(self):
        self.lines: list[str] = []

    def add(self, text: str = "") -> None:
        print(text, flush=True)
        self.lines.append(text)

    def heading(self, text: str) -> None:
        self.add()
        self.add(f"## {text}")
        self.add()

    def block(self, text: str) -> None:
        self.add(FENCE)
        self.add(text)
        self.add(FENCE)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def next_trace_path(folder: Path, item_id: str) -> Path:
    stem = f"trace-{item_id}-{date.today().strftime('%m%d%Y')}"
    n = 1
    while (folder / f"{stem}-{n}.md").exists():
        n += 1
    return folder / f"{stem}-{n}.md"


def show(value) -> str:
    if value is None or value == "" or value == []:
        return "(none)"
    if isinstance(value, list):
        return join_list(value)
    return str(value)


# ---------------------------------------------------------------------------
# Hooks into solution.py
# ---------------------------------------------------------------------------

def load_solution_module():
    spec = importlib.util.spec_from_file_location("solution", ROOT / "aisolution" / "solution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ChatCapture(BaseCallbackHandler):
    """Record the messages sent to the chat model and the model call duration."""

    def __init__(self):
        self.messages = None
        self.started = None
        self.seconds = None

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.messages = messages[0] if messages else None
        self.started = time.perf_counter()

    def on_llm_end(self, response, **kwargs):
        if self.started is not None:
            self.seconds = time.perf_counter() - self.started


class PipelineProbe:
    """Stand-in for the assistant's answer pipeline: adds the capture callback and keeps the raw result."""

    def __init__(self, pipeline, owner):
        self._pipeline = pipeline
        self._owner = owner

    def invoke(self, request, config=None):
        config = dict(config or {})
        config["callbacks"] = list(config.get("callbacks") or []) + [self._owner.chat_capture]
        result = self._pipeline.invoke(request, config=config)
        self._owner.last_pipeline_result = result
        return result


def build_tracing_assistant(solution):
    class TracingAssistant(solution.KnowledgeAssistant):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._answer_pipeline = PipelineProbe(self._answer_pipeline, self)
            self.reset_capture()

        def reset_capture(self) -> None:
            self.last_retrieval = None
            self.last_pipeline_result = None
            self.retrieval_seconds = None
            self.chat_capture = ChatCapture()

        def _retrieve_and_prepare(self, request: dict) -> dict:
            started = time.perf_counter()
            prepared = super()._retrieve_and_prepare(request)
            self.retrieval_seconds = time.perf_counter() - started
            self.last_retrieval = prepared
            return prepared

    return TracingAssistant()


def run_once(assistant, item: dict) -> dict:
    assistant.reset_capture()
    started = time.perf_counter()
    raw = assistant.answer(item["question"], item.get("as_of"))
    total = time.perf_counter() - started
    answer = to_answer(raw)
    result = score_item(item, answer)
    result["answer"] = asdict(answer)
    pipeline = assistant.last_pipeline_result or {}
    return {
        "answer": answer,
        "result": result,
        "draft": pipeline.get("draft"),
        "retrieval": assistant.last_retrieval,
        "messages": assistant.chat_capture.messages,
        "retrieval_seconds": assistant.retrieval_seconds,
        "llm_seconds": assistant.chat_capture.seconds,
        "total_seconds": total,
    }


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def token_locations(token: str, documents) -> dict:
    """Ranks of retrieved chunks containing the token, split by visibility to the model."""
    visible, after_cut = [], []
    for rank, document in enumerate(documents, start=1):
        text = document.page_content
        if contains(text[:CONTEXT_CHARS], token):
            visible.append(rank)
        elif contains(text, token):
            after_cut.append(rank)
    return {"visible": visible, "after_cut": after_cut}


def money_and_rate_values(text: str) -> set[str]:
    return {f"${v:g}" if unit == "$" else f"{v:g}%" for unit, v in _numbers(text) if unit in ("$", "%")}


def deep_search(vector_store, question: str, snapshot_id: str, token: str, deep_k: int) -> dict:
    hits = vector_store.similarity_search_with_score(question, k=deep_k, filter={"snapshot_id": snapshot_id})
    first = None
    for rank, (document, distance) in enumerate(hits, start=1):
        if contains(document.page_content, token):
            first = (rank, distance, document.metadata)
            break
    stored = vector_store.get(where={"snapshot_id": snapshot_id}, include=["documents", "metadatas"])
    pages: dict[str, dict] = {}
    for text, metadata in zip(stored["documents"], stored["metadatas"]):
        if contains(text, token):
            entry = pages.setdefault(metadata.get("url", ""), {"metadata": metadata, "chunks": 0})
            entry["chunks"] += 1
    return {"first": first, "pages": pages, "deep_k": len(hits)}


def load_page_files(snapshots_root: Path) -> dict[str, tuple[str, dict[str, str]]]:
    """Map snapshot_id -> (snapshot folder name, {url: page file name})."""
    mapping = {}
    for folder in sorted(p for p in snapshots_root.iterdir() if (p / "manifest.json").exists()):
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        files = {page["url"]: Path(page["file"]).name for page in manifest["pages"]}
        mapping[manifest["snapshot_id"]] = (folder.name, files)
    return mapping


def tokens_seen(text: str, item: dict) -> str:
    """must_include / must_not_include tokens found in a chunk, marking those only after the cut."""
    seen = []
    for label, tokens in (("", item.get("must_include", [])), ("TRAP ", item.get("must_not_include", []))):
        for token in tokens:
            if contains(text[:CONTEXT_CHARS], token):
                seen.append(f"{label}`{token}`")
            elif contains(text, token):
                seen.append(f"{label}`{token}` (after cut)")
    return ", ".join(seen) or "-"


SAYS_PROMPT = (
    "Using only the text below, answer the question in at most 20 words. Quote exact values "
    "(numbers, dates, rates, hours) as written. If the text does not answer the question, reply "
    "exactly: (no answer)\n\nQuestion: {question}\n\nText:\n{text}"
)


def chunk_says(question: str, documents) -> list[str]:
    """Ask the chat model what each chunk alone answers, using only the part the model sees."""
    if not documents:
        return []
    model = ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"), temperature=0, timeout=25, max_retries=2)
    prompts = [SAYS_PROMPT.format(question=question, text=d.page_content[:CONTEXT_CHARS]) for d in documents]
    replies = model.batch(prompts)
    return [re.sub(r"\s+", " ", reply.content).strip() for reply in replies]


def table_cell(text: str) -> str:
    return text.replace("|", "\\|")


def check_index_state(solution, snapshots) -> list[str]:
    state_path = ROOT / "aisolution" / ".chroma" / "index_state.json"
    signature = f"{solution.INDEX_VERSION}:{os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')}"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    indexed = state.get("indexed", {}) if state.get("signature") == signature else {}
    return [s.snapshot_id for s in snapshots if indexed.get(s.snapshot_id) != s.taken_on]


def diagnose(item: dict, run: dict, statuses: dict, trap_in_evidence: list[str]) -> str:
    result = run["result"]
    if item.get("category") == "should_refuse":
        return "OK" if result["correctness"] >= 0.999 else "refusal issue (the answer did not decline cleanly or stated numbers)"
    if not item.get("must_include"):
        return "cannot score: this item has no must_include tokens, so its max score is 0"
    missed, traps = result["missed"], result["trap_hits"]
    if not missed and not traps:
        if result["source_hit"] == 0:
            return "OK (correct answer, but no expected source URL was cited)"
        return "OK"
    problem = {token: statuses.get(token) for token in missed}
    if any(status == "not_in_snapshot" for status in problem.values()):
        return "fact not in snapshot: " + join_list(t for t, s in problem.items() if s == "not_in_snapshot")
    if any(status == "below_k" for status in problem.values()):
        return "retrieval miss: " + join_list(t for t, s in problem.items() if s == "below_k")
    if any(status == "after_cut" for status in problem.values()):
        return "lost in truncation: " + join_list(t for t, s in problem.items() if s == "after_cut")
    if traps and set(traps) & set(trap_in_evidence):
        return "stale evidence: " + join_list(sorted(set(traps) & set(trap_in_evidence)))
    detail = []
    if missed:
        detail.append("missing " + join_list(missed))
    if traps:
        detail.append("stated trap value " + join_list(traps))
    return "generation error (evidence was visible to the model): " + "; ".join(detail)


# ---------------------------------------------------------------------------
# Main trace
# ---------------------------------------------------------------------------

def resolve_item(items: list[dict], key: str) -> tuple[int, dict] | None:
    if key.isdigit():
        number = int(key)
        return (number, items[number - 1]) if 1 <= number <= len(items) else None
    for number, item in enumerate(items, start=1):
        if item["id"].lower() == key.lower():
            return number, item
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Trace how aisolution answers one eval question")
    ap.add_argument("question", help="question number (1-based) or id such as dup-073")
    ap.add_argument("--eval", type=Path, default=ROOT / "eval" / "public_eval.jsonl")
    ap.add_argument("--snapshots", type=Path, default=ROOT / "snapshots")
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--repeat", type=int, default=1, help="run the same question N times to check answer stability")
    ap.add_argument("--deep-k", type=int, default=50, help="k for the deep search when a fact is not retrieved")
    args = ap.parse_args()

    items = [json.loads(line) for line in args.eval.read_text(encoding="utf-8").splitlines() if line.strip()]
    resolved = resolve_item(items, args.question)
    if resolved is None:
        print(f"Error: '{args.question}' is not a number from 1 to {len(items)} or a known id.", file=sys.stderr)
        print("Valid ids:", file=sys.stderr)
        for number, item in enumerate(items, start=1):
            print(f"  {number:2d}  {item['id']}", file=sys.stderr)
        sys.exit(2)
    number, item = resolved

    solution = load_solution_module()
    global CONTEXT_CHARS
    CONTEXT_CHARS = solution.EVIDENCE_CHARS
    snapshots = load_snapshots(args.snapshots)
    not_indexed = check_index_state(solution, snapshots)
    if not_indexed:
        print(f"WARNING: not indexed yet, ingest will call the embedding API: {', '.join(not_indexed)}", flush=True)
    assistant = build_tracing_assistant(solution)
    for snapshot in snapshots:
        print(f"ingest {snapshot.snapshot_id} ({len(snapshot.pages)} pages)", flush=True)
        assistant.ingest(snapshot)

    selected = solution.select_snapshot({s.snapshot_id: s.taken_on for s in snapshots}, item.get("as_of"))
    run = run_once(assistant, item)
    runs = [run]
    documents = run["retrieval"]["documents"] if run["retrieval"] else []
    vector_store = assistant._vector_store

    report = Report()
    report.add(f"# Trace for {item['id']} (question {number} of {len(items)})")

    # 1. Question (includes the generated answer, so this section is written after answer())
    report.heading("1. Question")
    for label, key in [("id", "id"), ("question", "question"), ("as_of", "as_of"), ("category", "category"),
                       ("difficulty", "difficulty"), ("weight", "weight"), ("group", "group"),
                       ("expected_answer", "expected_answer"), ("must_include", "must_include"),
                       ("must_not_include", "must_not_include"), ("expected_sources", "expected_sources")]:
        report.add(f"- **{label}**: {show(item.get(key))}")
    report.add(f"- **my_answer**: {show(run['answer'].text)}")
    # Sources returned with the answer, marked when they match an expected source (same URL rule as the scorer)
    expected_keys = {_url_key(url) for url in item.get("expected_sources") or []}
    my_sources = run["answer"].sources or []
    report.add("- **my_sources**:" + ("" if my_sources else " (none)"))
    for position, source in enumerate(my_sources, start=1):
        if not expected_keys:
            mark = ""
        elif _url_key(source.url) in expected_keys:
            mark = "  [matches expected source]"
        else:
            mark = "  [not an expected source]"
        report.add(f"  {position}. {source.url} ({source.snapshot_id}){mark}")

    # 2. Snapshot selection
    report.heading("2. Snapshot selection")
    for snapshot in snapshots:
        report.add(f"- {snapshot.snapshot_id}: taken on {snapshot.taken_on}")
    report.add(f"- **as_of**: {item.get('as_of') or 'none = latest'}")
    report.add(f"- **selected**: {selected[0] + ' (' + selected[1] + ')' if selected else '(no snapshot on or before as_of)'}")

    # 3. Retrieved chunks
    report.heading(f"3. Retrieved chunks (k={solution.RETRIEVAL_K} from {solution.RETRIEVAL_CANDIDATES} "
                   f"candidates after merging duplicates, {len(documents)} returned)")
    # Distances come from the assistant's own retrieval; a merged chunk carries its group's best distance
    captured = run["retrieval"].get("distances", []) if run["retrieval"] else []
    distances = {
        (d.metadata.get("url"), d.metadata.get("section"), d.metadata.get("chunk_index")): distance
        for d, distance in zip(documents, captured)
    }

    page_files = load_page_files(args.snapshots)
    says = chunk_says(item["question"], documents)
    report.add("Says = what the chunk alone answers (model view only, first "
               f"{CONTEXT_CHARS:,} chars); Tokens seen = must_include / TRAP must_not_include values in the chunk.")
    report.add()
    report.add("| Rank | Distance | Published | Says | Tokens seen | Snapshot folder | File name |")
    report.add("|---|---|---|---|---|---|---|")
    for rank, document in enumerate(documents, start=1):
        metadata = document.metadata
        key = (metadata.get("url"), metadata.get("section"), metadata.get("chunk_index"))
        distance = distances.get(key)
        folder, files = page_files.get(metadata.get("snapshot_id"), ("(unknown)", {}))
        report.add(f"| {rank} | {f'{distance:.4f}' if distance is not None else '-'} | "
                   f"{metadata.get('published_on') or '(none)'} | {table_cell(says[rank - 1])} | "
                   f"{tokens_seen(document.page_content, item)} | "
                   f"`{folder}` | `{files.get(metadata.get('url'), '(not in manifest)')}` |")
    report.add()

    for rank, document in enumerate(documents, start=1):
        metadata = document.metadata
        text = document.page_content
        key = (metadata.get("url"), metadata.get("section"), metadata.get("chunk_index"))
        distance = distances.get(key)
        report.add(f"### Rank {rank}  (distance {distance:.4f}, lower is closer)" if distance is not None else f"### Rank {rank}")
        report.add()
        for field in ["snapshot_id", "snapshot_date", "url", "title", "section", "chunk_index", "published_on"]:
            report.add(f"- **{field}**: {show(metadata.get(field))}")
        extra = sorted(set(metadata) - {"snapshot_id", "snapshot_date", "url", "title", "section", "chunk_index", "published_on"})
        for field in extra:
            report.add(f"- **{field}**: {show(metadata.get(field))}")
        folder, files = page_files.get(metadata.get("snapshot_id"), ("(unknown)", {}))
        report.add(f"- **says**: {says[rank - 1]}")
        report.add(f"- **file**: snapshots/{folder}/pages/{files.get(metadata.get('url'), '(not in manifest)')}")
        cut = len(text) > CONTEXT_CHARS
        report.add(f"- **length**: {len(text):,} chars; "
                   f"{'TRUNCATED, the model sees only the first ' + format(CONTEXT_CHARS, ',') if cut else 'not truncated'}")
        report.add()
        report.block(text[:CONTEXT_CHARS] + cut_marker() + text[CONTEXT_CHARS:] if cut else text)
        report.add()

    # 4. Evidence check
    report.heading("4. Evidence check")
    statuses: dict[str, str] = {}
    report.add("**must_include tokens**")
    report.add()
    for token in item.get("must_include", []):
        where = token_locations(token, documents)
        if where["visible"]:
            statuses[token] = "visible"
            text = f"visible in ranks {where['visible']}"
            if where["after_cut"]:
                text += f"; only after the cut in ranks {where['after_cut']}"
        elif where["after_cut"]:
            statuses[token] = "after_cut"
            text = f"ONLY AFTER THE {CONTEXT_CHARS:,}-CHAR CUT in ranks {where['after_cut']} (model cannot see it)"
        else:
            statuses[token] = "not_retrieved"
            text = "NOT in any retrieved chunk"
        report.add(f"- `{token}`: {text}")
    if not item.get("must_include"):
        report.add("- (none)")
    report.add()
    report.add("**must_not_include tokens (trap / stale values)**")
    report.add()
    trap_in_evidence = []
    for token in item.get("must_not_include", []):
        where = token_locations(token, documents)
        if where["visible"]:
            trap_in_evidence.append(token)
            report.add(f"- `{token}`: present in evidence the model sees, ranks {where['visible']}")
        elif where["after_cut"]:
            report.add(f"- `{token}`: only after the cut, ranks {where['after_cut']}")
        else:
            report.add(f"- `{token}`: not in the evidence")
    if not item.get("must_not_include"):
        report.add("- (none)")
    report.add()
    report.add("**expected sources**")
    report.add()
    retrieved_urls = [_url_key(d.metadata.get("url", "")) for d in documents]
    for url in item.get("expected_sources", []):
        ranks = [rank for rank, key in enumerate(retrieved_urls, start=1) if key == _url_key(url)]
        report.add(f"- {url}: {'retrieved at ranks ' + str(ranks) if ranks else 'NOT retrieved'}")
    if not item.get("expected_sources"):
        report.add("- (none)")
    report.add()
    report.add("**publish dates in the evidence**")
    report.add()
    by_date: dict[str, set[str]] = {}
    for document in documents:
        published = document.metadata.get("published_on") or ""
        by_date.setdefault(published, set()).update(money_and_rate_values(document.page_content[:CONTEXT_CHARS]))
    for published in sorted(by_date, reverse=True):
        values = sorted(by_date[published])
        shown = ", ".join(values[:15]) + (" ..." if len(values) > 15 else "")
        report.add(f"- {published or '(no publish date)'}: {shown or '(no $ or % values)'}")
    dated = {d: v for d, v in by_date.items() if d}
    if len(dated) > 1 and len({frozenset(v) for v in dated.values()}) > 1:
        report.add()
        report.add("> FLAG: evidence from different publish dates carries different $/% values; check that the latest one is used.")

    # 5. Deep search
    report.heading("5. Deep search")
    missing = [t for t, s in statuses.items() if s == "not_retrieved"]
    if not missing:
        report.add("Not needed: every must_include token appears in the retrieved chunks.")
    for token in missing:
        found = deep_search(vector_store, item["question"], selected[0], token, args.deep_k) if selected else None
        report.add(f"### Token `{token}`")
        report.add()
        if not found:
            statuses[token] = "not_in_snapshot"
            report.add("- no snapshot selected")
            continue
        if found["first"]:
            rank, distance, metadata = found["first"]
            report.add(f"- first match in top {found['deep_k']}: rank {rank} (distance {distance:.4f}), "
                       f"{metadata.get('url')} | section: {metadata.get('section')} | "
                       f"published_on: {show(metadata.get('published_on'))}")
        else:
            report.add(f"- no match in the top {found['deep_k']} results")
        pages = found["pages"]
        statuses[token] = "below_k" if pages else "not_in_snapshot"
        report.add(f"- pages in {selected[0]} containing it: {len(pages)}")
        for url, entry in list(pages.items())[:10]:
            metadata = entry["metadata"]
            report.add(f"  - {url} ({entry['chunks']} chunk(s), published_on: {show(metadata.get('published_on'))})")
        report.add()

    # 6. Model input
    report.heading("6. Model input")
    messages = run["messages"] or []
    total_chars = 0
    for message in messages:
        content = message.content if isinstance(message.content, str) else json.dumps(message.content)
        total_chars += len(content)
        report.add(f"**{message.type} message** ({len(content):,} chars)")
        report.add()
        report.block(content)
        report.add()
    if not messages:
        report.add("(the model was not called)")
    report.add(f"Approximate input tokens: {total_chars // 4:,} (characters / 4)")

    # 7. Model output
    report.heading("7. Model output (raw DraftAnswer)")
    draft = run["draft"]
    if draft is not None:
        report.add(f"- **text**: {draft.text}")
        report.add(f"- **confident**: {draft.confident}")
        report.add(f"- **conflict_note**: {show(draft.conflict_note)}")
    else:
        report.add("(no draft; answer() returned before calling the model)")

    # 8. Final answer
    report.heading("8. Final answer after post-processing")
    answer = run["answer"]
    prepended = draft is not None and answer.text.startswith("I don't know.") and not draft.text.strip().startswith("I don't know.")
    report.add(f"- **text**: {answer.text}")
    report.add(f"- **\"I don't know.\" prepended by answer()**: {'yes' if prepended else 'no'}")
    report.add(f"- **confident**: {answer.confident}")
    report.add(f"- **conflict_note**: {show(answer.conflict_note)}")
    report.add("- **sources**:")
    for source in answer.sources or []:
        quote = re.sub(r"\s+", " ", source.quote or "")[:120]
        report.add(f"  - {source.url} ({source.snapshot_id}): \"{quote}\"")
    if not answer.sources:
        report.add("  - (none)")

    # 9. Score
    report.heading("9. Score")
    result = run["result"]
    refused = (not answer.confident) or bool(REFUSAL_RE.search(answer.text or ""))
    report.add(f"- **correctness**: {result['correctness']}")
    report.add(f"- **source score**: {'n/a (no expected sources)' if result['source_hit'] is None else result['source_hit']}")
    report.add(f"- **item score**: {result['score']} (weight {item.get('weight', 1.0)})")
    report.add(f"- **refusal detected**: {'yes' if refused else 'no'}")
    if item.get("category") == "should_refuse":
        extra_numbers = _numbers(answer.text) - _numbers(item["question"])
        report.add(f"- **numbers not in the question** (must be none): {sorted(v for _, v in extra_numbers) or '(none)'}")
    report.add(f"- **must_include found**: {show(result['matched'])}")
    report.add(f"- **must_include missing**: {show(result['missed'])}")
    report.add(f"- **must_not_include triggered**: {show(result['trap_hits'])}")
    how, _ = explain(item, result)
    report.add(f"- **explanation**: {how}")

    # 10. Timing
    report.heading("10. Timing")
    retrieval_s, llm_s, total_s = run["retrieval_seconds"], run["llm_seconds"], run["total_seconds"]
    report.add(f"- retrieval: {retrieval_s:.2f} s" if retrieval_s is not None else "- retrieval: (not run)")
    report.add(f"- LLM: {llm_s:.2f} s" if llm_s is not None else "- LLM: (not run)")
    report.add(f"- total answer(): {total_s:.2f} s "
               f"({'within' if total_s <= SPEED_TARGET_SECONDS else 'OVER'} the {SPEED_TARGET_SECONDS:g} s p95 target)")

    # 11. Repeat runs
    report.heading("11. Repeat runs")
    if args.repeat <= 1:
        report.add("Not requested (use --repeat N to check answer stability).")
    else:
        for _ in range(args.repeat - 1):
            runs.append(run_once(assistant, item))
        for index, extra in enumerate(runs, start=1):
            report.add(f"- run {index}: score {extra['result']['score']}, {extra['total_seconds']:.2f} s, "
                       f"missing {show(extra['result']['missed'])}")
            flat = re.sub(r"\s+", " ", extra["answer"].text)
            report.add(f"  - {flat}")
        scores = [r["result"]["score"] for r in runs]
        distinct = len({re.sub(r"\s+", " ", r["answer"].text.strip().lower()) for r in runs})
        report.add()
        report.add(f"Distinct answers: {distinct} of {len(runs)}; score min {min(scores)}, "
                   f"mean {sum(scores) / len(scores):.4f}, max {max(scores)}")

    # 12. Diagnosis
    report.heading("12. Diagnosis")
    report.add(f"**{diagnose(item, run, statuses, trap_in_evidence)}**")

    out = next_trace_path(args.results, item["id"])
    report.write(out)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
