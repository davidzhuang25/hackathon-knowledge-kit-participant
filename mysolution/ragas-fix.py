#!/usr/bin/env python3
"""
Add "max score" and "fix suggestion" columns to a Ragas CSV report.

Usage (from the project root):
    python mysolution/ragas-fix.py results/ragas-11072026-1.csv            # harness-based tips only
    python mysolution/ragas-fix.py results/ragas-11072026-1.csv --metrics all
    python mysolution/ragas-fix.py results/ragas-11072026-1.csv --metrics context_recall,faithfulness

Ragas metric hints are off by default. Turn them on with --metrics (comma-separated names,
"all" or "none") or the RAGAS_METRICS environment variable. A metric that is on but has an
empty cell in the CSV (it was not computed by ragas.py) is ignored.

Output:
    results/ragas-11072026-1-to-fix.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

METRICS = [
    "context_recall",
    "context_entity_recall",
    "context_precision",
    "faithfulness",
    "noise_sensitivity",
    "factual_correctness",
    "aspect_critic",
]

REQUIRED_COLUMNS = {
    "correct",
    "score",
    "how_score_calculated",
    "category",
    "must_include",
    "weight",
    "expected_sources",
}

LOW = 0.5
FAITHFUL_LOW = 0.7

# Metrics used for fix hints; set from --metrics in main(). Empty means metric hints are off.
ENABLED: set[str] = set()


def parse_metrics(value: str) -> set[str]:
    """Turn 'all', 'none' or a comma-separated list into metric names."""
    names = {name.strip().lower() for name in (value or "").split(",") if name.strip()}
    if not names or names == {"none"}:
        return set()
    if "all" in names:
        return set(METRICS)
    unknown = sorted(names - set(METRICS))
    if unknown:
        raise SystemExit(
            f"Unknown Ragas metric(s): {', '.join(unknown)}. "
            f"Choose from: {', '.join(METRICS)}, all, none."
        )
    return names


def metric(row: dict[str, str], name: str) -> float | None:
    """Return a metric value, or None when the metric is off or Ragas did not produce one."""
    if name not in ENABLED:
        return None
    value = row.get(name, "").strip()
    if not value or value.startswith("ERROR"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def maximum_score(row: dict[str, str]) -> float:
    """Return the highest item score allowed by the harness rules."""
    if row["category"] != "should_refuse" and not row["must_include"].strip():
        return 0.0
    return 1.0


def extract_list(text: str, label: str) -> str | None:
    match = re.search(rf"{re.escape(label)}=(\[[^\]]*\])", text)
    return match.group(1) if match else None


def suggestion(row: dict[str, str], max_score: float, total_weight: float) -> str:
    score = float(row["score"])
    if max_score == 0:
        return "Skip: this non-refusal question has an empty must_include list, so the scorer always gives 0."
    if score >= max_score - 1e-9:
        return "No fix needed: the question already has its maximum score."

    how = row["how_score_calculated"]
    gain = (max_score - score) * float(row["weight"]) / total_weight
    tips = [f"Potential gain: +{gain:.4f} weighted qa_score (weight {row['weight']})."]

    recall = metric(row, "context_recall")
    entity_recall = metric(row, "context_entity_recall")
    precision = metric(row, "context_precision")
    faithfulness = metric(row, "faithfulness")
    noise = metric(row, "noise_sensitivity")
    factual = metric(row, "factual_correctness")
    critic = metric(row, "aspect_critic")
    # Only enabled metrics whose cell holds an error count as errors; empty cells were not computed
    errors = [name for name in METRICS
              if name in ENABLED and row.get(name, "").strip().startswith("ERROR")]

    if row["correct"] == "yes" and "source=0" in how:
        tips.append(
            "SOURCE MISS: the answer is correct, but no expected source was cited. "
            f"Expected sources: {row['expected_sources'] or '(none)'}. Inspect Answer.sources and ensure "
            "the supporting official page survives retrieval and is included among the returned sources."
        )
        if precision is not None and precision < LOW:
            tips.append(
                f"Context precision is low ({precision:g}); re-rank official product/rate pages above "
                "blogs, calculators and unrelated chunks."
            )
        return " ".join(tips)

    if row["category"] == "should_refuse":
        if "refused=no" in how:
            tips.append(
                "REFUSAL: the evidence has no explicit value for the exact item, but the answer did not decline. "
                "Require confident=false when the exact branch, product, rate or fee is not stated."
            )
        numbers = extract_list(how, "numbers not in question")
        if numbers:
            tips.append(
                f"REFUSAL NUMBERS: the answer added {numbers}. A refusal must not state numbers that are absent "
                "from the question; omit related terms, caps, ranges and phone numbers."
            )
        if critic == 0:
            tips.append("Aspect critic is 0, confirming that the refusal behavior broke the assistant rules.")
        if errors:
            tips.append(f"Metric errors: {errors}; rerun ragas.py to replace the missing values.")
        return " ".join(tips)

    trap = extract_list(how, "must_not_include hit")
    if trap:
        tips.append(
            f"STALE OR WRONG VALUE: the answer included {trap}. Inspect publication/effective dates and product "
            "identity; prefer the newest relevant article or the official current product/rate page."
        )

    missed = extract_list(how, "missed")
    if missed:
        tips.append(f"MISSING ANSWER VALUES: {missed}.")

    if recall is not None and recall < LOW:
        detail = f"context_recall={recall:g}"
        if entity_recall is not None:
            detail += f", context_entity_recall={entity_recall:g}"
        tips.append(
            f"RETRIEVAL ({detail}): expected facts are absent from the retrieved context. Use mysolution/debug.py to "
            "check whether they rank below the retrieval limit, were removed during duplicate merging, or do not "
            "exist in the selected snapshot."
        )
    elif precision is not None and precision < LOW:
        tips.append(
            f"RANKING (context_precision={precision:g}): useful evidence exists, but irrelevant chunks rank too "
            "high. Inspect similarity distance, duplicate groups, product identity and official-page priority."
        )

    if faithfulness is not None and faithfulness < FAITHFUL_LOW:
        tips.append(
            f"GROUNDING (faithfulness={faithfulness:g}): the answer contains claims not supported by the context. "
            "Tighten the evidence-only prompt and check whether facts from different products were combined."
        )
    if noise is not None and noise >= LOW:
        tips.append(
            f"NOISY EVIDENCE (noise_sensitivity={noise:g}, lower is better): wrong claims came from retrieved "
            "chunks. Remove stale, duplicate or different-product evidence before generation."
        )
    if factual is not None and factual < LOW and (recall is None or recall >= LOW) \
            and (faithfulness is None or faithfulness >= FAITHFUL_LOW):
        tips.append(
            f"GENERATION (factual_correctness={factual:g}): the evidence was available and the answer was mostly "
            "grounded, but it omitted or changed expected facts. Require all relevant tiers, terms, names and "
            "yes/no supporting values."
        )

    if row["category"] in {"stale_trap", "conflict_resolution", "temporal_as_of"} \
            and not trap and (recall is None or recall >= LOW):
        tips.append(
            "DATE CHECK: compare publication and effective dates in the retrieved chunks. The newest matching "
            "article or official current page should win; do not use the snapshot date as a publication date."
        )
    if critic == 0:
        tips.append(
            "RULE COMPLIANCE: aspect_critic=0. Check completeness, product/tier disambiguation, stale-value "
            "handling and refusal behavior."
        )
    if errors:
        tips.append(f"Metric errors: {errors}; rerun ragas.py to replace the missing values.")

    return " ".join(tips)


def read_report(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Ragas report not found: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - set(fields))
        if missing:
            raise ValueError(f"Missing required CSV columns: {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("The Ragas report contains no data rows.")
    return fields, rows


def write_report(source: Path, fields: list[str], rows: list[dict[str, str]]) -> Path:
    position = fields.index("score") + 1
    output_fields = fields[:position] + ["max score", "fix suggestion"] + fields[position:]
    total_weight = sum(float(row["weight"]) for row in rows)

    for row in rows:
        max_score = maximum_score(row)
        row["max score"] = f"{max_score:g}"
        row["fix suggestion"] = suggestion(row, max_score, total_weight)

    output = source.with_name(f"{source.stem}-to-fix.csv")
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add max score and metric-based fix suggestions to a Ragas CSV report."
    )
    parser.add_argument("report", type=Path, help="input Ragas CSV, normally under results/")
    parser.add_argument(
        "--metrics",
        default=os.getenv("RAGAS_METRICS", "none"),
        help=f"Ragas metrics to use in fix hints: comma-separated names from {', '.join(METRICS)}, "
             "or 'all' / 'none' (default: none, or the RAGAS_METRICS environment variable)",
    )
    args = parser.parse_args()
    ENABLED.clear()
    ENABLED.update(parse_metrics(args.metrics))

    fields, rows = read_report(args.report)
    missing = sorted(name for name in ENABLED if name not in fields)
    if missing:
        print(f"Note: metric column(s) not in the CSV, ignored: {', '.join(missing)}")
    print(f"Ragas metric hints: {', '.join(sorted(ENABLED)) if ENABLED else 'off'}")
    output = write_report(args.report, fields, rows)

    total_weight = sum(float(row["weight"]) for row in rows)
    fixable = [row for row in rows if float(row["score"]) < float(row["max score"]) - 1e-9]
    gain = sum(
        (float(row["max score"]) - float(row["score"])) * float(row["weight"])
        for row in fixable
    ) / total_weight
    skipped = sum(float(row["max score"]) == 0 for row in rows)

    print(f"Wrote {output}")
    print(f"Questions: {len(rows)}; fixable: {len(fixable)}; skipped: {skipped}")
    print(f"Maximum potential weighted qa_score gain: +{gain:.4f}")


if __name__ == "__main__":
    main()
