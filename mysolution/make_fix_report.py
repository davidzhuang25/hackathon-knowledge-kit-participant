#!/usr/bin/env python3
"""
Add "max score" and "fix suggestion" columns to a Ragas evaluation report.

Usage:
    python mysolution/make_fix_report.py results/ragas-11062026-1.csv
    -> writes results/ragas-11062026-1-to-fix.csv
"""
from __future__ import annotations

import argparse
import csv
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

# Metric thresholds: a value below these (above, for noise sensitivity) flags that stage as a problem
LOW = 0.5
FAITHFUL_LOW = 0.7


def metric(row: dict, name: str) -> float | None:
    value = row.get(name, "")
    if value == "" or value.startswith("ERROR"):
        return None
    return float(value)


def max_score(row: dict) -> float:
    # For non-refusal questions with an empty must_include, the scorer always gives 0, however good the answer
    if row["category"] != "should_refuse" and not row["must_include"].strip():
        return 0.0
    return 1.0


def suggest(row: dict, max_s: float, total_weight: float) -> str:
    score = float(row["score"])
    if max_s == 0.0:
        return "Skip: must_include is empty, the scorer always gives 0 for this question."
    if score >= max_s - 1e-9:
        return "No fix needed: already at the max score."

    how = row["how_score_calculated"]
    gain = (max_s - score) * float(row["weight"]) / total_weight
    tips = [f"Potential gain: +{gain:.3f} qa_score (weight {row['weight']})."]

    cr, cer, cp = metric(row, "context_recall"), metric(row, "context_entity_recall"), metric(row, "context_precision")
    faith, noise = metric(row, "faithfulness"), metric(row, "noise_sensitivity")
    fc, critic = metric(row, "factual_correctness"), metric(row, "aspect_critic")
    errors = [m for m in METRICS if metric(row, m) is None]

    # 1. Only the source bonus is missing: the answer is correct but no expected page was cited
    if row["correct"] == "yes" and "source=0" in how:
        tips.append(
            "SOURCES: answer is correct; only the source bonus is missing. Cite one of the expected "
            f"pages ({row['expected_sources']}). Look at how Answer.sources is built in solution.answer(): "
            "list the chunks the model actually used and rank product/rate pages above /rates, calculators and blogs."
        )
        return " ".join(tips)

    # 2. Refusal questions
    if row["category"] == "should_refuse":
        if "refused=no" in how:
            tips.append(
                "REFUSAL: the site has no value for this, but the answer did not decline. Add a prompt rule: "
                "if no explicit value exists for this exact product, set confident=false."
            )
        extra = re.search(r"numbers not in question=(\[[^\]]*\])", how)
        if extra:
            tips.append(
                f"REFUSAL: the answer contains numbers {extra.group(1)} that are not in the question. When "
                "confident=false, replace the text with a fixed refusal without numbers (no phone number either)."
            )
        if critic == 0:
            tips.append("aspect_critic=0 confirms the answer broke the refusal rule.")
        return " ".join(tips)

    # 3. Stale or wrong value trap
    trap = re.search(r"must_not_include hit=(\[[^\]]*\])", how)
    if trap:
        tips.append(
            f"STALE VALUE: the answer repeats a wrong/old value {trap.group(1)}. Prefer the newest-published "
            "or official rate page; add page publish dates to chunk metadata and the prompt."
        )

    missed = re.search(r"missed=(\[[^\]]*\])", how)
    if missed:
        tips.append(f"Missing in answer: {missed.group(1)}.")

    # 4. Retrieval stage
    if cr is not None and cr < LOW:
        tips.append(
            f"RETRIEVAL: the expected facts were not in the retrieved text (context_recall={cr:g}"
            + (f", context_entity_recall={cer:g}" if cer is not None else "")
            + "). Look at _retrieve_and_prepare: remove the 1400-char cut-off, add keyword (BM25) search, "
            "retrieve more candidates and re-rank; check the chunk containing the fact exists in Chroma."
        )
    elif cp is not None and cp < LOW:
        tips.append(
            f"RANKING: relevant text was retrieved but ranked low (context_precision={cp:g}). Re-rank results, "
            "remove clutter (images, navigation) from chunks, boost official product/rate pages."
        )

    # 5. Generation stage
    if faith is not None and faith < FAITHFUL_LOW:
        tips.append(
            f"GENERATION: answer contains claims not supported by the retrieved text (faithfulness={faith:g}). "
            "Tighten the prompt to use only the evidence; check whether the model mixed sources or guessed."
        )
    if noise is not None and noise >= LOW:
        tips.append(
            f"WRONG SOURCE: wrong claims came from retrieved chunks (noise_sensitivity={noise:g}, lower is better). "
            "The model picked a stale or different-product chunk; add publish dates and prefer newest/official sources."
        )
    if fc is not None and fc < LOW and (cr is None or cr >= LOW) and (faith is None or faith >= FAITHFUL_LOW):
        tips.append(
            f"INCOMPLETE ANSWER: facts were retrieved and used faithfully but the answer differs from the "
            f"expected one (factual_correctness={fc:g}). Add prompt rules: list every tier/list item, state "
            "the actual value in yes/no questions, include specific names (e.g. states, products)."
        )

    # 6. General hint for date-sensitive categories (only when the facts were retrieved; otherwise retrieval is the problem)
    if row["category"] in ("stale_trap", "conflict_resolution", "temporal_as_of") and not trap \
            and (cr is None or cr >= LOW):
        tips.append(
            "CHECK DATES: for this category, compare the publish dates of the sources; the newest "
            "press release or the official rate page should win."
        )
    if critic == 0:
        tips.append("aspect_critic=0: answer does not follow the rules (specific values, all tiers, no hedging).")
    if errors:
        tips.append(f"Metric errors for {errors}; re-run ragas.py for those values.")
    return " ".join(tips)


def main() -> None:
    ap = argparse.ArgumentParser(description="Add max score and fix suggestions to a Ragas report")
    ap.add_argument("report", type=Path)
    args = ap.parse_args()

    with args.report.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames)
        rows = list(reader)

    total_weight = sum(float(r["weight"]) for r in rows)
    position = fields.index("score") + 1
    out_fields = fields[:position] + ["max score", "fix suggestion"] + fields[position:]

    for row in rows:
        max_s = max_score(row)
        row["max score"] = max_s
        row["fix suggestion"] = suggest(row, max_s, total_weight)

    out = args.report.with_name(f"{args.report.stem}-to-fix.csv")
    # utf-8-sig lets Excel detect the encoding correctly
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows)

    todo = [r for r in rows if float(r["score"]) < r["max score"] - 1e-9]
    gain = sum((r["max score"] - float(r["score"])) * float(r["weight"]) for r in todo) / total_weight
    print(f"Wrote {out}: {len(todo)} questions can improve, up to +{gain:.4f} qa_score")


if __name__ == "__main__":
    main()
