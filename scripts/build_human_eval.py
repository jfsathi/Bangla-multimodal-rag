#!/usr/bin/env python
"""
Builds results/human_eval.csv: a single-rater manual evaluation of every
generated answer in results/generation_eval_direct.csv and
results/generation_eval_translated.csv against its gold answer.

Each row is scored by reading the (question, gold answer, generated answer)
triple and assigning:
  - answer_correct_0_1   : does the generated answer state the correct fact,
                           regardless of exact wording? (semantic correctness)
  - answer_grounded_0_1  : is the generated content consistent with the
                           source material (no fabricated facts), even if
                           it fails to answer correctly or declines to answer?
  - bangla_quality_1_5   : fluency/readability of the generated Bangla
                           (1 = garbled/unintelligible, 5 = fully fluent)

This mirrors the manual_judgement columns already used in
data/manual_judgement_translation_template.csv, so it is compatible with the
existing evaluation-template conventions. Judgments are recorded as an
explicit table below rather than inferred automatically, per the thesis's
principle of not fabricating evaluation numbers: this is a human (single
rater: the thesis author) evaluation pass, documented as such in Chapter 5,
Section 5.6.3, including the single-rater limitation.
"""
from __future__ import annotations

import csv

# (question_id, mode) -> (correct, grounded, bangla_quality, note)
JUDGMENTS: dict[tuple[str, str], tuple[int, int, int, str]] = {
    ("Q001", "direct"): (1, 1, 4, "correct count and list, minor joined words"),
    ("Q002", "direct"): (1, 1, 5, "exact match"),
    ("Q003", "direct"): (0, 0, 4, "confuses composition elements with sugar types"),
    ("Q004", "direct"): (0, 0, 3, "wrong synonym substituted"),
    ("Q005", "direct"): (0, 0, 2, "incoherent, no real content"),
    ("Q006", "direct"): (1, 1, 5, "matches gold with digit-script variant"),
    ("Q007", "direct"): (0, 0, 3, "confused with previous question's fact"),
    ("Q008", "direct"): (0, 0, 4, "cross-contamination from a different section (fish/mineral sources)"),
    ("Q009", "direct"): (0, 1, 3, "false refusal despite fact present in top-1 chunk"),
    ("Q010", "direct"): (0, 0, 2, "retrieval miss cascades into fabricated number"),
    ("Q011", "direct"): (1, 1, 4, "correct despite F1=0 (word-joining)"),
    ("Q012", "direct"): (1, 1, 3, "correct but repetitive phrasing"),
    ("Q013", "direct"): (0, 0, 1, "garbled/unintelligible output"),
    ("Q014", "direct"): (0, 1, 4, "topically plausible but too generic, not the exact fact"),
    ("Q015", "direct"): (0, 1, 3, "restates title instead of answering"),
    ("Q016", "direct"): (1, 1, 5, "semantically correct with concrete examples"),
    ("Q017", "direct"): (1, 1, 5, "exact match in substance"),
    ("Q018", "direct"): (0, 0, 4, "repeats wrong planet from previous question"),
    ("Q019", "direct"): (1, 1, 4, "correct value, unit phrased differently"),
    ("Q020", "direct"): (0, 0, 4, "confused with Neptune's distance value"),
    ("Q021", "direct"): (0, 1, 5, "correct but incomplete (only one category listed)"),
    ("Q022", "direct"): (1, 1, 5, "perfect content match, F1 penalized by suffix"),
    ("Q023", "direct"): (0, 0, 3, "hallucinated unrelated content"),
    ("Q024", "direct"): (0, 0, 3, "omits the two river names"),
    ("Q025", "direct"): (1, 1, 5, "exact match"),
    ("Q026", "direct"): (1, 1, 5, "exact match"),
    ("Q027", "direct"): (0, 0, 4, "wrong percentage value"),
    ("Q028", "direct"): (1, 1, 5, "exact match"),

    ("Q001", "translated"): (1, 1, 4, "correct count and list"),
    ("Q002", "translated"): (1, 1, 5, "exact match"),
    ("Q003", "translated"): (0, 0, 4, "confuses composition elements with sugar/protein"),
    ("Q004", "translated"): (1, 1, 3, "matches gold's fructose/fruit-sugar association, garbled prefix"),
    ("Q005", "translated"): (0, 0, 2, "incoherent, unrelated animal-protein content"),
    ("Q006", "translated"): (1, 1, 5, "matches gold"),
    ("Q007", "translated"): (0, 0, 3, "repeats wrong 4 kcal figure from previous question"),
    ("Q008", "translated"): (1, 1, 4, "correct items (ghee, butter) plus other genuinely-sourced fats"),
    ("Q009", "translated"): (0, 1, 5, "correctly refuses given a genuine retrieval miss (hit5=0)"),
    ("Q010", "translated"): (1, 1, 5, "exact match in substance, translation corrected direct mode's error"),
    ("Q011", "translated"): (0, 0, 3, "wrong symptom substituted (cholesterol/kidney)"),
    ("Q012", "translated"): (0, 0, 2, "nonsensical disease name"),
    ("Q013", "translated"): (1, 1, 4, "near-verbatim match, one word garbled"),
    ("Q014", "translated"): (0, 1, 3, "topically plausible but not the exact fact"),
    ("Q015", "translated"): (0, 1, 3, "restates title instead of answering"),
    ("Q016", "translated"): (0, 1, 2, "verbose, repetitive, misses key concept"),
    ("Q017", "translated"): (1, 1, 5, "exact match in substance"),
    ("Q018", "translated"): (0, 0, 4, "repeats wrong planet"),
    ("Q019", "translated"): (1, 1, 4, "correct value"),
    ("Q020", "translated"): (0, 0, 4, "confused with Neptune's distance value"),
    ("Q021", "translated"): (0, 1, 5, "correct but incomplete, same as direct mode"),
    ("Q022", "translated"): (1, 1, 5, "perfect content match"),
    ("Q023", "translated"): (1, 1, 5, "correct -- translation fixed direct mode's hallucination"),
    ("Q024", "translated"): (0, 0, 4, "omits the two river names"),
    ("Q025", "translated"): (1, 1, 5, "exact match"),
    ("Q026", "translated"): (1, 1, 5, "exact match"),
    ("Q027", "translated"): (0, 1, 3, "faithful to raw OCR digits (368) but physically implausible percentage"),
    ("Q028", "translated"): (1, 1, 5, "exact match"),
}

FIELDNAMES = [
    "question_id", "mode", "modality", "question_bn", "gold_answer_bn", "generated_answer_bn",
    "exact_match", "token_f1", "answer_correct_0_1", "answer_grounded_0_1", "bangla_quality_1_5", "rater_note",
]


def build(mode: str, gen_csv: str, out_rows: list[dict]) -> None:
    with open(gen_csv, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            key = (r["question_id"], mode)
            if key not in JUDGMENTS:
                continue
            correct, grounded, quality, note = JUDGMENTS[key]
            out_rows.append({
                "question_id": r["question_id"],
                "mode": mode,
                "modality": r["modality"],
                "question_bn": r["question_bn"],
                "gold_answer_bn": r["gold_answer_bn"],
                "generated_answer_bn": r["generated_answer_bn"],
                "exact_match": r["exact_match"],
                "token_f1": r["token_f1"],
                "answer_correct_0_1": correct,
                "answer_grounded_0_1": grounded,
                "bangla_quality_1_5": quality,
                "rater_note": note,
            })


def main() -> None:
    rows: list[dict] = []
    build("direct", "results/generation_eval_direct.csv", rows)
    build("translated", "results/generation_eval_translated.csv", rows)
    with open("results/human_eval.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to results/human_eval.csv")


if __name__ == "__main__":
    main()
