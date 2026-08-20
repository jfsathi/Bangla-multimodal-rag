#!/usr/bin/env python
"""Fill in gold_chunk_id (and correct pdf_type) in data/eval_ground_truth_full.csv
by matching each question's expected keywords against the real corpus chunks
produced by the build. Run this only after scripts/build_final_eval_projects.py
has finished for a given pdf_id.

Matching is keyword-overlap based: for each question, every corpus chunk of the
document is scored by how many of the question's expected_keywords_text terms
appear in its retrieval text (bn+en), with a bonus for matching modality. The
best-scoring chunk's id is written back. Rows with zero matches are left blank
and printed for manual review.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import BanglaMultimodalRAGPipeline
from src.utils import normalize_text

ROOT = Path(__file__).resolve().parent.parent
GT_PATH = ROOT / "data" / "eval_ground_truth_full.csv"

PDF_TO_PROJECT = {
    "Science9": "workspace/projects/rag_test1-sciencebook9",
    "Geography": "workspace/projects/rag_test2-geography",
    "piechart": "workspace/projects/rag_test3-pie-chart",
    "Agriculture": "workspace/projects/rag_test4-agriculture",
    "BDHistory": "workspace/projects/rag_test5-bd-history",
    "PolashirJuddho": "workspace/projects/rag_test6-polashir-juddho",
    "UGC1": "workspace/projects/rag_test7-ugc1-law",
    "UGC2": "workspace/projects/rag_test8-ugc2-scholarship",
    "UGC3": "workspace/projects/rag_test9-ugc3-rokeya",
    "DUETBooklet": "workspace/projects/rag_test10-duet-booklet-en",
}


def score_chunk(chunk, keywords: list[str], modality_hint: str) -> int:
    text = normalize_text((chunk.text_bn or "") + " " + (chunk.text_en or "")).lower()
    score = sum(1 for kw in keywords if normalize_text(kw).lower() in text)
    if chunk.modality == modality_hint:
        score += 1
    return score


def main() -> None:
    pipeline = BanglaMultimodalRAGPipeline()
    with GT_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys())

    corpus_cache: dict[str, list] = {}
    unmatched = []
    matched = 0

    for row in rows:
        pdf_id = row["pdf_id"]
        if row.get("gold_chunk_id"):
            matched += 1
            continue
        project_dir = PDF_TO_PROJECT.get(pdf_id)
        if not project_dir or not (ROOT / project_dir).exists():
            unmatched.append((row["question_id"], pdf_id, "no project built"))
            continue
        if project_dir not in corpus_cache:
            corpus_cache[project_dir] = pipeline.load_corpus(project_dir)
        chunks = corpus_cache[project_dir]
        if not chunks:
            unmatched.append((row["question_id"], pdf_id, "empty corpus"))
            continue

        keywords = [k for k in row["expected_keywords_text"].split(";") if k.strip()]
        best_chunk = None
        best_score = 0
        for chunk in chunks:
            s = score_chunk(chunk, keywords, row["modality"])
            if s > best_score:
                best_score = s
                best_chunk = chunk
        if best_chunk is not None and best_score > 0:
            row["gold_chunk_id"] = best_chunk.chunk_id
            row["page_no"] = str(best_chunk.page)
            matched += 1
        else:
            unmatched.append((row["question_id"], pdf_id, row["question_text"][:60]))

    with GT_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Matched {matched}/{len(rows)} questions to a gold chunk.")
    if unmatched:
        print(f"\n{len(unmatched)} unmatched (review manually):")
        for qid, pdf_id, reason in unmatched:
            print(f"  {qid} [{pdf_id}] {reason}")


if __name__ == "__main__":
    main()
