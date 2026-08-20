#!/usr/bin/env python
"""Architecture ablation: original-proposal image handling (OCR-only, no
vision-model caption) vs. the current architecture (OCR + Ollama vision-model
caption fusion, src/captioner.py:OllamaVisionCaptioner).

This isolates the one accuracy-relevant change made to the proposed
architecture (config.enable_picture_description False->True,
config.caption_backend="ollama_vision") and measures it directly, on the 16
image-modality gold questions across the 6 documents that contain images.
Everything else (embedding model, retrieval mode, LLM, top_k) is held fixed
at the values already established as best in the main matrix, so the delta
is attributable to the captioning change alone.

Usage:
    python scripts/eval_caption_ablation.py build   # rebuild the 6 baseline (no-caption) projects
    python scripts/eval_caption_ablation.py eval    # run retrieval+generation on the 16 image questions
    python scripts/eval_caption_ablation.py both    # build then eval (default)
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_metrics as em  # noqa: E402
from src.config import ProjectConfig, load_project_config  # noqa: E402
from src.generator import OllamaGenerator  # noqa: E402
from src.pipeline import BanglaMultimodalRAGPipeline  # noqa: E402
from src.text_processing import clean_bangla_spelling  # noqa: E402
from src.utils import normalize_text  # noqa: E402

MATERIALS_DIR = ROOT / "materials"
BASELINE_ROOT = ROOT / "workspace" / "projects_baseline_nocap"

# Only the 6 documents that have image-modality gold questions.
DOCS = [
    ("Science9", "Science9.pdf", "baseline1-sciencebook9", "bn"),
    ("Geography", "Geography.pdf", "baseline2-geography", "bn"),
    ("piechart", "piechart.pdf", "baseline3-pie-chart", "bn"),
    ("Agriculture", "Bangla_Agriculture_RAG_Test_Document.pdf", "baseline4-agriculture", "bn"),
    ("BDHistory", "bangladesh_history.pdf", "baseline5-bd-history", "bn"),
    ("PolashirJuddho", "polashir_juddho.pdf", "baseline6-polashir-juddho", "bn"),
]
PDF_TO_BASELINE_PROJECT = {pdf_id: str(BASELINE_ROOT / project_name) for pdf_id, _, project_name, _ in DOCS}

EMBEDDING_MODEL = "BAAI/bge-m3"
MODE = "direct"
LLM = "qwen2.5:3b-instruct"
OUT_PATH = ROOT / "results" / "generation_eval_baseline_nocap_image_direct.csv"

FIELDNAMES = [
    "question_id", "pdf_id", "modality", "question_type", "difficulty",
    "embedding_model", "mode", "llm_model", "answer_language",
    "question_text", "gold_answer_text", "generated_answer_text",
    "gold_chunk_id", "retrieved_chunk_ids", "rank_of_gold", "hit_at_5", "top1_score",
    "exact_match", "token_f1", "rouge_l", "keyword_recall", "semantic_similarity", "is_abstention",
]


def build_baseline_projects() -> None:
    pipeline = BanglaMultimodalRAGPipeline()
    for pdf_id, filename, project_name, answer_language in DOCS:
        pdf_path = MATERIALS_DIR / filename
        if not pdf_path.exists():
            print(f"[skip] missing file: {pdf_path}")
            continue
        project_dir = BASELINE_ROOT / project_name
        print(f"\n=== [baseline-nocap] Building {pdf_id} -> {project_dir} ===", flush=True)
        t0 = time.time()
        config = ProjectConfig(
            project_name=project_name,
            embedding_models=[EMBEDDING_MODEL],
            ocr_strategy="auto_best",
            ocr_engine="easyocr",
            auto_ocr_on_bad_bangla=True,
            min_bangla_quality=0.55,
            images_scale=3.0,
            enable_picture_description=False,  # <-- the ablated setting (original proposal default)
            enable_image_ocr=True,
            query_top_k=5,
            answer_language=answer_language,
            ollama_model="qwen3.5:latest",
            ollama_base_url="http://localhost:11434",
        )
        try:
            summary = pipeline.build_project([pdf_path], project_dir=project_dir, config=config)
            print(f"[done] {pdf_id}: {summary.total_chunks} chunks, {time.time() - t0:.1f}s", flush=True)
        except Exception as exc:
            print(f"[FAILED] {pdf_id}: {exc}", flush=True)


def run_ablation_eval() -> None:
    pipeline = BanglaMultimodalRAGPipeline()
    gt_rows = em.load_ground_truth(str(ROOT / "data" / "eval_ground_truth_full.csv"))
    image_rows = [r for r in gt_rows if r["modality"] == "image" and r["pdf_id"] in PDF_TO_BASELINE_PROJECT]
    print(f"Evaluating {len(image_rows)} image-modality questions against baseline (no-caption) projects...", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8-sig", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in image_rows:
            project_dir = PDF_TO_BASELINE_PROJECT[row["pdf_id"]]
            if not em.index_exists(project_dir, EMBEDDING_MODEL, MODE):
                print(f"[skip] no baseline index for {row['pdf_id']}", flush=True)
                continue
            config = load_project_config(pipeline.project_paths(project_dir)["config"])
            answer_language = row.get("answer_language") or "bn"

            _, retrieved = pipeline.retrieve(
                project_dir=project_dir, question=row["question_text"],
                embedding_model=EMBEDDING_MODEL, mode=MODE, top_k=5,
            )
            retrieved = pipeline._enrich_image_chunks(retrieved, config)

            generator = OllamaGenerator(base_url=config.ollama_base_url, model_name=LLM, temperature=config.temperature)
            answer = generator.generate(question=row["question_text"], retrieved_chunks=retrieved, answer_language=answer_language)
            if answer_language == "bn":
                answer = clean_bangla_spelling(answer)
                not_found_fallback = "প্রদত্ত নথিতে এর উত্তর পাওয়া যায়নি।"
            else:
                not_found_fallback = "The answer was not found in the provided document."
            if not normalize_text(answer):
                answer = not_found_fallback

            retrieved_ids = [r.chunk.chunk_id for r in retrieved]
            gold_id = row["gold_chunk_id"]
            rank = next((i + 1 for i, cid in enumerate(retrieved_ids) if cid == gold_id), None)
            gold_answer = row["gold_answer_text"]
            sem_sim = ""
            try:
                embedder = pipeline._get_embedder(EMBEDDING_MODEL, config.embedding_device, config.ollama_base_url)
                vecs = embedder.encode_queries([answer, gold_answer])
                sem_sim = round(float((vecs[0] * vecs[1]).sum()), 4)
            except Exception as exc:
                print(f"[warn] semantic_similarity failed for {row['question_id']}: {exc}", flush=True)

            writer.writerow({
                "question_id": row["question_id"], "pdf_id": row["pdf_id"], "modality": row["modality"],
                "question_type": row["question_type"], "difficulty": row.get("difficulty", ""),
                "embedding_model": EMBEDDING_MODEL, "mode": MODE, "llm_model": LLM,
                "answer_language": answer_language, "question_text": row["question_text"],
                "gold_answer_text": gold_answer, "generated_answer_text": answer,
                "gold_chunk_id": gold_id, "retrieved_chunk_ids": "|".join(retrieved_ids),
                "rank_of_gold": rank or "", "hit_at_5": int(bool(rank) and rank <= 5),
                "top1_score": round(retrieved[0].score, 4) if retrieved else "",
                "exact_match": em.exact_match(answer, gold_answer),
                "token_f1": round(em.token_f1(answer, gold_answer), 4),
                "rouge_l": round(em.rouge_l(answer, gold_answer), 4),
                "keyword_recall": round(em.keyword_recall(answer, row["expected_keywords_text"]), 4),
                "semantic_similarity": sem_sim,
                "is_abstention": int(em.looks_like_abstention(answer)),
            })
            print(f"[ok] {row['question_id']} f1={round(em.token_f1(answer, gold_answer), 3)}", flush=True)
    print(f"Wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode in ("build", "both"):
        build_baseline_projects()
    if mode in ("eval", "both"):
        run_ablation_eval()
