#!/usr/bin/env python
"""Build all 10 final-evaluation projects (one PDF each) with a consistent config.

Used for the Q1-journal result-analysis sweep: 10 documents (9 Bangla + 1 full
English) covering native-text, scanned, and mixed PDFs with text/table/image/
chart content, each indexed across all 5 configured embedding models in both
direct and translated retrieval mode.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ProjectConfig
from src.pipeline import BanglaMultimodalRAGPipeline

MATERIALS_DIR = Path(__file__).resolve().parent.parent / "materials"
PROJECTS_ROOT = Path(__file__).resolve().parent.parent / "workspace" / "projects"

# (pdf_id, filename in materials/, project directory name, answer_language)
DOCS = [
    ("Science9", "Science9.pdf", "rag_test1-sciencebook9", "bn"),
    ("Geography", "Geography.pdf", "rag_test2-geography", "bn"),
    ("piechart", "piechart.pdf", "rag_test3-pie-chart", "bn"),
    ("Agriculture", "Bangla_Agriculture_RAG_Test_Document.pdf", "rag_test4-agriculture", "bn"),
    ("BDHistory", "bangladesh_history.pdf", "rag_test5-bd-history", "bn"),
    ("PolashirJuddho", "polashir_juddho.pdf", "rag_test6-polashir-juddho", "bn"),
    ("UGC1", "UGC1(Public University law).pdf", "rag_test7-ugc1-law", "bn"),
    ("UGC2", "UGC2( Scholarship).pdf", "rag_test8-ugc2-scholarship", "bn"),
    ("UGC3", "UGC3 (Rokeya Chair).pdf", "rag_test9-ugc3-rokeya", "bn"),
    ("DUETBooklet", "information booklet duet.pdf", "rag_test10-duet-booklet-en", "en"),
]

EMBEDDING_MODELS = [
    "nomic-embed-text-v2-moe:latest",
    "BAAI/bge-m3",
    "intfloat/multilingual-e5-base",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]


def main() -> None:
    pipeline = BanglaMultimodalRAGPipeline()
    only = sys.argv[1:] if len(sys.argv) > 1 else None

    for pdf_id, filename, project_name, answer_language in DOCS:
        if only and pdf_id not in only:
            continue
        pdf_path = MATERIALS_DIR / filename
        if not pdf_path.exists():
            print(f"[skip] missing file: {pdf_path}")
            continue
        project_dir = PROJECTS_ROOT / project_name
        print(f"\n=== Building {pdf_id} -> {project_dir} ===")
        t0 = time.time()
        config = ProjectConfig(
            project_name=project_name,
            embedding_models=EMBEDDING_MODELS,
            ocr_strategy="auto_best",
            ocr_engine="easyocr",
            auto_ocr_on_bad_bangla=True,
            min_bangla_quality=0.55,
            images_scale=3.0,
            enable_picture_description=True,
            caption_backend="ollama_vision",
            vision_model="llava:latest",
            enable_image_ocr=True,
            query_top_k=5,
            answer_language=answer_language,
            ollama_model="qwen3.5:latest",
            ollama_base_url="http://localhost:11434",
        )
        try:
            summary = pipeline.build_project([pdf_path], project_dir=project_dir, config=config)
            elapsed = time.time() - t0
            print(f"[done] {pdf_id}: {summary.total_chunks} chunks, {elapsed:.1f}s, indexes={list(summary.index_locations.keys())}")
        except Exception as exc:
            print(f"[FAILED] {pdf_id}: {exc}")


if __name__ == "__main__":
    main()
