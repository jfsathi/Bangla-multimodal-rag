#!/usr/bin/env python
"""Build the 11th, supplementary evaluation project: a 1-page English IICT/DUET
admission notice, added alongside (not replacing) DUETBooklet to test whether
the English-language retrieval gap seen on the 33-page booklet is tied to
document length/density rather than language handling itself. Indexed under
BAAI/bge-m3 only (the established best embedding model) to keep this
supplementary build fast.
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

PDF_ID = "IICTNotice"
FILENAME = "notice of admission test iict duet.pdf"
PROJECT_NAME = "rag_test11-iict-admission-notice"
ANSWER_LANGUAGE = "en"


def main() -> None:
    pipeline = BanglaMultimodalRAGPipeline()
    pdf_path = MATERIALS_DIR / FILENAME
    project_dir = PROJECTS_ROOT / PROJECT_NAME
    print(f"=== Building {PDF_ID} -> {project_dir} ===", flush=True)
    t0 = time.time()
    config = ProjectConfig(
        project_name=PROJECT_NAME,
        embedding_models=[
            "nomic-embed-text-v2-moe:latest",
            "BAAI/bge-m3",
            "intfloat/multilingual-e5-base",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ],
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
        answer_language=ANSWER_LANGUAGE,
        ollama_model="qwen3.5:latest",
        ollama_base_url="http://localhost:11434",
    )
    summary = pipeline.build_project([pdf_path], project_dir=project_dir, config=config)
    print(f"[done] {PDF_ID}: {summary.total_chunks} chunks, {time.time() - t0:.1f}s, "
          f"indexes={list(summary.index_locations.keys())}", flush=True)


if __name__ == "__main__":
    main()
