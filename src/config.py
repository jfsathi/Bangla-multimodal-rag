from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ProjectConfig:
    project_name: str = "bangla_docling_rag"
    embedding_models: list[str] = field(
        default_factory=lambda: [
            "nomic-embed-text-v2-moe:latest",
            "BAAI/bge-m3",  # SentenceTransformer/HuggingFace version, not Ollama bge-m3
            "intfloat/multilingual-e5-base",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ]
    )
    ocr_engine: str = "easyocr"
    # OCR strategy options: auto_best, native_only, force_selected_ocr, compare_easyocr_tesseract
    ocr_strategy: str = "auto_best"
    force_full_page_ocr: bool = False
    auto_ocr_on_bad_bangla: bool = True
    min_bangla_quality: float = 0.55
    min_ocr_confidence: float = 0.35
    bitmap_area_threshold: float = 0.08
    images_scale: float = 3.0
    tesseract_cmd: str = ""
    translation_model: str = "facebook/nllb-200-distilled-600M"
    caption_model: str = "Salesforce/blip-image-captioning-base"
    caption_device: str = "auto"
    embedding_device: str = "auto"
    query_top_k: int = 5
    text_chunk_size: int = 1800
    text_chunk_overlap: int = 250
    generate_page_images: bool = False
    generate_picture_images: bool = True
    # "ollama_vision" reads/describes the picture with a local multimodal Ollama
    # model (see vision_model). It reads Bangla/English text, chart axes, and
    # table structure inside images, so it generalizes across arbitrary PDFs far
    # better than the small English-only BLIP model. Use "blip" only as an
    # offline fallback when no Ollama vision model is installed.
    caption_backend: str = "ollama_vision"
    enable_picture_description: bool = True
    enable_image_ocr: bool = True
    vision_model: str = "llava:latest"
    answer_language: str = "bn"
    ollama_model: str = "qwen3.5:latest"
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def save_project_config(config: ProjectConfig, path: str | Path) -> None:
    target = Path(path)
    target.write_text(yaml.safe_dump(config.to_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8")


def load_project_config(path: str | Path) -> ProjectConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ProjectConfig(**payload)
