from __future__ import annotations

import logging
from pathlib import Path

from .utils import normalize_text

LOGGER = logging.getLogger(__name__)


class BlipCaptioner:
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self._pipe = None

    def _ensure_loaded(self) -> None:
        if self._pipe is not None:
            return
        try:
            import torch
            from transformers import pipeline
        except Exception as exc:  # pragma: no cover - import-time dependency
            raise RuntimeError("transformers and torch are required for image captioning.") from exc

        if self.device == "cpu":
            device_index = -1
        elif self.device == "cuda":
            device_index = 0
        else:
            device_index = 0 if torch.cuda.is_available() else -1
        self._pipe = pipeline("image-to-text", model=self.model_name, device=device_index)

    def caption(self, image_path: str | Path) -> str:
        self._ensure_loaded()
        assert self._pipe is not None
        result = self._pipe(str(image_path), max_new_tokens=64)
        if not result:
            return ""
        caption = result[0].get("generated_text", "")
        return normalize_text(caption)
