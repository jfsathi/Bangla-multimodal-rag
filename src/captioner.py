from __future__ import annotations

import base64
import logging
from pathlib import Path

import requests

from .utils import normalize_text

LOGGER = logging.getLogger(__name__)


class OllamaVisionCaptioner:
    """Describes images/charts/tables using a local Ollama vision-language model.

    BLIP-base only produces short generic English captions ("a chart with numbers
    on it") and cannot read Bangla text or explain what a chart/table shows. A
    local multimodal Ollama model (e.g. llava) can actually read the rendered
    text/labels/numbers in the image and describe chart trends or table
    structure in Bangla, which generalizes far better across arbitrary PDFs.
    Runs inside the Ollama server process, so it does not compete for VRAM with
    embedding models already resident in this app's own process.
    """

    # English instructions: local vision models (e.g. llava) follow English
    # instructions far more reliably than Bangla ones, and separately they
    # reliably hallucinate when asked to transcribe Bangla script. So this
    # prompt deliberately asks only for the VISUAL structure (chart type,
    # number of elements/axes, layout, subject) and explicitly forbids
    # guessing any text/numbers - actual text/number reading is already done
    # more reliably by EasyOCR elsewhere in the pipeline. The English output
    # is later machine-translated to Bangla by the existing bilingual chunk step.
    _PROMPT = (
        "This image was extracted from a document page. Describe in 2-4 short "
        "English sentences only its VISUAL structure and type: is it a photo, "
        "a chart (bar/line/pie/scatter), a diagram, a flowchart, a map, a table, "
        "or a logo; how many bars/sections/boxes/rows are visible; how the "
        "elements are arranged or connected; and the general subject if visually "
        "obvious (e.g. a building, a person, a natural scene). "
        "Do NOT attempt to read, transcribe, or guess any text, labels, numbers, "
        "or captions written inside the image - leave those out completely, even "
        "if they are in a script you do not recognize. Only describe shapes, "
        "layout, and visual type."
    )

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434", timeout_seconds: int = 180):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def caption(self, image_path: str | Path) -> str:
        path = Path(image_path)
        if not path.exists() or not path.is_file():
            return ""
        try:
            image_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        except Exception as exc:
            LOGGER.warning("Could not read image for vision captioning %s: %s", path, exc)
            return ""

        payload = {
            "model": self.model_name,
            "prompt": self._PROMPT,
            "images": [image_b64],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 220},
            "keep_alive": "5m",
        }
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=(10, self.timeout_seconds),
            )
            response.raise_for_status()
        except Exception as exc:
            LOGGER.warning("Ollama vision captioning failed for %s using %s: %s", path, self.model_name, exc)
            return ""
        body = response.json()
        return normalize_text(body.get("response", ""))


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
