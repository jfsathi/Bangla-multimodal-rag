from __future__ import annotations

import logging
import re
from pathlib import Path

from .text_processing import clean_bangla_spelling
from .utils import normalize_text

LOGGER = logging.getLogger(__name__)

_PLANET_NAMES = (
    "বুধ",
    "শুক্র",
    "পৃথিবী",
    "মঙ্গল",
    "বৃহস্পতি",
    "শনি",
    "নেপচুন",
    "ইউরেনাস",
    "অ্যারিস",
    "প্লুটো",
)
_GENERIC_IMAGE_MARKERS = (
    "image extracted from the document",
    "image under heading",
    "নথিতে থেকে ছবি",
)
_DISTANCE_RE = re.compile(
    r"([\d০-৯]+(?:[.,][\d০-৯]+)?)\s*(?:কোটি|km|কিমি)?",
    re.IGNORECASE,
)


class ImageOcrReader:
    """Bangla/English OCR for chart, diagram, and figure images."""

    def __init__(self, languages: tuple[str, ...] = ("bn", "en"), gpu: bool = False):
        self.languages = languages
        self.gpu = gpu
        self._reader = None

    def _ensure_loaded(self) -> None:
        if self._reader is not None:
            return
        try:
            import easyocr
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("easyocr is required for image OCR.") from exc
        self._reader = easyocr.Reader(list(self.languages), gpu=self.gpu, verbose=False)

    def read_lines(self, image_path: str | Path) -> list[str]:
        path = Path(image_path)
        if not path.exists() or not path.is_file():
            return []
        try:
            self._ensure_loaded()
            assert self._reader is not None
            raw = self._reader.readtext(str(path), detail=0, paragraph=False)
        except Exception as exc:
            LOGGER.warning("Image OCR failed for %s: %s", path, exc)
            return []
        lines = [clean_bangla_spelling(normalize_text(line)) for line in raw]
        return [line for line in lines if line]


def is_generic_image_text(text: str) -> bool:
    normalized = normalize_text(text).lower()
    if not normalized:
        return True
    return any(marker in normalized for marker in _GENERIC_IMAGE_MARKERS) and "চিত্র" not in normalized


def extract_figure_title(lines: list[str]) -> str:
    for line in lines:
        if re.search(r"চিত্র\s*[\d০-৯]+", line, re.IGNORECASE):
            return line
    return ""


def _looks_like_planet(line: str) -> str | None:
    compact = re.sub(r"\s+", "", line)
    for planet in _PLANET_NAMES:
        if planet in compact and len(compact) <= len(planet) + 4:
            return planet
    return None


def _looks_like_distance(line: str) -> bool:
    if not _DISTANCE_RE.search(line):
        return False
    if "কোটি" in line or "কিমি" in line or "km" in line.lower():
        return True
    # Short numeric-only tokens near a planet are usually chart values.
    digits = re.sub(r"[^\d০-৯.,]", "", line)
    return bool(digits) and len(digits) <= 8


def verbalize_chart_from_ocr_lines(lines: list[str]) -> str:
    """Turn OCR tokens from a chart/diagram into readable facts for retrieval and LLM context."""
    if not lines:
        return ""

    title = extract_figure_title(lines)
    facts: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        planet = _looks_like_planet(line)
        if planet:
            distance = ""
            if i + 1 < len(lines) and _looks_like_distance(lines[i + 1]):
                distance = lines[i + 1]
                i += 2
            else:
                i += 1
            if distance:
                facts.append(f"{planet}: {distance}")
            continue
        i += 1

    parts: list[str] = []
    if title:
        parts.append(title)
    if facts:
        parts.append("গ্রহ/বস্তু ও গড় দূরত্ব: " + "; ".join(facts))
    body = "\n".join(parts)
    if body:
        return body
    return "\n".join(lines)


def ocr_image_to_text(image_path: str | Path, reader: ImageOcrReader | None = None) -> str:
    reader = reader or ImageOcrReader()
    lines = reader.read_lines(image_path)
    return verbalize_chart_from_ocr_lines(lines)


def chart_fact_hints(question: str, ocr_text: str) -> str:
    """Pick chart rows that match the question (e.g. Mars distance from Figure 1)."""
    if not ocr_text:
        return ""
    q = normalize_text(question)
    terms = [t for t in re.split(r"\s+", q) if len(t) >= 2]
    hints: list[str] = []
    segments: list[str] = []
    for line in ocr_text.splitlines():
        segments.extend(part.strip() for part in re.split(r"[;\n]", line) if part.strip())
    for segment in segments:
        norm = normalize_text(segment)
        if not norm:
            continue
        if any(term in norm for term in terms):
            hints.append(segment.strip())
    if hints:
        return "\n".join(dict.fromkeys(hints[:5]))

    title = extract_figure_title(ocr_text.splitlines())
    if title and any(k in q for k in ("চিত্র", "figure", "chart", "graph")):
        for segment in segments:
            if ":" in segment:
                hints.append(segment.strip())
        return "\n".join(dict.fromkeys(hints[:5]))
    return ""
