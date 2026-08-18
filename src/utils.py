from __future__ import annotations

import json
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np


BANGLA_RE = re.compile(r"[\u0980-\u09FF]")
WHITESPACE_RE = re.compile(r"\s+")
HEADING_RE = re.compile(r"^(#{1,6})\s+")


# Common OCR/font-extraction repairs for Bangla PDFs.
# These are intentionally conservative and target patterns that repeatedly appear
# in NCTB-style PDFs and generated scanned PDFs. They improve lexical matching,
# table-row facts, and short factual answers.
BANGLA_OCR_REPLACEMENTS = {
    "শকর্ণ রা": "শর্করা",
    "শক র্ণ রা": "শর্করা",
    "শকর্ণরা": "শর্করা",
    "আমষ": "আমিষ",
    "ভিটোমন": "ভিটামিন",
    "খনজ": "খনিজ",
    "পোন": "পানি",
    "নরোপদ": "নিরাপদ",
    "প্রেধান": "প্রধান",
    "কিাজ": "কাজ",
    "দেুর্যাগ": "দুর্যোগ",
    "দু যর্ণোগ": "দুর্যোগ",
    "দু{যোগ": "দুর্যোগ",
    "দুর্যাগ": "দুর্যোগ",
    "সরবরোহি": "সরবরাহ",
    "শক্ত সরবরাহ": "শক্তি সরবরাহ",
    "শক্ত দয়": "শক্তি দেয়",
    "দহি": "দেহ",
    "মরোমত": "মেরামত",
    "শোরীরবৃত্তিীয়": "শারীরবৃত্তীয়",
    "রাগ": "রোগ",
    "ডেোল": "ডাল",
    "ডেম": "ডিম",
    "ভিোত": "ভাত",
    "তোৎক্ষণিক": "তাৎক্ষণিক",
}


def clean_bangla_ocr_text(text: str) -> str:
    """Repair frequent Bangla OCR/custom-font extraction artifacts.

    This is not a full spell corrector. It only fixes high-confidence patterns
    that hurt retrieval, e.g. ``শকর্ণ রা`` vs ``শর্করা``.
    """
    text = text or ""
    for bad, good in BANGLA_OCR_REPLACEMENTS.items():
        text = text.replace(bad, good)

    # Remove spaces accidentally inserted around Bangla virama-like fragments.
    text = re.sub(r"([\u0980-\u09FF])\s+([\u09CD\u09BC])", r"\1\2", text)
    text = re.sub(r"([\u09CD\u09BC])\s+([\u0980-\u09FF])", r"\1\2", text)
    return text


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def slugify(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-").lower()
    return value or "default"


def normalize_text_basic(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text or "").strip()


def normalize_text(text: str) -> str:
    return clean_bangla_ocr_text(normalize_text_basic(text))


def safe_truncate(text: str, max_len: int = 300) -> str:
    text = normalize_text(text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def contains_bangla(text: str) -> bool:
    return bool(BANGLA_RE.search(text or ""))


BAD_BANGLA_MARKERS = ["=", "ǂ", "7", "9", ">", "<", "￾", "ƒ", "ƶ", "", "Ǭ"]


def bangla_text_quality_score(text: str) -> float:
    """Estimate whether extracted Bangla text is readable.

    Many Bangla PDFs use custom font encodings. Native extraction can then
    produce strings such as ``ভিাষা`` or include symbols like ``ǂ`` and ``=``.
    A low score should trigger full-page OCR for that document.
    """
    text = normalize_text(text)
    if not text:
        return 0.0
    bangla_chars = sum(1 for ch in text if "\u0980" <= ch <= "\u09FF")
    letters = sum(1 for ch in text if ch.isalpha())
    if letters == 0:
        return 0.0
    bad_count = sum(text.count(marker) for marker in BAD_BANGLA_MARKERS)
    # Digits used inside ordinary Bangla text are fine, but a very large amount
    # of ASCII digits embedded inside words is a strong sign of bad extraction.
    suspicious_digit_count = sum(1 for ch in text if ch in "79")
    bangla_ratio = bangla_chars / max(letters, 1)
    bad_penalty = min((bad_count + suspicious_digit_count * 0.25) / max(len(text), 1) * 8.0, 0.65)
    return max(0.0, min(1.0, bangla_ratio - bad_penalty))


EXTRACTION_BAD_CHARS = set("{}[]²³ɸǂǃǬǭƒƶ­﻿￾�|=<>¦")
CONTROL_CHAR_RE = re.compile(r"[\x00--]")


def bangla_extraction_stats(text: str) -> dict[str, Any]:
    """Return generic extraction quality diagnostics for Bangla PDF text.

    This intentionally checks the raw extracted text before heavy spelling cleanup.
    A good Bangla extraction has a high Bangla-character ratio and very few
    custom-font/OCR noise markers. A bad custom-font extraction often contains
    characters such as {, }, ², ɸ, ǂ, ǃ, control codes, or letters split by =.
    """
    raw = normalize_text_basic(text or "")
    total = len(raw)
    bangla_chars = sum(1 for ch in raw if "ঀ" <= ch <= "৿")
    ascii_letters = sum(1 for ch in raw if ("a" <= ch.lower() <= "z"))
    digits = sum(1 for ch in raw if ch.isdigit() or "০" <= ch <= "৯")
    bad_chars = sum(1 for ch in raw if ch in EXTRACTION_BAD_CHARS)
    control_chars = len(CONTROL_CHAR_RE.findall(raw))
    # Suspicious digits inside Bangla words, e.g. বৈ7শিষ্ট্য, are common in broken fonts.
    embedded_digit_hits = len(re.findall(r"[ঀ-৿][0-9][ঀ-৿]", raw))
    brace_like_hits = raw.count("{") + raw.count("}") + raw.count("=")
    words = re.findall(r"[ঀ-৿A-Za-z0-9]+", raw)
    suspicious_words = [w for w in words if any(ch in w for ch in "{}²ɸǂǃǬǭ=<>¦") or CONTROL_CHAR_RE.search(w)]

    bangla_ratio = bangla_chars / max(total, 1)
    bad_ratio = (bad_chars + control_chars + embedded_digit_hits * 2 + brace_like_hits * 0.75) / max(total, 1)
    # Reward Bangla coverage, penalize custom-font/OCR garbage. Clamp 0..1.
    score = bangla_ratio - min(bad_ratio * 4.0, 0.85)
    if total < 20 and bangla_chars < 5:
        score = 0.0
    score = max(0.0, min(1.0, score))
    return {
        "chars": total,
        "bangla_chars": bangla_chars,
        "ascii_letters": ascii_letters,
        "digits": digits,
        "bad_chars": bad_chars,
        "control_chars": control_chars,
        "embedded_digit_hits": embedded_digit_hits,
        "suspicious_word_count": len(suspicious_words),
        "bangla_ratio": round(bangla_ratio, 4),
        "bad_ratio": round(bad_ratio, 4),
        "quality_score": round(score, 4),
        "preview": safe_truncate(raw, 220),
    }


def is_bad_bangla_extraction(text: str, threshold: float = 0.55) -> bool:
    stats = bangla_extraction_stats(text)
    if stats["chars"] < 30:
        return True
    if stats["quality_score"] < threshold:
        return True
    if stats["control_chars"] > 0:
        return True
    if stats["bad_chars"] >= 3 and stats["bad_ratio"] > 0.01:
        return True
    return False


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return iter(())
    def _gen() -> Iterator[dict[str, Any]]:
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)
    return _gen()


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def copy_file(src: str | Path, dst: str | Path) -> Path:
    target = Path(dst)
    ensure_dir(target.parent)
    shutil.copy2(src, target)
    return target


def sentence_like_split(text: str, limit: int = 1400) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    if len(text) <= limit:
        return [text]
    parts = re.split(r"(?<=[.!?।])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for part in parts:
        part = part.strip()
        if not part:
            continue
        projected = current_len + len(part) + (1 if current else 0)
        if current and projected > limit:
            chunks.append(" ".join(current))
            current = [part]
            current_len = len(part)
        elif len(part) > limit:
            # hard split a single very long sentence
            for i in range(0, len(part), limit):
                sub = part[i : i + limit]
                if current:
                    chunks.append(" ".join(current))
                    current = []
                    current_len = 0
                chunks.append(sub)
            current = []
            current_len = 0
        else:
            current.append(part)
            current_len = projected
    if current:
        chunks.append(" ".join(current))
    return chunks


def split_text_with_overlap(text: str, chunk_size: int = 1800, overlap: int = 250) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    pieces: list[str] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(text_len, start + chunk_size)
        if end < text_len:
            last_break = max(
                text.rfind(" ", start, end),
                text.rfind("।", start, end),
                text.rfind(".", start, end),
                text.rfind("\n", start, end),
            )
            if last_break > start + chunk_size // 2:
                end = last_break + 1
        chunk = text[start:end].strip()
        if chunk:
            pieces.append(chunk)
        if end >= text_len:
            break
        start = max(0, end - overlap)
    return pieces


def format_source_tag(source_path: str, page: int) -> str:
    return f"[{Path(source_path).name} p.{page}]"


def detect_pdf_type(native_text_ratio: float) -> str:
    if native_text_ratio >= 0.70:
        return "native_text"
    if native_text_ratio >= 0.30:
        return "mixed"
    return "scanned"


def estimate_native_text_ratio(pdf_path: str | Path) -> float:
    try:
        import fitz  # PyMuPDF
    except Exception:
        return 0.0

    doc = fitz.open(str(pdf_path))
    total = len(doc)
    if total == 0:
        return 0.0
    native_pages = 0
    for page in doc:
        text = normalize_text(page.get_text("text"))
        if len(text) >= 40:
            native_pages += 1
    return native_pages / total


def cosine_top_k(matrix: np.ndarray, query: np.ndarray, k: int) -> list[tuple[int, float]]:
    if matrix.size == 0:
        return []
    query = query.astype(np.float32)
    if query.ndim == 1:
        query = query[None, :]
    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    query_norm[query_norm == 0] = 1.0
    matrix_norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix_norm[matrix_norm == 0] = 1.0
    scores = (matrix @ query.T).squeeze(1) / (matrix_norm.squeeze(1) * query_norm.squeeze())
    k = min(k, len(scores))
    top_idx = np.argpartition(scores, -k)[-k:]
    ranked = sorted(((int(idx), float(scores[idx])) for idx in top_idx), key=lambda x: x[1], reverse=True)
    return ranked


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).lower().split()
    ref_tokens = normalize_text(reference).lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    for tok in pred_tokens:
        pred_counts[tok] = pred_counts.get(tok, 0) + 1
    for tok in ref_tokens:
        ref_counts[tok] = ref_counts.get(tok, 0) + 1
    overlap = sum(min(pred_counts.get(tok, 0), ref_counts.get(tok, 0)) for tok in set(pred_counts) | set(ref_counts))
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def chunk_id_for(source_path: str, page: int, modality: str, ordinal: int) -> str:
    return f"{slugify(Path(source_path).stem)}-p{page}-{modality}-{ordinal:04d}"
