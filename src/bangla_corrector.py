from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from difflib import get_close_matches

BANGLA_RE = re.compile(r"[\u0980-\u09FF]+")
PROTECTED_PATTERNS = [
    re.compile(r"^\d+(\.\d+)?%?$"),
    re.compile(r"^[০-৯]+(\.[০-৯]+)?%?$"),
    re.compile(r"^p\.\d+$", re.IGNORECASE),
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def load_ocr_corrections() -> dict[str, str]:
    path = _project_root() / "data" / "bn_ocr_corrections.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_lexicon() -> set[str]:
    words: set[str] = set()
    for name in ["bn_common_words.txt", "bn_domain_terms.txt"]:
        path = _project_root() / "data" / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            word = line.strip()
            if word and not word.startswith("#"):
                words.add(word)
    return words


def is_protected_token(token: str) -> bool:
    token = token.strip()
    if not token:
        return True
    if any(pattern.match(token) for pattern in PROTECTED_PATTERNS):
        return True
    if any(ch in token for ch in ["/", "\\", ".", "@", "_", "-", "[", "]"]):
        return True
    return False


def normalize_bangla_spacing(text: str) -> str:
    if not text:
        return text
    text = "".join(ch for ch in text if ord(ch) >= 32 or ch in "\n\t\r")
    # Rejoin separated Bangla vowel signs and nasal signs.
    text = re.sub(r"([ক-হড়ঢ়য়])\s+([ািীুূৃেৈোৌঁংঃ])", r"\1\2", text)
    text = re.sub(r"([ািীুূৃেৈোৌঁংঃ])\s+([ক-হড়ঢ়য়])", r"\1\2", text)
    # Common split forms from Bangla PDF font extraction/OCR.
    replacements = {
        "ঝ ুঁ কি": "ঝুঁকি", "ঝ ু ঁ ক": "ঝুঁকি", "ঝ ুঁ ক": "ঝুঁকি",
        "ক ৃ ষ": "কৃষ", "কৃ ষ": "কৃষ", "ভি ূ গোল": "ভূগোল",
        "প ু ন": "পুন", "দ ু র্যোগ": "দুর্যোগ", "দ ু যর্ণোগ": "দুর্যোগ",
        "খো দ্য": "খাদ্য", "স্বো স্থ্য": "স্বাস্থ্য", "পো নি": "পানি",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    for ch in ["{", "}", "²", "ɸ", "ǂ", "ǃ", "Ǭ", "ǭ", "ƶ", "\u0001", "\u0002", "�"]:
        text = text.replace(ch, "")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([।,;:?!])", r"\1", text)
    return text.strip()


def apply_exact_corrections(text: str) -> str:
    corrections = load_ocr_corrections()
    for wrong, right in sorted(corrections.items(), key=lambda x: len(x[0]), reverse=True):
        text = text.replace(wrong, right)
    return text


def fuzzy_correct_token(token: str, lexicon: set[str]) -> str:
    if is_protected_token(token) or token in lexicon:
        return token
    if not BANGLA_RE.search(token) or len(token) < 4:
        return token
    cutoff = 0.88 if len(token) <= 7 else 0.91
    matches = get_close_matches(token, lexicon, n=1, cutoff=cutoff)
    return matches[0] if matches else token


def fuzzy_correct_text(text: str, enabled: bool = False) -> str:
    if not enabled:
        return text
    lexicon = load_lexicon()
    if not lexicon:
        return text
    parts = re.findall(r"[\u0980-\u09FF]+|[^\u0980-\u09FF]+", text)
    return "".join(fuzzy_correct_token(part, lexicon) if BANGLA_RE.fullmatch(part) else part for part in parts)


def correct_bangla_text(text: str, fuzzy: bool = False) -> str:
    """Conservative Bangla correction layer for extracted text.

    Safe order: spacing/control cleanup -> exact OCR/font map -> optional
    lexicon-based fuzzy correction -> final spacing cleanup.
    """
    if not text:
        return text
    text = normalize_bangla_spacing(text)
    text = apply_exact_corrections(text)
    text = fuzzy_correct_text(text, enabled=fuzzy)
    text = normalize_bangla_spacing(text)
    return text
