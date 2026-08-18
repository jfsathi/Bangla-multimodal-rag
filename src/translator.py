from __future__ import annotations

import logging
from dataclasses import dataclass

from .utils import contains_bangla, normalize_text, sentence_like_split

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TranslationPair:
    text_bn: str
    text_en: str


class NllbTranslator:
    """Local translator using the NLLB 200 distilled model.

    It is used in two directions:
    - Bangla -> English for the translated retrieval branch
    - English -> Bangla for captions or generated answers when needed
    """

    LANG_CODES = {
        "bn": "ben_Beng",
        "en": "eng_Latn",
    }

    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None
        self._torch = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - import-time dependency
            raise RuntimeError(
                "transformers and torch are required for translation."
            ) from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        target_device = self._resolve_device(torch)
        if target_device != "cpu":
            self._model.to(target_device)
        self._model.eval()

    def _resolve_device(self, torch_module) -> str:
        if self.device == "cpu":
            return "cpu"
        if self.device == "cuda" and torch_module.cuda.is_available():
            return "cuda"
        if self.device == "auto" and torch_module.cuda.is_available():
            return "cuda"
        return "cpu"

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        text = normalize_text(text)
        if not text or source_lang == target_lang:
            return text
        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None

        if source_lang not in self.LANG_CODES or target_lang not in self.LANG_CODES:
            raise ValueError(f"Unsupported translation pair: {source_lang}->{target_lang}")

        device = self._resolve_device(self._torch)
        pieces = sentence_like_split(text, limit=1000)
        outputs: list[str] = []
        self._tokenizer.src_lang = self.LANG_CODES[source_lang]
        forced_bos = self._tokenizer.convert_tokens_to_ids(self.LANG_CODES[target_lang])

        for piece in pieces:
            inputs = self._tokenizer(piece, return_tensors="pt", truncation=True, padding=True)
            if device != "cpu":
                inputs = {k: v.to(device) for k, v in inputs.items()}
            with self._torch.no_grad():
                tokens = self._model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos,
                    max_new_tokens=512,
                )
            translated = self._tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
            outputs.append(normalize_text(translated))
        return normalize_text(" ".join(outputs))

    def make_bilingual(self, text: str, preferred_source: str | None = None) -> TranslationPair:
        text = normalize_text(text)
        if not text:
            return TranslationPair(text_bn="", text_en="")

        source = preferred_source
        if source is None:
            source = "bn" if contains_bangla(text) else "en"

        if source == "bn":
            return TranslationPair(
                text_bn=text,
                text_en=self.translate(text, "bn", "en"),
            )
        if source == "en":
            return TranslationPair(
                text_bn=self.translate(text, "en", "bn"),
                text_en=text,
            )
        raise ValueError(f"Unsupported source language: {source}")
