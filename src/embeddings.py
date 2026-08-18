from __future__ import annotations

from typing import Iterable
import os

import numpy as np
import requests


class SentenceTransformerEmbedder:
    """Embedding wrapper with two fully local backends.

    1. Ollama embedding models such as nomic-embed-text-v2-moe:latest and nomic-embed-text:latest.
    2. SentenceTransformer/HuggingFace models already cached on your laptop, such as:
       - BAAI/bge-m3
       - intfloat/multilingual-e5-base
       - sentence-transformers/paraphrase-multilingual-mpnet-base-v2
       - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

    The system is designed for offline/local use. Ollama models are called via
    localhost. SentenceTransformer models must already be present in the local
    HuggingFace cache or local model folder; otherwise loading will fail instead
    of silently downloading from the internet.
    """

    def __init__(self, model_name: str, device: str = "auto", ollama_base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.device = device
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self._model = None

    def _is_ollama_model(self) -> bool:
        name = self.model_name.lower()
        # Use local Ollama for the models listed by `ollama list`.
        # Important: bge-m3:latest is an Ollama model here, not HuggingFace BAAI/bge-m3.
        return (
            name.startswith("ollama:")
            or name.startswith("nomic-")
            or name.startswith("mxbai-")
            or name.startswith("bge-")
            or name in {"bge-m3", "bge-m3:latest"}
        )

    def _ollama_model_name(self) -> str:
        return self.model_name.split(":", 1)[1] if self.model_name.lower().startswith("ollama:") else self.model_name

    @staticmethod
    def _normalize_rows(rows: list[list[float]]) -> np.ndarray:
        arr = np.asarray(rows, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def _truncate_for_ollama(self, text: str) -> str:
        """Keep local Ollama embedding calls stable for long OCR/table chunks.

        Some Ollama embedding runners return HTTP 500 when the input is longer
        than the embedding model context. Truncating here is safer than crashing
        the whole project build. You can override the limit with:
        OLLAMA_EMBED_MAX_CHARS=8000
        """
        max_chars = int(os.environ.get("OLLAMA_EMBED_MAX_CHARS", "4000"))
        text = str(text or "").strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars]

    def _ollama_embed_batch(self, batch: list[str]) -> list[list[float]]:
        """Call Ollama's current embedding endpoint.

        Ollama's current docs use POST /api/embed with an `input` field. The old
        /api/embeddings endpoint is superseded and can fail for some newer
        embedding models, so we avoid it unless the new endpoint is unavailable.
        """
        model = self._ollama_model_name()
        payload = {
            "model": model,
            "input": batch,
            "truncate": True,
            "keep_alive": "10m",
        }
        response = requests.post(
            f"{self.ollama_base_url}/api/embed",
            json=payload,
            timeout=600,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Ollama embedding failed for model '{model}' at /api/embed. "
                f"HTTP {response.status_code}: {response.text[:1000]}"
            )
        data = response.json()
        if "embeddings" not in data:
            raise RuntimeError(
                f"Ollama /api/embed did not return embeddings for model '{model}'. "
                f"Response keys: {list(data.keys())}"
            )
        return data["embeddings"]

    def _ollama_embed_one_legacy(self, text: str) -> list[float]:
        """Last-resort fallback for old Ollama installations."""
        model = self._ollama_model_name()
        response = requests.post(
            f"{self.ollama_base_url}/api/embeddings",
            json={"model": model, "prompt": text, "keep_alive": "10m"},
            timeout=600,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Ollama legacy embedding failed for model '{model}' at /api/embeddings. "
                f"HTTP {response.status_code}: {response.text[:1000]}"
            )
        data = response.json()
        if "embedding" not in data:
            raise RuntimeError(
                f"Ollama /api/embeddings did not return an embedding for model '{model}'. "
                f"Response keys: {list(data.keys())}"
            )
        return data["embedding"]

    def _encode_ollama(self, texts: Iterable[str]) -> np.ndarray:
        payload_texts = [self._truncate_for_ollama(str(t or "")) for t in texts]
        if not payload_texts:
            return np.zeros((0, 0), dtype=np.float32)

        # Small batches are more stable on laptops and avoid Ollama 500 errors
        # with long Bangla OCR/table chunks.
        batch_size = int(os.environ.get("OLLAMA_EMBED_BATCH_SIZE", "4"))
        batch_size = max(1, batch_size)
        embeddings: list[list[float]] = []

        for start in range(0, len(payload_texts), batch_size):
            batch = payload_texts[start : start + batch_size]
            try:
                embeddings.extend(self._ollama_embed_batch(batch))
            except Exception as batch_exc:
                # If a batch fails, retry each item individually through /api/embed.
                # If that still fails, try the legacy endpoint and include the real
                # Ollama error in the final exception.
                for text in batch:
                    try:
                        embeddings.extend(self._ollama_embed_batch([text]))
                    except Exception as single_exc:
                        try:
                            embeddings.append(self._ollama_embed_one_legacy(text))
                        except Exception as legacy_exc:
                            model_name = self._ollama_model_name()
                            hint = ""
                            combined_error = f"{batch_exc} {single_exc} {legacy_exc}"
                            if "unsupported value: NaN" in combined_error and "bge-m3" in model_name.lower():
                                hint = (
                                    "\n\nThis looks like the known local Ollama bge-m3 NaN failure. "
                                    "For thesis comparison, keep bge-m3 as the SentenceTransformer model "
                                    "BAAI/bge-m3 if it is cached locally, or rebuild without bge-m3:latest. "
                                    "The app will now skip failed indexes instead of stopping the whole build."
                                )
                            raise RuntimeError(
                                "Ollama could not create embeddings.\n"
                                f"Model: {model_name}\n"
                                f"Batch endpoint error: {batch_exc}\n"
                                f"Single endpoint error: {single_exc}\n"
                                f"Legacy endpoint error: {legacy_exc}\n\n"
                                "Check that the selected model is an embedding model, "
                                "Ollama is running, and the model name matches `ollama list`."
                                f"{hint}"
                            ) from legacy_exc

        return self._normalize_rows(embeddings)

    def _ensure_loaded(self) -> None:
        if self._is_ollama_model():
            return
        if self._model is not None:
            return
        # Force HuggingFace/SentenceTransformer to use local cache only.
        # If a model is not already downloaded, the user gets a clear error.
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - import-time dependency
            raise RuntimeError(
                "sentence-transformers and torch are required for HuggingFace/SentenceTransformer embeddings."
            ) from exc

        if self.device == "cpu":
            target_device = "cpu"
        elif self.device == "cuda":
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self._model = SentenceTransformer(self.model_name, device=target_device)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load local embedding model '{self.model_name}'. "
                "Because this RAG app is running offline, download/cache the model first "
                "or use one of your Ollama embedding models such as "
                "nomic-embed-text-v2-moe:latest or nomic-embed-text:latest."
            ) from exc

    def _maybe_prefix(self, texts: Iterable[str], kind: str) -> list[str]:
        if self._is_ollama_model():
            return list(texts)
        lower = self.model_name.lower()
        if "e5" not in lower:
            return list(texts)
        prefix = "query: " if kind == "query" else "passage: "
        return [prefix + text for text in texts]

    def encode_queries(self, texts: Iterable[str]) -> np.ndarray:
        self._ensure_loaded()
        payload = self._maybe_prefix(texts, kind="query")
        if self._is_ollama_model():
            return self._encode_ollama(payload)
        assert self._model is not None
        embeddings = self._model.encode(
            payload,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_passages(self, texts: Iterable[str]) -> np.ndarray:
        self._ensure_loaded()
        payload = self._maybe_prefix(texts, kind="passage")
        if self._is_ollama_model():
            return self._encode_ollama(payload)
        assert self._model is not None
        embeddings = self._model.encode(
            payload,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)
