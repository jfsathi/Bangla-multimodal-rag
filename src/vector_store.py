from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .schemas import ChunkRecord, RetrievedChunk
from .utils import cosine_top_k, ensure_dir, iter_jsonl, write_json, write_jsonl


@dataclass(slots=True)
class IndexMetadata:
    mode: str
    embedding_model: str
    chunk_count: int


class SimpleVectorStore:
    def __init__(self, index_dir: str | Path):
        self.index_dir = ensure_dir(index_dir)
        self.embeddings_path = self.index_dir / "embeddings.npy"
        self.chunks_path = self.index_dir / "chunks.jsonl"
        self.metadata_path = self.index_dir / "metadata.json"
        self._embeddings: np.ndarray | None = None
        self._chunks: list[ChunkRecord] | None = None

    def save(self, chunks: list[ChunkRecord], embeddings: np.ndarray, metadata: IndexMetadata) -> None:
        np.save(self.embeddings_path, embeddings.astype(np.float32))
        write_jsonl(self.chunks_path, [chunk.to_dict() for chunk in chunks])
        write_json(self.metadata_path, {
            "mode": metadata.mode,
            "embedding_model": metadata.embedding_model,
            "chunk_count": metadata.chunk_count,
        })
        self._embeddings = embeddings.astype(np.float32)
        self._chunks = chunks

    def _ensure_loaded(self) -> None:
        if self._embeddings is not None and self._chunks is not None:
            return
        self._embeddings = np.load(self.embeddings_path)
        self._chunks = [ChunkRecord.from_dict(row) for row in iter_jsonl(self.chunks_path)]

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[RetrievedChunk]:
        self._ensure_loaded()
        assert self._embeddings is not None
        assert self._chunks is not None
        ranked = cosine_top_k(self._embeddings, query_embedding, k=top_k)
        results: list[RetrievedChunk] = []
        for rank, (idx, score) in enumerate(ranked, start=1):
            results.append(RetrievedChunk(chunk=self._chunks[idx], score=score, rank=rank))
        return results
