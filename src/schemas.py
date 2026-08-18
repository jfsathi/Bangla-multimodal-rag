from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class RawTextNode:
    document_id: str
    source_path: str
    page: int
    text: str
    heading: str = ""
    label: str = "text"


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    document_id: str
    source_path: str
    page: int
    modality: str
    heading: str
    pdf_type: str
    text_bn: str
    text_en: str
    retrieval_text_bn: str
    retrieval_text_en: str
    raw_table_md: str = ""
    image_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChunkRecord":
        return cls(**payload)


@dataclass(slots=True)
class RetrievedChunk:
    chunk: ChunkRecord
    score: float
    rank: int


@dataclass(slots=True)
class AnswerResult:
    mode: str
    answer: str
    query_for_retrieval: str
    retrieved_chunks: list[RetrievedChunk]


@dataclass(slots=True)
class CompareAnswerResult:
    question: str
    direct: AnswerResult
    translated: AnswerResult


@dataclass(slots=True)
class BuildSummary:
    project_dir: str
    corpus_path: str
    total_documents: int
    total_chunks: int
    index_locations: dict[str, dict[str, str]]
