from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from .captioner import BlipCaptioner, OllamaVisionCaptioner
from .config import ProjectConfig, load_project_config, save_project_config
from .docling_extractor import DoclingExtractor
from .embeddings import SentenceTransformerEmbedder
from .generator import OllamaGenerator
from .image_ocr import ImageOcrReader, is_generic_image_text, ocr_image_to_text
from .schemas import AnswerResult, BuildSummary, ChunkRecord, CompareAnswerResult, RetrievedChunk
from .text_processing import (
    clean_bangla_spelling,
    group_text_nodes_by_page,
    make_image_chunk,
    make_structure_chunks,
    make_table_chunks,
    make_text_chunks,
)
from .translator import NllbTranslator
from .utils import (
    contains_bangla,
    copy_file,
    ensure_dir,
    iter_jsonl,
    normalize_text,
    slugify,
    write_json,
    write_jsonl,
)
from .vector_store import IndexMetadata, SimpleVectorStore

LOGGER = logging.getLogger(__name__)


class BanglaMultimodalRAGPipeline:
    def __init__(self) -> None:
        self._embedder_cache: dict[str, SentenceTransformerEmbedder] = {}
        self._vector_store_cache: dict[str, SimpleVectorStore] = {}
        self._translator_cache: dict[str, NllbTranslator] = {}
        self._captioner_cache: dict[str, BlipCaptioner | OllamaVisionCaptioner] = {}
        self._image_ocr_cache: ImageOcrReader | None = None

    def _get_image_ocr_reader(self) -> ImageOcrReader:
        if self._image_ocr_cache is None:
            self._image_ocr_cache = ImageOcrReader()
        return self._image_ocr_cache

    def _enrich_image_chunks(self, retrieved: list[RetrievedChunk], config: ProjectConfig) -> list[RetrievedChunk]:
        """Run OCR on retrieved image chunks when indexed text is still a placeholder."""
        if not config.enable_image_ocr:
            return retrieved
        reader = self._get_image_ocr_reader()
        enriched: list[RetrievedChunk] = []
        for item in retrieved:
            chunk = item.chunk
            if chunk.modality != "image" or not chunk.image_path:
                enriched.append(item)
                continue
            ocr_text = normalize_text(chunk.metadata.get("ocr_text", ""))
            if not ocr_text or is_generic_image_text(chunk.text_bn):
                ocr_text = ocr_image_to_text(chunk.image_path, reader)
            if not ocr_text:
                enriched.append(item)
                continue
            heading_part = f"শিরোনাম: {chunk.heading}\n" if chunk.heading else ""
            if is_generic_image_text(chunk.text_bn):
                new_text = heading_part + ocr_text
            else:
                new_text = chunk.text_bn
            if ocr_text not in new_text:
                new_text = f"{new_text}\n{ocr_text}".strip()
            metadata = dict(chunk.metadata)
            metadata["ocr_text"] = ocr_text
            chunk = replace(
                chunk,
                text_bn=new_text,
                text_en=new_text,
                retrieval_text_bn=new_text,
                metadata=metadata,
            )
            enriched.append(RetrievedChunk(chunk=chunk, score=item.score, rank=item.rank))
        return enriched

    @staticmethod
    def project_paths(project_dir: str | Path) -> dict[str, Path]:
        root = Path(project_dir)
        return {
            "root": root,
            "pdfs": ensure_dir(root / "data" / "pdfs"),
            "artifacts": ensure_dir(root / "artifacts"),
            "corpus": ensure_dir(root / "corpus"),
            "indexes": ensure_dir(root / "indexes"),
            "results": ensure_dir(root / "results"),
            "config": root / "project_config.yaml",
            "corpus_jsonl": root / "corpus" / "chunks.jsonl",
        }

    def _get_translator(self, config: ProjectConfig) -> NllbTranslator:
        key = f"{config.translation_model}|{config.caption_device}"
        if key not in self._translator_cache:
            self._translator_cache[key] = NllbTranslator(
                model_name=config.translation_model,
                device=config.caption_device,
            )
        return self._translator_cache[key]

    def _get_captioner(self, config: ProjectConfig) -> BlipCaptioner | OllamaVisionCaptioner:
        backend = getattr(config, "caption_backend", "ollama_vision")
        if backend == "ollama_vision":
            key = f"ollama_vision|{config.vision_model}|{config.ollama_base_url}"
            if key not in self._captioner_cache:
                self._captioner_cache[key] = OllamaVisionCaptioner(
                    model_name=config.vision_model,
                    base_url=config.ollama_base_url,
                )
            return self._captioner_cache[key]
        key = f"blip|{config.caption_model}|{config.caption_device}"
        if key not in self._captioner_cache:
            self._captioner_cache[key] = BlipCaptioner(
                model_name=config.caption_model,
                device=config.caption_device,
            )
        return self._captioner_cache[key]

    def _get_embedder(self, model_name: str, device: str, ollama_base_url: str = "http://localhost:11434") -> SentenceTransformerEmbedder:
        key = f"{model_name}|{device}|{ollama_base_url}"
        if key not in self._embedder_cache:
            self._embedder_cache[key] = SentenceTransformerEmbedder(
                model_name=model_name,
                device=device,
                ollama_base_url=ollama_base_url,
            )
        return self._embedder_cache[key]

    def _get_vector_store(self, index_dir: str | Path) -> SimpleVectorStore:
        key = str(Path(index_dir).resolve())
        if key not in self._vector_store_cache:
            self._vector_store_cache[key] = SimpleVectorStore(index_dir)
        return self._vector_store_cache[key]

    def _bilingualize_chunk(self, chunk: ChunkRecord, translator: NllbTranslator) -> ChunkRecord:
        text_source = chunk.text_bn or chunk.text_en
        retrieval_source = chunk.retrieval_text_bn or chunk.retrieval_text_en or text_source
        preferred_source = "bn" if contains_bangla(text_source or retrieval_source) else "en"
        content_pair = translator.make_bilingual(text_source, preferred_source=preferred_source)
        retrieval_pair = translator.make_bilingual(retrieval_source, preferred_source=preferred_source)
        return replace(
            chunk,
            text_bn=content_pair.text_bn,
            text_en=content_pair.text_en,
            retrieval_text_bn=retrieval_pair.text_bn,
            retrieval_text_en=retrieval_pair.text_en,
        )

    def build_project(
        self,
        pdf_paths: Iterable[str | Path],
        project_dir: str | Path,
        config: ProjectConfig,
    ) -> BuildSummary:
        paths = self.project_paths(project_dir)
        save_project_config(config, paths["config"])

        local_pdfs: list[Path] = []
        for pdf_path in pdf_paths:
            source = Path(pdf_path)
            target = paths["pdfs"] / source.name
            if source.resolve() != target.resolve():
                copy_file(source, target)
            local_pdfs.append(target)

        extractor = DoclingExtractor(config)
        translator = self._get_translator(config)
        captioner = self._get_captioner(config)

        all_chunks: list[ChunkRecord] = []
        table_ordinal = 1
        image_ordinal = 1

        for pdf_path in local_pdfs:
            asset = extractor.extract(pdf_path, paths["artifacts"])
            doc_text_chunks = make_text_chunks(
                document_id=asset.document_id,
                source_path=asset.source_path,
                pdf_type=asset.pdf_type,
                nodes=asset.raw_text_nodes,
                chunk_size=config.text_chunk_size,
                overlap=config.text_chunk_overlap,
            )
            for chunk in doc_text_chunks:
                all_chunks.append(self._bilingualize_chunk(chunk, translator))

            # Generic extra chunks for ordered structures such as flow charts,
            # process steps, timelines, numbered lists, and cycles. This is not
            # tied to a specific PDF; it improves questions like "প্রথম ধাপ কী",
            # "শেষ ধাপ কী", "ধাপগুলো কী", and timeline/process questions.
            structure_ordinal = 1
            for page_no, heading, page_text in group_text_nodes_by_page(asset.raw_text_nodes):
                structure_chunks = make_structure_chunks(
                    document_id=asset.document_id,
                    source_path=asset.source_path,
                    page=page_no,
                    pdf_type=asset.pdf_type,
                    heading=heading,
                    text=page_text,
                    ordinal_start=structure_ordinal,
                )
                for structure_chunk in structure_chunks:
                    all_chunks.append(self._bilingualize_chunk(structure_chunk, translator))
                structure_ordinal += len(structure_chunks)

            for table in asset.tables:
                if not normalize_text(table.get("table_markdown", "")):
                    continue
                table_chunks = make_table_chunks(
                    source_path=table["source_path"],
                    document_id=table["document_id"],
                    page=table["page"],
                    pdf_type=asset.pdf_type,
                    heading=table.get("heading", ""),
                    table_markdown=table.get("table_markdown", ""),
                    image_path=table.get("image_path", ""),
                    ordinal_start=table_ordinal,
                )
                for table_chunk in table_chunks:
                    all_chunks.append(self._bilingualize_chunk(table_chunk, translator))
                table_ordinal += len(table_chunks)

            for picture in asset.pictures:
                caption = normalize_text(picture.get("docling_description") or picture.get("docling_caption") or "")
                ocr_text = ""
                if config.enable_image_ocr and picture.get("image_path"):
                    try:
                        ocr_text = ocr_image_to_text(picture["image_path"], self._get_image_ocr_reader())
                    except Exception as exc:
                        LOGGER.warning("Image OCR failed for %s: %s", picture.get("image_path"), exc)
                if not caption and config.enable_picture_description and picture.get("image_path"):
                    try:
                        caption = captioner.caption(picture["image_path"])
                    except Exception as exc:
                        LOGGER.warning("Captioning failed for %s: %s", picture.get("image_path"), exc)
                        caption = ""
                if not caption:
                    caption = "Image extracted from the document."

                image_chunk = make_image_chunk(
                    source_path=picture["source_path"],
                    document_id=picture["document_id"],
                    page=picture["page"],
                    pdf_type=asset.pdf_type,
                    heading=picture.get("heading", ""),
                    image_path=picture.get("image_path", ""),
                    caption=caption,
                    ordinal=image_ordinal,
                    ocr_text=ocr_text,
                )
                all_chunks.append(self._bilingualize_chunk(image_chunk, translator))
                image_ordinal += 1

        write_jsonl(paths["corpus_jsonl"], [chunk.to_dict() for chunk in all_chunks])

        index_locations: dict[str, dict[str, str]] = {}
        failed_indexes: dict[str, dict[str, str]] = {}
        for model_name in config.embedding_models:
            index_locations[model_name] = {}
            for mode in ("direct", "translated"):
                index_dir = paths["indexes"] / slugify(model_name) / mode
                try:
                    self.build_index(
                        chunks=all_chunks,
                        index_dir=index_dir,
                        embedding_model_name=model_name,
                        embedding_device=config.embedding_device,
                        ollama_base_url=config.ollama_base_url,
                        mode=mode,
                    )
                    index_locations[model_name][mode] = str(index_dir)
                except Exception as exc:
                    LOGGER.exception("Embedding index failed for model=%s mode=%s", model_name, mode)
                    failed_indexes.setdefault(model_name, {})[mode] = str(exc)

        # Keep only models that produced at least one usable index. This lets the
        # project build continue when one comparison model fails, e.g. an Ollama
        # embedding model returning NaN.
        index_locations = {model: modes for model, modes in index_locations.items() if modes}
        if failed_indexes:
            write_json(paths["results"] / "failed_indexes.json", failed_indexes)

        summary = BuildSummary(
            project_dir=str(paths["root"]),
            corpus_path=str(paths["corpus_jsonl"]),
            total_documents=len(local_pdfs),
            total_chunks=len(all_chunks),
            index_locations=index_locations,
        )
        write_json(paths["results"] / "build_summary.json", {
            "project_dir": summary.project_dir,
            "corpus_path": summary.corpus_path,
            "total_documents": summary.total_documents,
            "total_chunks": summary.total_chunks,
            "index_locations": summary.index_locations,
            "failed_indexes": failed_indexes,
        })
        return summary

    def build_index(
        self,
        *,
        chunks: list[ChunkRecord],
        index_dir: str | Path,
        embedding_model_name: str,
        embedding_device: str,
        ollama_base_url: str,
        mode: str,
    ) -> None:
        if mode not in {"direct", "translated"}:
            raise ValueError(f"Unsupported index mode: {mode}")
        if mode == "direct":
            texts = [
                chunk.retrieval_text_bn
                or chunk.text_bn
                or chunk.retrieval_text_en
                or chunk.text_en
                or ""
                for chunk in chunks
            ]
        else:
            texts = [
                chunk.retrieval_text_en
                or chunk.text_en
                or chunk.retrieval_text_bn
                or chunk.text_bn
                or ""
                for chunk in chunks
            ]
        embedder = self._get_embedder(embedding_model_name, embedding_device, ollama_base_url)
        embeddings = embedder.encode_passages(texts)
        store = self._get_vector_store(index_dir)
        store.save(
            chunks=chunks,
            embeddings=embeddings,
            metadata=IndexMetadata(mode=mode, embedding_model=embedding_model_name, chunk_count=len(chunks)),
        )

    def load_corpus(self, project_dir: str | Path) -> list[ChunkRecord]:
        corpus_path = self.project_paths(project_dir)["corpus_jsonl"]
        return [ChunkRecord.from_dict(row) for row in iter_jsonl(corpus_path)]

    def _index_dir_for(self, project_dir: str | Path, embedding_model: str, mode: str) -> Path:
        return self.project_paths(project_dir)["indexes"] / slugify(embedding_model) / mode

    def retrieve(
        self,
        *,
        project_dir: str | Path,
        question: str,
        embedding_model: str,
        mode: str,
        top_k: int | None = None,
    ) -> tuple[str, list[RetrievedChunk]]:
        config = load_project_config(self.project_paths(project_dir)["config"])
        if mode not in {"direct", "translated"}:
            raise ValueError("mode must be either 'direct' or 'translated'")

        translator = self._get_translator(config)
        retrieval_query = normalize_text(question)
        if mode == "translated":
            if contains_bangla(retrieval_query):
                retrieval_query = translator.translate(retrieval_query, "bn", "en")
        else:
            if not contains_bangla(retrieval_query):
                retrieval_query = translator.translate(retrieval_query, "en", "bn")

        index_dir = self._index_dir_for(project_dir, embedding_model, mode)
        store = self._get_vector_store(index_dir)
        embedder = self._get_embedder(embedding_model, config.embedding_device, config.ollama_base_url)
        query_vector = embedder.encode_queries([retrieval_query])[0]
        requested_top_k = top_k or config.query_top_k
        results = store.search(query_vector, top_k=requested_top_k)
        return retrieval_query, results

    def answer(
        self,
        *,
        project_dir: str | Path,
        question: str,
        embedding_model: str,
        mode: str,
        answer_language: str | None = None,
        top_k: int | None = None,
        ollama_model: str | None = None,
        ollama_base_url: str | None = None,
    ) -> AnswerResult:
        config = load_project_config(self.project_paths(project_dir)["config"])
        answer_language = answer_language or config.answer_language
        query_for_retrieval, retrieved = self.retrieve(
            project_dir=project_dir,
            question=question,
            embedding_model=embedding_model,
            mode=mode,
            top_k=top_k,
        )
        retrieved = self._enrich_image_chunks(retrieved, config)
        base_url = ollama_base_url or config.ollama_base_url
        # Precise RAG generation path:
        # Retrieved chunks are passed directly to the LLM for final answer generation.
        generator = OllamaGenerator(
            base_url=base_url,
            model_name=ollama_model or config.ollama_model,
            temperature=config.temperature,
        )
        answer = generator.generate(question=question, retrieved_chunks=retrieved, answer_language=answer_language)
        answer = clean_bangla_spelling(answer)
        if not normalize_text(answer):
            answer = "প্রদত্ত নথিতে এর উত্তর পাওয়া যায়নি।"
        return AnswerResult(
            mode=mode,
            answer=answer,
            query_for_retrieval=query_for_retrieval,
            retrieved_chunks=retrieved,
        )

    def compare_answer(
        self,
        *,
        project_dir: str | Path,
        question: str,
        embedding_model: str,
        answer_language: str | None = None,
        top_k: int | None = None,
        ollama_model: str | None = None,
        ollama_base_url: str | None = None,
    ) -> CompareAnswerResult:
        direct = self.answer(
            project_dir=project_dir,
            question=question,
            embedding_model=embedding_model,
            mode="direct",
            answer_language=answer_language,
            top_k=top_k,
            ollama_model=ollama_model,
            ollama_base_url=ollama_base_url,
        )
        translated = self.answer(
            project_dir=project_dir,
            question=question,
            embedding_model=embedding_model,
            mode="translated",
            answer_language=answer_language,
            top_k=top_k,
            ollama_model=ollama_model,
            ollama_base_url=ollama_base_url,
        )
        return CompareAnswerResult(question=question, direct=direct, translated=translated)
