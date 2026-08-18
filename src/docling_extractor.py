from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ProjectConfig
from .schemas import RawTextNode
from .text_processing import clean_bangla_spelling
from .utils import (
    bangla_extraction_stats,
    bangla_text_quality_score,
    detect_pdf_type,
    ensure_dir,
    estimate_native_text_ratio,
    is_bad_bangla_extraction,
    normalize_text,
    safe_truncate,
)

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ExtractedAsset:
    document_id: str
    source_path: str
    pdf_type: str
    raw_text_nodes: list[RawTextNode]
    tables: list[dict[str, Any]]
    pictures: list[dict[str, Any]]
    markdown_path: str
    artifact_dir: str


@dataclass(slots=True)
class _ExtractionCandidate:
    name: str
    doc: Any
    markdown: str
    raw_text_nodes: list[RawTextNode]
    tables: list[dict[str, Any]]
    pictures: list[dict[str, Any]]
    doc_stats: dict[str, Any]
    page_stats: dict[int, dict[str, Any]]


class DoclingExtractor:
    """Docling-first PDF extractor with extraction-quality validation.

    Important idea for Bangla RAG:
    good retrieval and good LLM answers are only possible when the indexed text is
    readable. So this extractor does not blindly trust native PDF text. It first
    checks extraction quality. If the PDF text looks corrupted, it runs full-page
    OCR and chooses the better text per page.
    """

    def __init__(self, config: ProjectConfig):
        self.config = config

    def _build_converter(self, force_full_page_ocr: bool | None = None, ocr_engine: str | None = None):
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                EasyOcrOptions,
                PdfPipelineOptions,
                TableStructureOptions,
                TesseractCliOcrOptions,
            )
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Docling is not installed. Install requirements first.") from exc

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = self.config.ocr_engine.lower() != "none"
        pipeline_options.do_table_structure = True
        try:
            from docling.datamodel.pipeline_options import TableFormerMode
            pipeline_options.table_structure_options = TableStructureOptions(
                do_cell_matching=True,
                mode=TableFormerMode.ACCURATE,
            )
        except Exception:
            pipeline_options.table_structure_options = TableStructureOptions(do_cell_matching=True)

        pipeline_options.images_scale = self.config.images_scale
        pipeline_options.generate_page_images = self.config.generate_page_images
        pipeline_options.generate_picture_images = self.config.generate_picture_images
        pipeline_options.do_picture_description = self.config.enable_picture_description

        engine = (ocr_engine or self.config.ocr_engine).lower()
        force_ocr = self.config.force_full_page_ocr if force_full_page_ocr is None else bool(force_full_page_ocr)
        if engine == "easyocr":
            pipeline_options.ocr_options = EasyOcrOptions(
                lang=["bn", "en"],
                force_full_page_ocr=force_ocr,
                bitmap_area_threshold=self.config.bitmap_area_threshold,
            )
        elif engine == "tesseract":
            pipeline_options.ocr_options = TesseractCliOcrOptions(
                lang=["ben", "eng"],
                force_full_page_ocr=force_ocr,
                bitmap_area_threshold=self.config.bitmap_area_threshold,
            )
        elif engine == "rapidocr":
            try:
                from docling.datamodel.pipeline_options import RapidOcrOptions
            except Exception as exc:
                raise RuntimeError(
                    "RapidOCR is not available in this Docling installation. "
                    "Install Docling with RapidOCR support or choose EasyOCR/Tesseract."
                ) from exc
            pipeline_options.ocr_options = RapidOcrOptions(
                force_full_page_ocr=force_ocr,
                bitmap_area_threshold=self.config.bitmap_area_threshold,
            )
        elif engine == "auto":
            from docling.datamodel.pipeline_options import OcrAutoOptions
            pipeline_options.ocr_options = OcrAutoOptions(
                force_full_page_ocr=force_ocr,
                bitmap_area_threshold=self.config.bitmap_area_threshold,
            )
        elif engine == "none":
            pipeline_options.do_ocr = False
        else:
            raise ValueError(f"Unsupported OCR engine: {self.config.ocr_engine}")

        return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})

    @staticmethod
    def _get_page_no(item: Any) -> int:
        if hasattr(item, "page_no"):
            try:
                return int(getattr(item, "page_no"))
            except Exception:
                pass
        prov = getattr(item, "prov", None)
        if prov:
            candidate = prov[0] if isinstance(prov, (list, tuple)) and prov else prov
            if hasattr(candidate, "page_no"):
                try:
                    return int(getattr(candidate, "page_no"))
                except Exception:
                    pass
        return -1

    @staticmethod
    def _class_name(item: Any) -> str:
        return item.__class__.__name__

    @staticmethod
    def _text_by_page(nodes: list[RawTextNode]) -> dict[int, str]:
        pages: dict[int, list[str]] = {}
        for node in nodes:
            if node.page < 1:
                continue
            pages.setdefault(node.page, []).append(node.text)
        return {page: " ".join(parts) for page, parts in pages.items()}

    def _save_page_images(self, doc: Any, output_dir: Path, document_id: str) -> None:
        if not self.config.generate_page_images:
            return
        pages_dir = ensure_dir(output_dir / "pages")
        for page_no, page in getattr(doc, "pages", {}).items():
            try:
                page_no = getattr(page, "page_no", page_no)
                image_path = pages_dir / f"{document_id}-page-{page_no}.png"
                with image_path.open("wb") as fp:
                    page.image.pil_image.save(fp, format="PNG")
            except Exception as exc:
                LOGGER.warning("Could not export page image %s page %s: %s", document_id, page_no, exc)

    def _convert_once(self, pdf_path: Path, force_ocr: bool | None, ocr_engine: str | None = None):
        converter = self._build_converter(force_full_page_ocr=force_ocr, ocr_engine=ocr_engine)
        engine_note = f" engine={ocr_engine or self.config.ocr_engine}"
        LOGGER.info("Converting %s with Docling%s%s", pdf_path.name, " using forced full-page OCR" if force_ocr else "", engine_note)
        conv_res = converter.convert(str(pdf_path))
        doc_obj = conv_res.document
        try:
            md = doc_obj.export_to_markdown(page_break_placeholder="\n\n[PAGE_BREAK]\n\n")
        except TypeError:
            md = doc_obj.export_to_markdown()
        return doc_obj, md

    def _collect_doc_content(
        self,
        *,
        doc: Any,
        pdf_path: Path,
        document_id: str,
        artifact_dir: Path,
        candidate_name: str,
    ) -> tuple[list[RawTextNode], list[dict[str, Any]], list[dict[str, Any]]]:
        tables_dir = ensure_dir(artifact_dir / f"tables_{candidate_name}")
        pictures_dir = ensure_dir(artifact_dir / f"pictures_{candidate_name}")
        raw_text_nodes: list[RawTextNode] = []
        tables: list[dict[str, Any]] = []
        pictures: list[dict[str, Any]] = []
        current_heading = ""
        table_counter = 0
        picture_counter = 0

        try:
            from docling_core.types.doc import PictureItem, TableItem
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("docling-core could not be imported.") from exc

        for element, _level in doc.iterate_items():
            class_name = self._class_name(element)
            page_no = self._get_page_no(element)

            if class_name.lower().endswith("sectionheaderitem") or "heading" in class_name.lower():
                heading_text = normalize_text(getattr(element, "text", ""))
                if heading_text:
                    current_heading = heading_text
                    raw_text_nodes.append(RawTextNode(
                        document_id=document_id,
                        source_path=str(pdf_path),
                        page=page_no,
                        text=heading_text,
                        heading=current_heading,
                        label=f"{candidate_name}:heading",
                    ))
                continue

            if isinstance(element, TableItem):
                table_counter += 1
                table_df: pd.DataFrame | None = None
                table_md = ""
                table_html = ""
                try:
                    table_df = element.export_to_dataframe(doc=doc)
                    table_md = clean_bangla_spelling(table_df.to_markdown(index=False))
                    table_html = element.export_to_html(doc=doc)
                except Exception as exc:
                    LOGGER.warning("Could not export table %s #%s (%s): %s", document_id, table_counter, candidate_name, exc)

                csv_path = tables_dir / f"{document_id}-{candidate_name}-table-{table_counter}.csv"
                html_path = tables_dir / f"{document_id}-{candidate_name}-table-{table_counter}.html"
                img_path = tables_dir / f"{document_id}-{candidate_name}-table-{table_counter}.png"
                if table_df is not None:
                    table_df.to_csv(csv_path, index=False)
                if table_html:
                    html_path.write_text(table_html, encoding="utf-8")
                table_image_path = ""
                try:
                    with img_path.open("wb") as fp:
                        element.get_image(doc).save(fp, "PNG")
                    table_image_path = str(img_path)
                except Exception:
                    table_image_path = ""

                tables.append({
                    "document_id": document_id,
                    "source_path": str(pdf_path),
                    "page": page_no,
                    "heading": clean_bangla_spelling(current_heading),
                    "table_markdown": table_md,
                    "csv_path": str(csv_path) if csv_path.exists() else "",
                    "html_path": str(html_path) if html_path.exists() else "",
                    "image_path": table_image_path,
                    "summary_hint": safe_truncate(table_md, 400),
                    "extraction_source": candidate_name,
                })
                continue

            if isinstance(element, PictureItem):
                picture_counter += 1
                img_path = pictures_dir / f"{document_id}-{candidate_name}-picture-{picture_counter}.png"
                try:
                    with img_path.open("wb") as fp:
                        element.get_image(doc).save(fp, "PNG")
                except Exception as exc:
                    LOGGER.warning("Could not export picture %s #%s (%s): %s", document_id, picture_counter, candidate_name, exc)
                    continue

                try:
                    caption = normalize_text(element.caption_text(doc=doc))
                except Exception:
                    caption = ""
                description = ""
                meta = getattr(element, "meta", None)
                if meta is not None:
                    desc = getattr(meta, "description", None)
                    if desc is not None:
                        description = normalize_text(getattr(desc, "text", ""))

                pictures.append({
                    "document_id": document_id,
                    "source_path": str(pdf_path),
                    "page": page_no,
                    "heading": clean_bangla_spelling(current_heading),
                    "image_path": str(img_path),
                    "docling_caption": clean_bangla_spelling(caption),
                    "docling_description": clean_bangla_spelling(description),
                    "extraction_source": candidate_name,
                })
                continue

            text_value = normalize_text(getattr(element, "text", ""))
            if text_value:
                raw_text_nodes.append(RawTextNode(
                    document_id=document_id,
                    source_path=str(pdf_path),
                    page=page_no,
                    text=text_value,
                    heading=current_heading,
                    label=f"{candidate_name}:{class_name.lower()}",
                ))
        return raw_text_nodes, tables, pictures

    def _make_candidate(self, pdf_path: Path, document_id: str, artifact_dir: Path, name: str, force_ocr: bool | None, ocr_engine: str | None = None) -> _ExtractionCandidate:
        doc, markdown = self._convert_once(pdf_path, force_ocr=force_ocr, ocr_engine=ocr_engine)
        (artifact_dir / f"raw_{name}.md").write_text(markdown, encoding="utf-8")
        nodes, tables, pictures = self._collect_doc_content(
            doc=doc,
            pdf_path=pdf_path,
            document_id=document_id,
            artifact_dir=artifact_dir,
            candidate_name=name,
        )
        page_text = self._text_by_page(nodes)
        page_stats = {page: bangla_extraction_stats(text) for page, text in page_text.items()}
        doc_stats = bangla_extraction_stats("\n".join(page_text.values()) or markdown)
        doc_stats["legacy_quality_score"] = round(bangla_text_quality_score(markdown), 4)
        return _ExtractionCandidate(name, doc, markdown, nodes, tables, pictures, doc_stats, page_stats)

    def extract(self, pdf_path: str | Path, artifact_root: str | Path) -> ExtractedAsset:
        pdf_path = Path(pdf_path)
        document_id = pdf_path.stem
        artifact_dir = ensure_dir(Path(artifact_root) / document_id)

        native_text_ratio = estimate_native_text_ratio(pdf_path)
        pdf_type = detect_pdf_type(native_text_ratio)

        # Native extraction means: Docling reads the PDF text layer/layout without OCR.
        # This is fastest and best for clean Unicode PDFs, but can be very bad for
        # Bangla PDFs with custom/broken font encoding. Therefore we always score it.
        native = self._make_candidate(
            pdf_path=pdf_path,
            document_id=document_id,
            artifact_dir=artifact_dir,
            name="native",
            force_ocr=False,
            ocr_engine="none",
        )

        strategy = getattr(self.config, "ocr_strategy", "auto_best").lower()
        configured_engine = self.config.ocr_engine.lower()
        native_is_bad = (
            native.doc_stats["quality_score"] < self.config.min_bangla_quality
            or any(st["quality_score"] < self.config.min_bangla_quality for st in native.page_stats.values())
            or is_bad_bangla_extraction(native.markdown, self.config.min_bangla_quality)
            or pdf_type in {"scanned", "mostly_scanned"}
        )

        candidates: list[_ExtractionCandidate] = [native]
        tried_ocr = False

        def add_ocr_candidate(engine: str, name: str) -> None:
            nonlocal tried_ocr
            if engine == "none":
                return
            try:
                cand = self._make_candidate(
                    pdf_path=pdf_path,
                    document_id=document_id,
                    artifact_dir=artifact_dir,
                    name=name,
                    force_ocr=True,
                    ocr_engine=engine,
                )
                candidates.append(cand)
                tried_ocr = True
            except Exception as exc:
                LOGGER.warning("OCR candidate failed for %s using %s: %s", pdf_path.name, engine, exc)

        if strategy == "native_only":
            pass
        elif strategy == "force_selected_ocr" or self.config.force_full_page_ocr:
            add_ocr_candidate(configured_engine, f"ocr_{configured_engine}")
        elif strategy == "compare_easyocr_tesseract":
            # Best for experiments: compare the two most useful Bangla OCR paths.
            # RapidOCR can be selected directly with force_selected_ocr if installed.
            add_ocr_candidate("easyocr", "ocr_easyocr")
            add_ocr_candidate("tesseract", "ocr_tesseract")
        else:
            # auto_best: only run OCR when native extraction looks corrupted/empty.
            if self.config.auto_ocr_on_bad_bangla and configured_engine != "none" and native_is_bad:
                add_ocr_candidate(configured_engine, f"ocr_{configured_engine}")

        selected_source_by_page: dict[int, str] = {}
        selected_nodes: list[RawTextNode] = []
        selected_candidate_for_assets = native

        # Choose best candidate page by page. This is more robust than choosing one
        # OCR engine for the whole PDF because some pages are tables/charts while
        # others are normal paragraphs.
        all_pages: set[int] = set()
        for cand in candidates:
            all_pages |= set(cand.page_stats)
            all_pages |= {n.page for n in cand.raw_text_nodes if n.page > 0}

        for page in sorted(all_pages):
            best = native
            best_stats = native.page_stats.get(page, {"quality_score": 0.0, "bangla_chars": 0, "chars": 0})
            for cand in candidates[1:]:
                cand_stats = cand.page_stats.get(page, {"quality_score": 0.0, "bangla_chars": 0, "chars": 0})
                # Prefer OCR when native is bad and OCR is comparable/better, or
                # when OCR is clearly better and has enough Bangla text.
                native_score = float(best_stats.get("quality_score", 0.0))
                cand_score = float(cand_stats.get("quality_score", 0.0))
                native_bn = float(best_stats.get("bangla_chars", 0.0))
                cand_bn = float(cand_stats.get("bangla_chars", 0.0))
                if self.config.force_full_page_ocr and cand_bn > 0:
                    best, best_stats = cand, cand_stats
                elif native_score < self.config.min_bangla_quality and cand_score >= native_score - 0.03 and cand_bn >= max(5, native_bn * 0.5):
                    best, best_stats = cand, cand_stats
                elif cand_score > native_score + 0.08 and cand_bn > native_bn * 0.75:
                    best, best_stats = cand, cand_stats
            selected_source_by_page[page] = best.name

        for cand in candidates:
            selected_nodes.extend([n for n in cand.raw_text_nodes if selected_source_by_page.get(n.page, "native") == cand.name])
        selected_nodes.sort(key=lambda n: (n.page, n.label))

        # Use tables/pictures from the candidate used by the most pages.
        if selected_source_by_page:
            source_counts: dict[str, int] = {}
            for src in selected_source_by_page.values():
                source_counts[src] = source_counts.get(src, 0) + 1
            main_source = max(source_counts.items(), key=lambda kv: kv[1])[0]
            selected_candidate_for_assets = next((c for c in candidates if c.name == main_source), native)
        # Clean selected text before chunking. Raw candidate markdown remains saved for audit.
        cleaned_nodes: list[RawTextNode] = []
        for node in selected_nodes:
            cleaned_nodes.append(RawTextNode(
                document_id=node.document_id,
                source_path=node.source_path,
                page=node.page,
                text=clean_bangla_spelling(node.text),
                heading=clean_bangla_spelling(node.heading),
                label=node.label,
            ))

        # For tables/pictures, choose the candidate used by most selected pages.
        tables = selected_candidate_for_assets.tables
        pictures = selected_candidate_for_assets.pictures
        selected_doc = selected_candidate_for_assets.doc
        selected_markdown = "\n\n".join(self._text_by_page(cleaned_nodes).get(p, "") for p in sorted(self._text_by_page(cleaned_nodes)))
        markdown_path = artifact_dir / f"{document_id}.md"
        markdown_path.write_text(selected_markdown, encoding="utf-8")
        self._save_page_images(selected_doc, artifact_dir, document_id)

        final_page_text = self._text_by_page(cleaned_nodes)
        final_page_stats = {page: bangla_extraction_stats(text) for page, text in final_page_text.items()}
        extraction_report = {
            "document_id": document_id,
            "source_file": pdf_path.name,
            "native_text_ratio": native_text_ratio,
            "pdf_type": pdf_type,
            "ocr_engine": self.config.ocr_engine,
            "ocr_strategy": getattr(self.config, "ocr_strategy", "auto_best"),
            "force_full_page_ocr": self.config.force_full_page_ocr,
            "auto_ocr_on_bad_bangla": self.config.auto_ocr_on_bad_bangla,
            "quality_threshold": self.config.min_bangla_quality,
            "tried_full_page_ocr": tried_ocr,
            "candidate_document_stats": {c.name: c.doc_stats for c in candidates},
            "native_document_stats": native.doc_stats,
            "ocr_document_stats": {c.name: c.doc_stats for c in candidates if c.name != "native"},
            "final_document_stats": bangla_extraction_stats(selected_markdown),
            "selected_source_by_page": {str(k): v for k, v in selected_source_by_page.items()},
            "pages": [],
        }
        all_report_pages: set[int] = set(final_page_stats)
        for c in candidates:
            all_report_pages |= set(c.page_stats)
        for page in sorted(all_report_pages):
            extraction_report["pages"].append({
                "page": page,
                "selected_source": selected_source_by_page.get(page, "native"),
                "candidates": {c.name: c.page_stats.get(page) for c in candidates},
                "native": native.page_stats.get(page),
                "ocr": {c.name: c.page_stats.get(page) for c in candidates if c.name != "native"},
                "final": final_page_stats.get(page),
            })
        (artifact_dir / "extraction_quality.json").write_text(json.dumps(extraction_report, ensure_ascii=False, indent=2), encoding="utf-8")

        return ExtractedAsset(
            document_id=document_id,
            source_path=str(pdf_path),
            pdf_type=pdf_type,
            raw_text_nodes=cleaned_nodes,
            tables=tables,
            pictures=pictures,
            markdown_path=str(markdown_path),
            artifact_dir=str(artifact_dir),
        )
