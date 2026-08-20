from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.config import ProjectConfig
from src.pipeline import BanglaMultimodalRAGPipeline
from src.utils import ensure_dir, normalize_text, slugify

WORKSPACE = Path("workspace") / "projects"
AVAILABLE_EMBEDDINGS = [
    "nomic-embed-text-v2-moe:latest",
    "BAAI/bge-m3",
    "intfloat/multilingual-e5-base",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]
AVAILABLE_LLMS = [
    "qwen3.5:latest",
    "qwen2.5:3b-instruct",
    "llama3-chatqa:latest",
    "mistral:7b-instruct-q4_K_M",
]


def project_dir_for(name: str) -> Path:
    return ensure_dir(WORKSPACE / slugify(name or "bangla_docling_rag"))


def save_uploaded_pdfs(uploaded_files, target_dir: Path) -> list[Path]:
    ensure_dir(target_dir)
    saved: list[Path] = []
    for file in uploaded_files:
        out = target_dir / file.name
        out.write_bytes(file.getbuffer())
        saved.append(out)
    return saved


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def built_models(project_dir: Path) -> list[str]:
    summary = load_json(project_dir / "results" / "build_summary.json", {})
    return list((summary.get("index_locations") or {}).keys())


def existing_projects() -> list[str]:
    projects = []
    if WORKSPACE.exists():
        for path in sorted(WORKSPACE.iterdir()):
            if path.is_dir() and (path / "results" / "build_summary.json").exists():
                projects.append(path.name)
    return projects


def modes_for_model(project_dir: Path, model: str) -> list[str]:
    summary = load_json(project_dir / "results" / "build_summary.json", {})
    return list(((summary.get("index_locations") or {}).get(model) or {}).keys())


def render_answer(result) -> None:
    st.subheader("Generated answer")
    if normalize_text(result.answer):
        st.success(result.answer)
    else:
        st.warning("The LLM returned an empty answer.")
    st.caption(f"Retrieval mode: {result.mode} | Query used for retrieval: {result.query_for_retrieval}")


def render_retrieved(result) -> None:
    st.subheader(f"Retrieved chunks — {result.mode}")
    for item in result.retrieved_chunks:
        c = item.chunk
        with st.expander(f"#{item.rank} | {Path(c.source_path).name} | p.{c.page} | {c.modality} | score={item.score:.3f}"):
            st.markdown(f"**Heading:** {c.heading or '-'}")
            if c.text_bn:
                st.markdown("**Bangla context**")
                st.write(c.text_bn)
            if c.modality == "image":
                ocr_text = c.metadata.get("ocr_text") or c.text_bn or c.text_en
                if ocr_text:
                    st.markdown("**Chart/image OCR**")
                    st.write(ocr_text)
            if c.text_en:
                st.markdown("**English/translated context**")
                st.write(c.text_en)
            if c.raw_table_md:
                st.markdown("**Raw table markdown**")
                st.code(c.raw_table_md)
            if c.image_path:
                image_path = Path(c.image_path)
                # Some extracted chunks may contain placeholder paths like "." or a folder.
                # Streamlit crashes if st.image() receives those, so render only real image files.
                if str(image_path).strip() not in {"", "."} and image_path.exists() and image_path.is_file():
                    st.image(str(image_path), caption=f"image chunk p.{c.page}")
            st.caption(f"document_id={c.document_id}, source_file={Path(c.source_path).name}")


def render_extraction_quality(project_dir: Path) -> None:
    records = []
    for path in sorted((project_dir / "artifacts").glob("*/extraction_quality.json")):
        payload = load_json(path, {})
        records.append({
            "file": path.parent.name,
            "pdf_type": payload.get("pdf_type", ""),
            "strategy": payload.get("ocr_strategy", ""),
            "native_score": payload.get("native_score", ""),
            "ocr_score": payload.get("ocr_score", ""),
            "final_score": payload.get("final_score", ""),
            "pages_using_ocr": payload.get("pages_using_ocr", ""),
            "pages_total": payload.get("pages_total", ""),
        })
    with st.expander("Extraction Quality Viewer", expanded=False):
        if records:
            st.dataframe(pd.DataFrame(records), use_container_width=True)
        else:
            st.info("No extraction_quality.json found yet. Build the project first.")


def append_qa_log(project_dir: Path, payload: dict[str, Any]) -> None:
    results_dir = ensure_dir(project_dir / "results")
    path = results_dir / "qa_log.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def result_to_log(question: str, embedding_model: str, llm_model: str, result) -> dict[str, Any]:
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "embedding_model": embedding_model,
        "llm_model": llm_model,
        "retrieval_mode": result.mode,
        "query_for_retrieval": result.query_for_retrieval,
        "answer": result.answer,
        "retrieved": [
            {
                "rank": item.rank,
                "score": item.score,
                "source_file": Path(item.chunk.source_path).name,
                "page": item.chunk.page,
                "modality": item.chunk.modality,
                "heading": item.chunk.heading,
                "chunk_id": item.chunk.chunk_id,
            }
            for item in result.retrieved_chunks
        ],
    }


def main() -> None:
    st.set_page_config(page_title="Bangla Multimodal RAG", layout="wide")
    st.title("Local Bangla Multimodal RAG ")
    st.caption("Docling extraction → chunk embedding → vector store → question embedding → retrieval → LLM answer → evaluation")

    pipeline = BanglaMultimodalRAGPipeline()

    with st.sidebar:
        st.header("Project")
        existing = existing_projects()
        selected_existing = st.selectbox("Open existing project", [""] + existing, index=0)
        default_project_name = selected_existing or "rag_test"
        project_name = st.text_input("Project name", value=default_project_name)
        project_dir = project_dir_for(project_name)
        if selected_existing and selected_existing != project_name:
            st.caption(f"Using existing project: {project_dir}")
        else:
            st.caption(str(project_dir))

        st.header("Build settings")
        uploaded = st.file_uploader("Upload Bangla PDFs", type=["pdf"], accept_multiple_files=True)
        embedding_models = st.multiselect("Embedding models to build", AVAILABLE_EMBEDDINGS, default=AVAILABLE_EMBEDDINGS[:2])
        ocr_strategy = st.selectbox("OCR strategy", ["auto_best", "native_only", "force_selected_ocr", "compare_easyocr_tesseract"], index=0)
        ocr_engine = st.selectbox("Selected OCR engine", ["easyocr", "tesseract", "rapidocr", "none"], index=0)
        min_quality = st.slider("Bangla extraction quality threshold", 0.10, 0.90, 0.55, 0.05)
        image_scale = st.slider("Docling image scale", 1.0, 4.0, 3.0, 0.5)
        enable_image_caption = st.checkbox("Generate image captions for image chunks (recommended)", value=True)
        caption_backend = st.selectbox(
            "Image caption backend",
            ["ollama_vision", "blip"],
            index=0,
            help=(
                "ollama_vision: uses a local Ollama vision model (e.g. llava) to read and describe "
                "charts/diagrams/tables/photos in Bangla - works for any PDF. "
                "blip: offline English-only fallback, weaker for Bangla charts."
            ),
        )
        vision_model = st.text_input("Ollama vision model (for ollama_vision backend)", value="llava:latest")
        enable_image_ocr = st.checkbox("OCR on chart/image chunks (recommended)", value=True)

        # Keep only top_k here (used during build). Move LLM & answer settings to main UI after build.
        st.header("Answer settings")
        top_k = st.slider("Top-k retrieved chunks passed to LLM", 1, 10, 5)
        ollama_base_url_build = st.text_input("Local Ollama base URL", value="http://localhost:11434")

    # Ensure required values exist before the build path uses them.
    answer_language = "bn"
    llm_model = AVAILABLE_LLMS[0] if AVAILABLE_LLMS else ""

    if st.sidebar.button("Build / Rebuild project", type="primary"):
        if not uploaded:
            st.error("Upload at least one PDF first.")
        elif not embedding_models:
            st.error("Select at least one embedding model.")
        else:
            pdf_dir = project_dir / "data" / "uploaded"
            saved_files = save_uploaded_pdfs(uploaded, pdf_dir)
            config = ProjectConfig(
                project_name=project_name,
                embedding_models=embedding_models,
                ocr_engine=ocr_engine,
                ocr_strategy=ocr_strategy,
                auto_ocr_on_bad_bangla=(ocr_strategy == "auto_best"),
                force_full_page_ocr=(ocr_strategy == "force_selected_ocr"),
                min_bangla_quality=min_quality,
                images_scale=image_scale,
                enable_picture_description=enable_image_caption,
                caption_backend=caption_backend,
                vision_model=vision_model,
                enable_image_ocr=enable_image_ocr,
                query_top_k=top_k,
                answer_language=answer_language,
                ollama_model=llm_model,
                ollama_base_url=ollama_base_url_build,
            )
            with st.spinner("Extracting PDFs, building chunks, embedding, and saving vector indexes..."):
                try:
                    summary = pipeline.build_project(saved_files, project_dir=project_dir, config=config)
                    st.success(f"Build complete: {summary.total_documents} PDFs, {summary.total_chunks} chunks.")
                    st.json(summary.index_locations)
                except Exception as exc:
                    st.error(f"Build failed: {exc}")

    render_extraction_quality(project_dir)

    models = built_models(project_dir)
    if not models:
        st.info("Build a project first, then ask questions.")
        return

    # Answer settings are shown after a project is available so users can
    # pick which local Ollama LLM(s) to use for generating answers.
    st.header("Answer settings")
    compare_llms = st.checkbox("Compare multiple Ollama LLMs", value=False)
    selected_llms = []
    if compare_llms:
        selected_llms = st.multiselect("LLMs to compare", AVAILABLE_LLMS, default=AVAILABLE_LLMS[:2])
        if not selected_llms:
            st.warning("Pick at least one LLM to compare.")
    llm_model = st.selectbox("Local Ollama LLM", AVAILABLE_LLMS, index=0, disabled=compare_llms)
    answer_language = st.radio("Answer language", ["bn", "en"], horizontal=True)
    ollama_base_url = st.text_input("Local Ollama base URL", value="http://localhost:11434")

    st.header("Question answering")
    question = st.text_input("Question", placeholder="যেমন: গ্রীষ্ম ঋতুর বাংলা মাস কোন দুটি?")
    col1, col2, col3 = st.columns(3)
    with col1:
        embedding_model = st.selectbox("Embedding model", models)
    with col2:
        available_modes = modes_for_model(project_dir, embedding_model)
        mode_options = [m for m in ["direct", "translated"] if m in available_modes]
        retrieval_mode = st.radio("Retrieval mode", mode_options + ["compare"] if len(mode_options) == 2 else mode_options, horizontal=True)
    with col3:
        compare_embeddings = st.checkbox("Compare embedding models", value=False)

    ask_disabled = not question.strip() or (compare_llms and not selected_llms)
    if compare_llms and not selected_llms:
        st.warning("Select at least one LLM to compare before asking.")

    if st.button("Ask", type="primary", disabled=ask_disabled):
        try:
            if compare_embeddings and compare_llms:
                st.error("Compare embeddings and compare LLMs cannot be used at the same time. Uncheck one option.")
                return

            if compare_llms:
                st.subheader("LLM comparison")
                rows = []
                for llm in selected_llms:
                    with st.spinner(f"Running {llm}..."):
                        if retrieval_mode == "compare":
                            compare = pipeline.compare_answer(
                                project_dir=project_dir,
                                question=question,
                                embedding_model=embedding_model,
                                answer_language=answer_language,
                                top_k=top_k,
                                ollama_model=llm,
                                ollama_base_url=ollama_base_url,
                            )
                            rows.append({
                                "llm_model": llm,
                                "mode": "direct",
                                "answer": compare.direct.answer,
                                "top1_source": Path(compare.direct.retrieved_chunks[0].chunk.source_path).name if compare.direct.retrieved_chunks else "",
                                "top1_page": compare.direct.retrieved_chunks[0].chunk.page if compare.direct.retrieved_chunks else "",
                                "top1_modality": compare.direct.retrieved_chunks[0].chunk.modality if compare.direct.retrieved_chunks else "",
                            })
                            rows.append({
                                "llm_model": llm,
                                "mode": "translated",
                                "answer": compare.translated.answer,
                                "top1_source": Path(compare.translated.retrieved_chunks[0].chunk.source_path).name if compare.translated.retrieved_chunks else "",
                                "top1_page": compare.translated.retrieved_chunks[0].chunk.page if compare.translated.retrieved_chunks else "",
                                "top1_modality": compare.translated.retrieved_chunks[0].chunk.modality if compare.translated.retrieved_chunks else "",
                            })
                            append_qa_log(project_dir, result_to_log(question, embedding_model, llm, compare.direct))
                            append_qa_log(project_dir, result_to_log(question, embedding_model, llm, compare.translated))
                        else:
                            result = pipeline.answer(
                                project_dir=project_dir,
                                question=question,
                                embedding_model=embedding_model,
                                mode=retrieval_mode,
                                answer_language=answer_language,
                                top_k=top_k,
                                ollama_model=llm,
                                ollama_base_url=ollama_base_url,
                            )
                            top = result.retrieved_chunks[0].chunk if result.retrieved_chunks else None
                            rows.append({
                                "llm_model": llm,
                                "mode": result.mode,
                                "answer": result.answer,
                                "top1_source": Path(top.source_path).name if top else "",
                                "top1_page": top.page if top else "",
                                "top1_modality": top.modality if top else "",
                            })
                            append_qa_log(project_dir, result_to_log(question, embedding_model, llm, result))
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
                return

            if compare_embeddings:
                st.subheader("Embedding comparison")
                rows = []
                for model in models:
                    modes = modes_for_model(project_dir, model)
                    run_mode = "direct" if "direct" in modes else modes[0]
                    with st.spinner(f"Running {model} / {run_mode}..."):
                        result = pipeline.answer(
                            project_dir=project_dir,
                            question=question,
                            embedding_model=model,
                            mode=run_mode,
                            answer_language=answer_language,
                            top_k=top_k,
                            ollama_model=llm_model,
                            ollama_base_url=ollama_base_url,
                        )
                    top = result.retrieved_chunks[0].chunk if result.retrieved_chunks else None
                    rows.append({
                        "embedding_model": model,
                        "mode": run_mode,
                        "answer": result.answer,
                        "top1_source": Path(top.source_path).name if top else "",
                        "top1_page": top.page if top else "",
                        "top1_modality": top.modality if top else "",
                    })
                    append_qa_log(project_dir, result_to_log(question, model, llm_model, result))
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
                return

            if retrieval_mode == "compare":
                st.subheader("Translation comparison: direct Bangla vs translated-English retrieval")
                compare = pipeline.compare_answer(
                    project_dir=project_dir,
                    question=question,
                    embedding_model=embedding_model,
                    answer_language=answer_language,
                    top_k=top_k,
                    ollama_model=llm_model,
                    ollama_base_url=ollama_base_url,
                )
                tab1, tab2 = st.tabs(["Direct Bangla", "Translated-English"])
                with tab1:
                    render_answer(compare.direct)
                    render_retrieved(compare.direct)
                    append_qa_log(project_dir, result_to_log(question, embedding_model, llm_model, compare.direct))
                with tab2:
                    render_answer(compare.translated)
                    render_retrieved(compare.translated)
                    append_qa_log(project_dir, result_to_log(question, embedding_model, llm_model, compare.translated))
            else:
                with st.spinner("Embedding question, retrieving chunks, and generating answer with LLM..."):
                    result = pipeline.answer(
                        project_dir=project_dir,
                        question=question,
                        embedding_model=embedding_model,
                        mode=retrieval_mode,
                        answer_language=answer_language,
                        top_k=top_k,
                        ollama_model=llm_model,
                        ollama_base_url=ollama_base_url,
                    )
                render_answer(result)
                render_retrieved(result)
                append_qa_log(project_dir, result_to_log(question, embedding_model, llm_model, result))
                st.success("QA result saved to results/qa_log.jsonl")
        except RuntimeError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")


if __name__ == "__main__":
    main()
