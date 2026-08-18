# Objective-aligned Bangla Multimodal RAG code map

This version keeps only the steps aligned with the thesis pipeline:

1. Docling extracts Bangla PDF text, tables, and images.
2. Text/table/image-caption chunks are created.
3. Chunks are embedded with the selected embedding model.
4. Embeddings are stored in a vector store.
5. The user question is embedded with the same embedding model.
6. Vector search retrieves top-k chunks.
7. Retrieved chunks are passed to the LLM.
8. The LLM generates the final answer.
9. Direct-vs-translated retrieval, embedding comparison, and QA evaluation logs are available.

## Main files and important lines

Run this command after extraction if line numbers change:

```powershell
python -m compileall src app.py scripts
```

### `src/docling_extractor.py`
- Lines 65-132: builds the Docling converter and OCR options.
- Lines 177-187: runs one Docling conversion pass.
- Lines 189-312: collects text, tables, and pictures from Docling output.
- Lines 330-475: chooses the final extraction output and writes `raw_native.md`, `raw_ocr.md`, final `.md`, and `extraction_quality.json`.

### `src/text_processing.py`
- Lines 178-199: Bangla cleanup/correction entry point.
- Lines 266-306: creates text chunks.
- Lines 396-450: turns table rows into readable facts.
- Lines 452-543: creates table chunks.
- Lines 577-628: creates structure chunks for lists/flows/timelines.
- Lines 630-669: creates image-caption chunks.

### `src/captioner.py`
- Lines 11-34: converts extracted images into text captions when image captioning is enabled.
- The raw image is not embedded directly; the caption text is embedded.

### `src/embeddings.py`
- Lines 39-52: detects local Ollama embedding models.
- Lines 77-177: calls Ollama `/api/embed` for local embeddings.
- Lines 179-219: loads SentenceTransformer/HuggingFace models such as `BAAI/bge-m3`.
- Lines 221-241: embeds queries and document passages with the same model class.

### `src/vector_store.py`
- Lines 28-37: stores chunk embeddings in `embeddings.npy` and chunks in `chunks.jsonl`.
- Lines 45-53: retrieves top-k chunks by cosine similarity.

### `src/pipeline.py`
- Lines 108-250: `build_project(...)` runs extraction → chunking → translation → embedding → vector indexes.
- Lines 252-275: `build_index(...)` embeds chunks and saves indexes.
- Lines 284-312: `retrieve(...)` embeds the question and retrieves chunks from the matching index.
- Lines 314-350: `answer(...)` passes retrieved chunks to the LLM and returns the generated answer.
- Lines 352-380: `compare_answer(...)` runs direct Bangla retrieval and translated-English retrieval for the same question.

### `src/generator.py`
- Lines 65-98: builds the LLM prompt from retrieved chunks.
- Lines 100-134: calls Ollama `/api/generate` and returns the LLM-generated answer.

### `app.py`
- Lines 144-232: project build UI and config.
- Lines 234-321: question answering UI.
- Lines 265-294: embedding comparison.
- Lines 297-315: direct-vs-translated comparison.
- Lines 61-90: generated answer and retrieved chunk display.
- Lines 113-142: QA evaluation logging.

### `scripts/evaluate_metrics.py`
- Single source of truth for Chapter 5: `retrieve_only` (Hit@k/MRR sweep across embedding
  models and modes), `generate` (end-to-end retrieval+LLM run with Exact Match/F1/latency),
  and `aggregate` (rolls both into `results/aggregate/*.json`).

### `scripts/build_human_eval.py`
- Builds `results/human_eval.csv`: single-rater manual judgement (correctness, groundedness,
  Bangla fluency) over every row in `results/generation_eval_{direct,translated}.csv`.

## Removed / not used in this version

- No extractive-first answer generation.
- No best-evidence selector that bypasses the LLM.
- No answer validation gate that changes the final answer.
- No LLM comparison UI in the main app.

The LLM always receives retrieved chunks and generates the final answer.
