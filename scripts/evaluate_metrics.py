#!/usr/bin/env python
"""
Chapter-5 evaluation harness for the Bangla Multimodal RAG pipeline.

This script is the single source of truth for every quantitative number
reported in Chapter 5 (Result Analysis and Discussion): retrieval quality
(Hit@1/3/5, MRR), answer quality (Exact Match, token-level F1), and
computational performance (retrieval / generation / end-to-end latency).
No number in the thesis is hand-typed; every table is regenerated from the
CSV/JSON files this script writes to `results/`.

Two sub-evaluations are run:

1. Retrieval-only sweep (`retrieve_only`): for every (embedding_model, mode)
   pair, every question is retrieved (no LLM call) so the embedding-model
   comparison and the direct-vs-translated retrieval comparison are cheap
   to reproduce or extend.
2. Full generation sweep (`generate`): for a chosen (embedding_model, mode)
   pair, every question is answered end-to-end (retrieval + LLM generation),
   with wall-clock timing split into retrieval-phase and generation-phase,
   and the generated Bangla answer is scored against the gold answer with
   Exact Match and token-level F1.

Usage (from the project root, with the venv activated):

    python scripts/evaluate_metrics.py retrieve_only ^
        --ground-truth data/eval_ground_truth.csv ^
        --out results/retrieval_eval.csv

    python scripts/evaluate_metrics.py generate ^
        --ground-truth data/eval_ground_truth.csv ^
        --embedding-model nomic-embed-text-v2-moe:latest ^
        --mode direct ^
        --llm qwen2.5:3b-instruct ^
        --out results/generation_eval_direct.csv

    python scripts/evaluate_metrics.py aggregate ^
        --retrieval-csv results/retrieval_eval.csv ^
        --generation-csv results/generation_eval_direct.csv results/generation_eval_translated.csv ^
        --out-dir results/aggregate
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PDF_TO_PROJECT = {
    "Science9": "workspace/projects/rag_test1-sciencebook9",
    "Geography": "workspace/projects/rag_test2-geography",
    "piechart": "workspace/projects/rag_test3-pie-chart",
}

_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_PUNCT_RE = re.compile(r"[.,;:!?()\[\]{}\"'০-৯\-]")


def normalize_for_metric(text: str) -> str:
    """Digit-unify (Bangla<->Arabic) and strip punctuation/whitespace so OCR-era
    digit-script mixing does not distort Exact Match / F1 scoring."""
    if not text:
        return ""
    text = text.translate(_BN_DIGITS)
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r"[।,;:!?()\[\]{}\"']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    return normalize_for_metric(text).split()


def exact_match(pred: str, gold: str) -> int:
    return int(normalize_for_metric(pred) != "" and normalize_for_metric(pred) == normalize_for_metric(gold))


def token_f1(pred: str, gold: str) -> float:
    pred_toks = tokenize(pred)
    gold_toks = tokenize(gold)
    if not pred_toks or not gold_toks:
        return 0.0
    common = defaultdict(int)
    for tok in pred_toks:
        common[tok] += 1
    overlap = 0
    gold_counts = defaultdict(int)
    for tok in gold_toks:
        gold_counts[tok] += 1
    for tok in set(pred_toks) & set(gold_toks):
        overlap += min(common[tok], gold_counts[tok])
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_toks)
    recall = overlap / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def load_ground_truth(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def index_exists(project_dir: str, embedding_model: str, mode: str) -> bool:
    from src.utils import slugify
    p = Path(project_dir) / "indexes" / slugify(embedding_model) / mode / "metadata.json"
    return p.exists()


def cmd_retrieve_only(args: argparse.Namespace) -> None:
    from src.pipeline import BanglaMultimodalRAGPipeline

    pipeline = BanglaMultimodalRAGPipeline()
    rows = load_ground_truth(args.ground_truth)
    embedding_models = args.embedding_models
    modes = args.modes

    fieldnames = [
        "question_id", "pdf_id", "modality", "question_type", "embedding_model", "mode",
        "top_k", "retrieval_latency_ms", "gold_chunk_id", "retrieved_chunk_ids",
        "rank_of_gold", "hit_at_1", "hit_at_3", "hit_at_5", "reciprocal_rank", "top1_score",
    ]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            project_dir = PDF_TO_PROJECT[row["pdf_id"]]
            top_k = int(args.top_k)
            for embedding_model in embedding_models:
                for mode in modes:
                    if not index_exists(project_dir, embedding_model, mode):
                        continue
                    t0 = time.perf_counter()
                    _, retrieved = pipeline.retrieve(
                        project_dir=project_dir,
                        question=row["question_bn"],
                        embedding_model=embedding_model,
                        mode=mode,
                        top_k=top_k,
                    )
                    latency_ms = (time.perf_counter() - t0) * 1000
                    retrieved_ids = [r.chunk.chunk_id for r in retrieved]
                    gold_id = row["gold_chunk_id"]
                    rank = next((i + 1 for i, cid in enumerate(retrieved_ids) if cid == gold_id), None)
                    writer.writerow({
                        "question_id": row["question_id"],
                        "pdf_id": row["pdf_id"],
                        "modality": row["modality"],
                        "question_type": row["question_type"],
                        "embedding_model": embedding_model,
                        "mode": mode,
                        "top_k": top_k,
                        "retrieval_latency_ms": round(latency_ms, 2),
                        "gold_chunk_id": gold_id,
                        "retrieved_chunk_ids": "|".join(retrieved_ids),
                        "rank_of_gold": rank or "",
                        "hit_at_1": int(rank == 1),
                        "hit_at_3": int(bool(rank) and rank <= 3),
                        "hit_at_5": int(bool(rank) and rank <= 5),
                        "reciprocal_rank": round(1.0 / rank, 4) if rank else 0.0,
                        "top1_score": round(retrieved[0].score, 4) if retrieved else "",
                    })
                    print(f"[retrieve_only] {row['question_id']} {embedding_model} {mode} "
                          f"rank={rank} latency={latency_ms:.1f}ms")
    print(f"\nWrote {out_path}")


def cmd_generate(args: argparse.Namespace) -> None:
    from src.config import load_project_config
    from src.generator import OllamaGenerator
    from src.pipeline import BanglaMultimodalRAGPipeline
    from src.text_processing import clean_bangla_spelling
    from src.utils import normalize_text

    pipeline = BanglaMultimodalRAGPipeline()
    rows = load_ground_truth(args.ground_truth)

    fieldnames = [
        "question_id", "pdf_id", "modality", "question_type", "embedding_model", "mode", "llm_model",
        "question_bn", "query_for_retrieval", "gold_answer_bn", "generated_answer_bn",
        "retrieval_latency_ms", "generation_latency_ms", "end_to_end_latency_ms",
        "gold_chunk_id", "retrieved_chunk_ids", "rank_of_gold", "hit_at_5", "top1_score",
        "exact_match", "token_f1",
    ]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            project_dir = PDF_TO_PROJECT[row["pdf_id"]]
            if not index_exists(project_dir, args.embedding_model, args.mode):
                print(f"[skip] no index for {row['pdf_id']} / {args.embedding_model} / {args.mode}")
                continue
            config = load_project_config(pipeline.project_paths(project_dir)["config"])

            t0 = time.perf_counter()
            query_for_retrieval, retrieved = pipeline.retrieve(
                project_dir=project_dir,
                question=row["question_bn"],
                embedding_model=args.embedding_model,
                mode=args.mode,
                top_k=int(args.top_k) if args.top_k else None,
            )
            retrieved = pipeline._enrich_image_chunks(retrieved, config)
            t1 = time.perf_counter()
            retrieval_latency_ms = (t1 - t0) * 1000

            generator = OllamaGenerator(
                base_url=config.ollama_base_url,
                model_name=args.llm,
                temperature=config.temperature,
            )
            answer = generator.generate(
                question=row["question_bn"], retrieved_chunks=retrieved, answer_language="bn"
            )
            answer = clean_bangla_spelling(answer)
            if not normalize_text(answer):
                answer = "প্রদত্ত নথিতে এর উত্তর পাওয়া যায়নি।"
            t2 = time.perf_counter()
            generation_latency_ms = (t2 - t1) * 1000
            end_to_end_latency_ms = (t2 - t0) * 1000

            retrieved_ids = [r.chunk.chunk_id for r in retrieved]
            gold_id = row["gold_chunk_id"]
            rank = next((i + 1 for i, cid in enumerate(retrieved_ids) if cid == gold_id), None)

            writer.writerow({
                "question_id": row["question_id"],
                "pdf_id": row["pdf_id"],
                "modality": row["modality"],
                "question_type": row["question_type"],
                "embedding_model": args.embedding_model,
                "mode": args.mode,
                "llm_model": args.llm,
                "question_bn": row["question_bn"],
                "query_for_retrieval": query_for_retrieval,
                "gold_answer_bn": row["gold_answer_bn"],
                "generated_answer_bn": answer,
                "retrieval_latency_ms": round(retrieval_latency_ms, 2),
                "generation_latency_ms": round(generation_latency_ms, 2),
                "end_to_end_latency_ms": round(end_to_end_latency_ms, 2),
                "gold_chunk_id": gold_id,
                "retrieved_chunk_ids": "|".join(retrieved_ids),
                "rank_of_gold": rank or "",
                "hit_at_5": int(bool(rank) and rank <= 5),
                "top1_score": round(retrieved[0].score, 4) if retrieved else "",
                "exact_match": exact_match(answer, row["gold_answer_bn"]),
                "token_f1": round(token_f1(answer, row["gold_answer_bn"]), 4),
            })
            f_out.flush()
            print(f"[generate] {row['question_id']} rank={rank} "
                  f"ret={retrieval_latency_ms:.0f}ms gen={generation_latency_ms:.0f}ms "
                  f"em={exact_match(answer, row['gold_answer_bn'])} f1={token_f1(answer, row['gold_answer_bn']):.2f}")
    print(f"\nWrote {out_path}")


def _mean(values: list[float]) -> float:
    return round(statistics.mean(values), 4) if values else 0.0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * pct
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return round(values[f], 2)
    return round(values[f] + (values[c] - values[f]) * (k - f), 2)


def cmd_aggregate(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- retrieval aggregate: Hit@k / MRR by (embedding_model, mode) ---
    retrieval_summary = defaultdict(lambda: defaultdict(list))
    with open(args.retrieval_csv, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            key = (r["embedding_model"], r["mode"])
            retrieval_summary[key]["hit1"].append(int(r["hit_at_1"]))
            retrieval_summary[key]["hit3"].append(int(r["hit_at_3"]))
            retrieval_summary[key]["hit5"].append(int(r["hit_at_5"]))
            retrieval_summary[key]["mrr"].append(float(r["reciprocal_rank"]))
            retrieval_summary[key]["latency"].append(float(r["retrieval_latency_ms"]))

    retrieval_table = []
    for (model, mode), vals in sorted(retrieval_summary.items()):
        retrieval_table.append({
            "embedding_model": model, "mode": mode, "n": len(vals["hit1"]),
            "hit_at_1": _mean(vals["hit1"]), "hit_at_3": _mean(vals["hit3"]),
            "hit_at_5": _mean(vals["hit5"]), "mrr": _mean(vals["mrr"]),
            "mean_retrieval_latency_ms": _mean(vals["latency"]),
        })
    (out_dir / "retrieval_summary.json").write_text(
        json.dumps(retrieval_table, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- generation aggregate: EM/F1/latency overall + by modality + by mode ---
    gen_rows = []
    for path in args.generation_csv:
        with open(path, encoding="utf-8-sig") as f:
            gen_rows.extend(list(csv.DictReader(f)))

    def summarize(rows: list[dict]) -> dict:
        return {
            "n": len(rows),
            "exact_match": _mean([int(r["exact_match"]) for r in rows]),
            "token_f1": _mean([float(r["token_f1"]) for r in rows]),
            "hit_at_5": _mean([int(r["hit_at_5"]) for r in rows]),
            "mean_retrieval_latency_ms": _mean([float(r["retrieval_latency_ms"]) for r in rows]),
            "mean_generation_latency_ms": _mean([float(r["generation_latency_ms"]) for r in rows]),
            "mean_e2e_latency_ms": _mean([float(r["end_to_end_latency_ms"]) for r in rows]),
            "p50_e2e_latency_ms": _percentile([float(r["end_to_end_latency_ms"]) for r in rows], 0.50),
            "p95_e2e_latency_ms": _percentile([float(r["end_to_end_latency_ms"]) for r in rows], 0.95),
        }

    overall_by_mode = defaultdict(list)
    for r in gen_rows:
        overall_by_mode[r["mode"]].append(r)
    by_mode_summary = {mode: summarize(rows) for mode, rows in overall_by_mode.items()}

    by_modality = defaultdict(list)
    for r in gen_rows:
        by_modality[r["modality"]].append(r)
    by_modality_summary = {mod: summarize(rows) for mod, rows in by_modality.items()}

    (out_dir / "generation_summary.json").write_text(
        json.dumps({"by_mode": by_mode_summary, "by_modality": by_modality_summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Retrieval summary:")
    for row in retrieval_table:
        print(" ", row)
    print("\nGeneration summary by mode:")
    for mode, s in by_mode_summary.items():
        print(" ", mode, s)
    print("\nGeneration summary by modality:")
    for mod, s in by_modality_summary.items():
        print(" ", mod, s)
    print(f"\nWrote {out_dir/'retrieval_summary.json'} and {out_dir/'generation_summary.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("retrieve_only")
    p1.add_argument("--ground-truth", required=True)
    p1.add_argument("--out", required=True)
    p1.add_argument("--embedding-models", nargs="+", required=True)
    p1.add_argument("--modes", nargs="+", default=["direct", "translated"])
    p1.add_argument("--top-k", default=5)
    p1.set_defaults(func=cmd_retrieve_only)

    p2 = sub.add_parser("generate")
    p2.add_argument("--ground-truth", required=True)
    p2.add_argument("--out", required=True)
    p2.add_argument("--embedding-model", required=True)
    p2.add_argument("--mode", required=True, choices=["direct", "translated"])
    p2.add_argument("--llm", required=True)
    p2.add_argument("--top-k", default=None)
    p2.set_defaults(func=cmd_generate)

    p3 = sub.add_parser("aggregate")
    p3.add_argument("--retrieval-csv", required=True)
    p3.add_argument("--generation-csv", nargs="+", required=True)
    p3.add_argument("--out-dir", required=True)
    p3.set_defaults(func=cmd_aggregate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
