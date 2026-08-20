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
import random
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
    "Agriculture": "workspace/projects/rag_test4-agriculture",
    "BDHistory": "workspace/projects/rag_test5-bd-history",
    "PolashirJuddho": "workspace/projects/rag_test6-polashir-juddho",
    "UGC1": "workspace/projects/rag_test7-ugc1-law",
    "UGC2": "workspace/projects/rag_test8-ugc2-scholarship",
    "UGC3": "workspace/projects/rag_test9-ugc3-rokeya",
    "DUETBooklet": "workspace/projects/rag_test10-duet-booklet-en",
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


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l(pred: str, gold: str) -> float:
    """ROUGE-L F-measure (LCS-based), a standard generation-quality metric
    that tolerates word reordering/insertion better than strict Exact Match."""
    pred_toks, gold_toks = tokenize(pred), tokenize(gold)
    if not pred_toks or not gold_toks:
        return 0.0
    lcs = _lcs_len(pred_toks, gold_toks)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_toks)
    recall = lcs / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def keyword_recall(pred: str, expected_keywords: str) -> float:
    """Fraction of hand-annotated expected keywords found (as substrings) in the
    generated answer. More robust to Bangla inflection/paraphrase than Exact
    Match, since it does not require the full answer string to match verbatim."""
    keywords = [k.strip() for k in (expected_keywords or "").split(";") if k.strip()]
    if not keywords:
        return 0.0
    norm_pred = normalize_for_metric(pred)
    if not norm_pred:
        return 0.0
    hits = sum(1 for kw in keywords if normalize_for_metric(kw) in norm_pred)
    return hits / len(keywords)


_NOT_FOUND_MARKERS = [
    "উত্তর পাওয়া যায়নি", "পাওয়া যায়নি", "not found", "not available",
    "answer was not found",
]


def looks_like_abstention(answer: str) -> bool:
    a = normalize_for_metric(answer)
    return any(normalize_for_metric(m) in a for m in _NOT_FOUND_MARKERS)


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
                        question=row["question_text"],
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
        "question_id", "pdf_id", "pdf_type", "modality", "question_type", "difficulty",
        "embedding_model", "mode", "llm_model",
        "answer_language", "question_text", "query_for_retrieval", "gold_answer_text", "generated_answer_text",
        "retrieval_latency_ms", "generation_latency_ms", "end_to_end_latency_ms",
        "gold_chunk_id", "retrieved_chunk_ids", "rank_of_gold", "hit_at_5", "top1_score",
        "exact_match", "token_f1", "rouge_l", "keyword_recall", "semantic_similarity", "is_abstention",
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
            answer_language = row.get("answer_language") or "bn"

            t0 = time.perf_counter()
            query_for_retrieval, retrieved = pipeline.retrieve(
                project_dir=project_dir,
                question=row["question_text"],
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
                question=row["question_text"], retrieved_chunks=retrieved, answer_language=answer_language
            )
            if answer_language == "bn":
                answer = clean_bangla_spelling(answer)
                not_found_fallback = "প্রদত্ত নথিতে এর উত্তর পাওয়া যায়নি।"
            else:
                not_found_fallback = "The answer was not found in the provided document."
            if not normalize_text(answer):
                answer = not_found_fallback
            t2 = time.perf_counter()
            generation_latency_ms = (t2 - t1) * 1000
            end_to_end_latency_ms = (t2 - t0) * 1000

            retrieved_ids = [r.chunk.chunk_id for r in retrieved]
            gold_id = row["gold_chunk_id"]
            rank = next((i + 1 for i, cid in enumerate(retrieved_ids) if cid == gold_id), None)

            gold_answer = row["gold_answer_text"]
            sem_sim = ""
            try:
                embedder = pipeline._get_embedder(args.embedding_model, config.embedding_device, config.ollama_base_url)
                vecs = embedder.encode_queries([answer, gold_answer])
                sem_sim = round(float((vecs[0] * vecs[1]).sum()), 4)
            except Exception as exc:
                print(f"[warn] semantic_similarity failed for {row['question_id']}: {exc}")

            writer.writerow({
                "question_id": row["question_id"],
                "pdf_id": row["pdf_id"],
                "pdf_type": row.get("pdf_type", ""),
                "modality": row["modality"],
                "question_type": row["question_type"],
                "difficulty": row.get("difficulty", ""),
                "embedding_model": args.embedding_model,
                "mode": args.mode,
                "llm_model": args.llm,
                "answer_language": answer_language,
                "question_text": row["question_text"],
                "query_for_retrieval": query_for_retrieval,
                "gold_answer_text": gold_answer,
                "generated_answer_text": answer,
                "retrieval_latency_ms": round(retrieval_latency_ms, 2),
                "generation_latency_ms": round(generation_latency_ms, 2),
                "end_to_end_latency_ms": round(end_to_end_latency_ms, 2),
                "gold_chunk_id": gold_id,
                "retrieved_chunk_ids": "|".join(retrieved_ids),
                "rank_of_gold": rank or "",
                "hit_at_5": int(bool(rank) and rank <= 5),
                "top1_score": round(retrieved[0].score, 4) if retrieved else "",
                "exact_match": exact_match(answer, gold_answer),
                "token_f1": round(token_f1(answer, gold_answer), 4),
                "rouge_l": round(rouge_l(answer, gold_answer), 4),
                "keyword_recall": round(keyword_recall(answer, row.get("expected_keywords_text", "")), 4),
                "semantic_similarity": sem_sim,
                "is_abstention": int(looks_like_abstention(answer)),
            })
            f_out.flush()
            print(f"[generate] {row['question_id']} rank={rank} "
                  f"ret={retrieval_latency_ms:.0f}ms gen={generation_latency_ms:.0f}ms "
                  f"em={exact_match(answer, gold_answer)} f1={token_f1(answer, gold_answer):.2f} "
                  f"rouge_l={rouge_l(answer, gold_answer):.2f} sem_sim={sem_sim}")
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

    def _floats(rows: list[dict], key: str) -> list[float]:
        out = []
        for r in rows:
            v = r.get(key, "")
            if v not in ("", None):
                try:
                    out.append(float(v))
                except ValueError:
                    pass
        return out

    def summarize(rows: list[dict]) -> dict:
        return {
            "n": len(rows),
            "exact_match": _mean([int(r["exact_match"]) for r in rows]),
            "token_f1": _mean([float(r["token_f1"]) for r in rows]),
            "rouge_l": _mean(_floats(rows, "rouge_l")),
            "keyword_recall": _mean(_floats(rows, "keyword_recall")),
            "semantic_similarity": _mean(_floats(rows, "semantic_similarity")),
            "abstention_rate": _mean([int(r["is_abstention"]) for r in rows if r.get("is_abstention", "") != ""]),
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

    by_pdf = defaultdict(list)
    for r in gen_rows:
        by_pdf[r["pdf_id"]].append(r)
    by_pdf_summary = {pdf_id: summarize(rows) for pdf_id, rows in by_pdf.items()}

    by_language = defaultdict(list)
    for r in gen_rows:
        by_language[r.get("answer_language") or "bn"].append(r)
    by_language_summary = {lang: summarize(rows) for lang, rows in by_language.items()}

    by_question_type = defaultdict(list)
    for r in gen_rows:
        by_question_type[r.get("question_type") or "unknown"].append(r)
    by_question_type_summary = {qt: summarize(rows) for qt, rows in by_question_type.items()}

    by_difficulty = defaultdict(list)
    for r in gen_rows:
        by_difficulty[r.get("difficulty") or "unknown"].append(r)
    by_difficulty_summary = {d: summarize(rows) for d, rows in by_difficulty.items()}

    by_pdf_type = defaultdict(list)
    for r in gen_rows:
        by_pdf_type[r.get("pdf_type") or "unknown"].append(r)
    by_pdf_type_summary = {pt: summarize(rows) for pt, rows in by_pdf_type.items()}

    (out_dir / "generation_summary.json").write_text(
        json.dumps({
            "by_mode": by_mode_summary,
            "by_modality": by_modality_summary,
            "by_pdf": by_pdf_summary,
            "by_language": by_language_summary,
            "by_question_type": by_question_type_summary,
            "by_difficulty": by_difficulty_summary,
            "by_pdf_type": by_pdf_type_summary,
        }, ensure_ascii=False, indent=2),
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
    print("\nGeneration summary by document:")
    for pdf_id, s in by_pdf_summary.items():
        print(" ", pdf_id, s)
    print("\nGeneration summary by language:")
    for lang, s in by_language_summary.items():
        print(" ", lang, s)
    print("\nGeneration summary by question type:")
    for qt, s in by_question_type_summary.items():
        print(" ", qt, s)
    print("\nGeneration summary by difficulty:")
    for d, s in by_difficulty_summary.items():
        print(" ", d, s)
    print("\nGeneration summary by pdf_type:")
    for pt, s in by_pdf_type_summary.items():
        print(" ", pt, s)
    print(f"\nWrote {out_dir/'retrieval_summary.json'} and {out_dir/'generation_summary.json'}")


def _confusion_2x2(rows: list[dict], predicate_key) -> dict:
    """2x2 matrix: rows = retrieval Hit@5 (yes/no), cols = answer correct per predicate_key (yes/no)."""
    tt = fp = fn = tf = 0  # hit&correct, hit&wrong, miss&correct, miss&wrong
    for r in rows:
        hit = bool(int(r["hit_at_5"]))
        correct = predicate_key(r)
        if hit and correct:
            tt += 1
        elif hit and not correct:
            fp += 1
        elif not hit and correct:
            fn += 1
        else:
            tf += 1
    n = len(rows)
    return {
        "n": n,
        "hit_and_correct": tt,
        "hit_and_wrong": fp,
        "miss_and_correct": fn,
        "miss_and_wrong": tf,
        "retrieval_hit_rate": _mean([int(int(r["hit_at_5"])) for r in rows]),
        "answer_correct_rate": _mean([int(predicate_key(r)) for r in rows]),
        # Of the wrong answers, what fraction trace back to a retrieval miss vs a
        # generation failure despite correct retrieval - the key RAG error-analysis split.
        "wrong_due_to_retrieval_miss_pct": round(100 * tf / max(fp + tf, 1), 2),
        "wrong_due_to_generation_pct": round(100 * fp / max(fp + tf, 1), 2),
    }


def _abstention_analysis(rows: list[dict]) -> dict:
    """Cross retrieval Hit@5 against whether the model abstained ("not found").

    A false abstention (evidence WAS retrieved but the model still refused) is a
    distinct failure mode from a correct/appropriate abstention, and matters for
    a RAG paper's error taxonomy separately from the raw hit/wrong split.
    """
    abstained_with_hit = sum(1 for r in rows if int(r["hit_at_5"]) and int(r["is_abstention"]))
    abstained_without_hit = sum(1 for r in rows if not int(r["hit_at_5"]) and int(r["is_abstention"]))
    total_abstentions = abstained_with_hit + abstained_without_hit
    total_hits = sum(1 for r in rows if int(r["hit_at_5"]))
    return {
        "n": len(rows),
        "total_abstentions": total_abstentions,
        "abstention_rate": round(total_abstentions / max(len(rows), 1), 4),
        "false_abstentions_despite_retrieval_hit": abstained_with_hit,
        "false_abstention_rate_given_hit": round(abstained_with_hit / max(total_hits, 1), 4),
        "abstentions_with_retrieval_miss": abstained_without_hit,
    }


def cmd_confusion(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_rows = []
    for path in args.generation_csv:
        with open(path, encoding="utf-8-sig") as f:
            gen_rows.extend(list(csv.DictReader(f)))
    gen_rows = [r for r in gen_rows if r.get("is_abstention", "") != ""]

    is_exact = lambda r: bool(int(r["exact_match"]))
    is_f1_correct = lambda r: float(r["token_f1"]) >= args.f1_threshold

    def grouped(key_fn) -> dict:
        groups = defaultdict(list)
        for r in gen_rows:
            groups[key_fn(r)].append(r)
        return {
            k: {"exact_match": _confusion_2x2(v, is_exact), "f1_threshold": _confusion_2x2(v, is_f1_correct)}
            for k, v in sorted(groups.items())
        }

    result = {
        "f1_threshold_used": args.f1_threshold,
        "overall": {"exact_match": _confusion_2x2(gen_rows, is_exact), "f1_threshold": _confusion_2x2(gen_rows, is_f1_correct)},
        "by_embedding_model": grouped(lambda r: r["embedding_model"]),
        "by_mode": grouped(lambda r: r["mode"]),
        "by_pdf": grouped(lambda r: r["pdf_id"]),
        "by_language": grouped(lambda r: r.get("answer_language") or "bn"),
        "by_modality": grouped(lambda r: r["modality"]),
        "abstention_analysis": {
            "overall": _abstention_analysis(gen_rows),
            "by_pdf": {k: _abstention_analysis(v) for k, v in sorted(defaultdict(list, {
                pid: [r for r in gen_rows if r["pdf_id"] == pid] for pid in {r["pdf_id"] for r in gen_rows}
            }).items())},
        },
    }
    (out_dir / "confusion_matrix.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== False abstention analysis (overall) ===")
    print(" ", result["abstention_analysis"]["overall"])

    def _print_matrix(label: str, m: dict) -> None:
        print(f"  {label}: n={m['n']}  hit_rate={m['retrieval_hit_rate']}  correct_rate={m['answer_correct_rate']}")
        print(f"           correct  wrong")
        print(f"    hit    {m['hit_and_correct']:>7}  {m['hit_and_wrong']:>5}")
        print(f"    miss   {m['miss_and_correct']:>7}  {m['miss_and_wrong']:>5}")
        print(f"    of wrong answers: {m['wrong_due_to_generation_pct']}% despite retrieval hit (generation failure), "
              f"{m['wrong_due_to_retrieval_miss_pct']}% due to retrieval miss")

    print("=== Overall (Exact Match) ===")
    _print_matrix("overall", result["overall"]["exact_match"])
    print("\n=== Overall (F1 >= %.2f) ===" % args.f1_threshold)
    _print_matrix("overall", result["overall"]["f1_threshold"])
    print("\n=== By embedding model (Exact Match) ===")
    for k, v in result["by_embedding_model"].items():
        _print_matrix(k, v["exact_match"])
    print("\n=== By document (Exact Match) ===")
    for k, v in result["by_pdf"].items():
        _print_matrix(k, v["exact_match"])
    print("\n=== By language (Exact Match) ===")
    for k, v in result["by_language"].items():
        _print_matrix(k, v["exact_match"])
    print(f"\nWrote {out_dir/'confusion_matrix.json'}")


def _sign_test_p_value(n_pos: int, n_neg: int) -> float:
    """Exact two-sided binomial sign test p-value, no scipy dependency."""
    import math
    n = n_pos + n_neg
    if n == 0:
        return 1.0
    k = min(n_pos, n_neg)
    # P(X <= k) under Binomial(n, 0.5), doubled for two-sided, capped at 1.0
    cumulative = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return round(min(1.0, 2 * cumulative), 6)


def _bootstrap_ci(diffs: list[float], n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    if not diffs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot) - 1]
    return (round(lo, 4), round(hi, 4))


def cmd_significance(args: argparse.Namespace) -> None:
    """Paired statistical comparison between two generation-eval configurations
    (e.g. two embedding models, or direct vs translated mode) on the SAME
    question set: mean metric difference, bootstrap 95% CI, and an exact
    two-sided sign test p-value. Answers "is config B actually better than
    config A, or within noise" - standard for a Q1 result-analysis section.
    """
    with open(args.csv_a, encoding="utf-8-sig") as f:
        rows_a = {r["question_id"]: r for r in csv.DictReader(f)}
    with open(args.csv_b, encoding="utf-8-sig") as f:
        rows_b = {r["question_id"]: r for r in csv.DictReader(f)}
    shared_ids = sorted(set(rows_a) & set(rows_b))
    if not shared_ids:
        print("No shared question_id values between the two CSVs - nothing to compare.")
        return

    metrics = ["exact_match", "token_f1", "rouge_l", "keyword_recall", "semantic_similarity"]
    result = {"label_a": args.label_a, "label_b": args.label_b, "n_paired": len(shared_ids), "metrics": {}}
    for metric in metrics:
        pairs = []
        for qid in shared_ids:
            va, vb = rows_a[qid].get(metric, ""), rows_b[qid].get(metric, "")
            if va in ("", None) or vb in ("", None):
                continue
            try:
                pairs.append((float(va), float(vb)))
            except ValueError:
                continue
        if not pairs:
            continue
        diffs = [b - a for a, b in pairs]
        n_pos = sum(1 for d in diffs if d > 0)
        n_neg = sum(1 for d in diffs if d < 0)
        ci_lo, ci_hi = _bootstrap_ci(diffs)
        result["metrics"][metric] = {
            "n": len(pairs),
            "mean_a": round(sum(a for a, _ in pairs) / len(pairs), 4),
            "mean_b": round(sum(b for _, b in pairs) / len(pairs), 4),
            "mean_diff_b_minus_a": round(sum(diffs) / len(diffs), 4),
            "bootstrap_95ci": [ci_lo, ci_hi],
            "n_b_better": n_pos,
            "n_a_better": n_neg,
            "n_tied": len(diffs) - n_pos - n_neg,
            "sign_test_p_value": _sign_test_p_value(n_pos, n_neg),
            "significant_at_0.05": _sign_test_p_value(n_pos, n_neg) < 0.05,
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"=== {args.label_a} vs {args.label_b} (n={len(shared_ids)} shared questions) ===")
    for metric, m in result["metrics"].items():
        sig = "SIGNIFICANT" if m["significant_at_0.05"] else "not significant"
        print(f"  {metric}: A={m['mean_a']} B={m['mean_b']} diff={m['mean_diff_b_minus_a']} "
              f"95%CI={m['bootstrap_95ci']} p={m['sign_test_p_value']} ({sig})")
    print(f"\nWrote {out_path}")


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

    p4 = sub.add_parser("confusion")
    p4.add_argument("--generation-csv", nargs="+", required=True)
    p4.add_argument("--out-dir", required=True)
    p4.add_argument("--f1-threshold", type=float, default=0.6)
    p4.set_defaults(func=cmd_confusion)

    p5 = sub.add_parser("significance")
    p5.add_argument("--csv-a", required=True)
    p5.add_argument("--csv-b", required=True)
    p5.add_argument("--label-a", default="A")
    p5.add_argument("--label-b", default="B")
    p5.add_argument("--out", required=True)
    p5.set_defaults(func=cmd_significance)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
