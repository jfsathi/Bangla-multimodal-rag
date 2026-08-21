#!/usr/bin/env python
"""Sensitivity analysis: recompute the retrieval, generation, and confusion-matrix
tables with DUETBooklet excluded and IICTNotice (the supplementary 1-page English
document) included in its place, using the same 5-embedding-model x 2-mode
structure as the canonical results. This is reported ALONGSIDE, not instead of,
the canonical ten-document evaluation - it answers "what would the headline
numbers look like on a more favorable English document" as an explicit,
labeled sensitivity check, not a silent substitution.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_metrics as em  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

EMBEDDING_MODELS = [
    "nomic-embed-text-v2-moe:latest",
    "BAAI/bge-m3",
    "intfloat/multilingual-e5-base",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]
MODES = ["direct", "translated"]


def slugify(name: str) -> str:
    return name.replace(":", "-").replace("/", "-").replace(".", "-").lower()


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


# ---------- 1. Retrieval: combine (9 Bangla docs from canonical) + (IICTNotice) ----------
canonical_retrieval = load_csv(RESULTS / "retrieval_eval_full.csv")
no_duet_retrieval = [r for r in canonical_retrieval if r["pdf_id"] != "DUETBooklet"]
iict_retrieval = load_csv(RESULTS / "retrieval_eval_iict_notice.csv")
combined_retrieval = no_duet_retrieval + iict_retrieval
print(f"Retrieval rows: canonical={len(canonical_retrieval)} no_duet_bangla={len(no_duet_retrieval)} "
      f"iict={len(iict_retrieval)} combined={len(combined_retrieval)}")

retrieval_summary = {}
for model in EMBEDDING_MODELS:
    for mode in MODES:
        rows = [r for r in combined_retrieval if r["embedding_model"] == model and r["mode"] == mode]
        n = len(rows)
        if n == 0:
            continue
        hit1 = sum(1 for r in rows if r.get("hit_at_1") == "1") / n
        hit3 = sum(1 for r in rows if r.get("hit_at_3") == "1") / n
        hit5 = sum(1 for r in rows if r.get("hit_at_5") == "1") / n
        mrr = sum(float(r.get("reciprocal_rank", 0) or 0) for r in rows) / n
        retrieval_summary[f"{model}|{mode}"] = {
            "n": n, "hit_at_1": round(hit1, 4), "hit_at_3": round(hit3, 4),
            "hit_at_5": round(hit5, 4), "mrr": round(mrr, 4),
        }

# ---------- 2. Generation (primary generator qwen2.5:3b): combine canonical (minus DUET) + IICTNotice ----------
gen_rows_all = []
for model in EMBEDDING_MODELS:
    slug = slugify(model)
    canon_path = RESULTS / f"generation_eval_{slug}_direct.csv"
    canon_path_t = RESULTS / f"generation_eval_{slug}_translated.csv"
    for p, mode in [(canon_path, "direct"), (canon_path_t, "translated")]:
        if p.exists():
            rows = load_csv(p)
            gen_rows_all.extend([r for r in rows if r["pdf_id"] != "DUETBooklet"])

iict_gen_files = {
    "nomic-embed-text-v2-moe:latest": "nomic-embed-text-v2-moe-latest",
    "BAAI/bge-m3": "qwen3b",  # special-cased filename from earlier run
    "intfloat/multilingual-e5-base": "intfloat-multilingual-e5-base",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": "sentence-transformers-paraphrase-multilingual-mpnet-base-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "sentence-transformers-paraphrase-multilingual-minilm-l12-v2",
}
for model, slug in iict_gen_files.items():
    for mode in MODES:
        p = RESULTS / f"generation_eval_iict_notice_{slug}_{mode}.csv"
        if p.exists():
            rows = load_csv(p)
            for r in rows:
                r["embedding_model"] = model
                r["mode"] = mode
            gen_rows_all.extend(rows)
        else:
            print(f"[warn] missing {p}")

print(f"Generation rows combined (primary generator, no-DUET + IICTNotice): {len(gen_rows_all)}")

def summarize_generation(rows: list[dict]) -> dict:
    n = len(rows)
    def f(key):
        vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
        return round(sum(vals) / len(vals), 4) if vals else None
    def ip(key):
        vals = [int(r[key]) for r in rows if r.get(key) not in (None, "")]
        return round(sum(vals) / len(vals), 4) if vals else None
    return {
        "n": n, "exact_match": f("exact_match"), "token_f1": f("token_f1"),
        "rouge_l": f("rouge_l"), "keyword_recall": f("keyword_recall"),
        "semantic_similarity": f("semantic_similarity"),
        "abstention_rate": ip("is_abstention"), "retrieval_hit_at_5": ip("hit_at_5"),
    }

gen_summary_by_mode = {mode: summarize_generation([r for r in gen_rows_all if r["mode"] == mode]) for mode in MODES}
gen_summary_overall = summarize_generation(gen_rows_all)

# ---------- 3. Confusion matrix (primary generator pooled, no-DUET + IICTNotice) ----------
def confusion(rows: list[dict], f1_threshold: float = 0.6) -> dict:
    n = len(rows)
    out = {}
    for label, predicate in [
        ("exact_match", lambda r: int(r["exact_match"]) == 1),
        ("f1_threshold", lambda r: float(r["token_f1"]) >= f1_threshold),
    ]:
        hit_correct = hit_wrong = miss_correct = miss_wrong = 0
        for r in rows:
            hit = r.get("hit_at_5") == "1"
            correct = predicate(r)
            if hit and correct:
                hit_correct += 1
            elif hit and not correct:
                hit_wrong += 1
            elif not hit and correct:
                miss_correct += 1
            else:
                miss_wrong += 1
        wrong = hit_wrong + miss_wrong
        out[label] = {
            "n": n, "hit_and_correct": hit_correct, "hit_and_wrong": hit_wrong,
            "miss_and_correct": miss_correct, "miss_and_wrong": miss_wrong,
            "retrieval_hit_rate": round((hit_correct + hit_wrong) / n, 4) if n else None,
            "answer_correct_rate": round((hit_correct + miss_correct) / n, 4) if n else None,
            "wrong_due_to_retrieval_miss_pct": round(100 * miss_wrong / wrong, 2) if wrong else None,
            "wrong_due_to_generation_pct": round(100 * hit_wrong / wrong, 2) if wrong else None,
        }
    return out

confusion_no_duet = confusion(gen_rows_all)

# ---------- 4. Recommended pipeline (bge-m3 + llama3-chatqa): combine canonical (minus DUET) + IICTNotice ----------
chatqa_rows_all = []
for mode, canon_file, iict_file in [
    ("direct", RESULTS / "generation_eval_llama3-chatqa_baai-bge-m3_direct.csv", RESULTS / "generation_eval_iict_notice_chatqa_direct.csv"),
    ("translated", RESULTS / "generation_eval_llama3-chatqa_baai-bge-m3_translated.csv", RESULTS / "generation_eval_iict_notice_chatqa_translated.csv"),
]:
    canon_rows = [r for r in load_csv(canon_file) if r["pdf_id"] != "DUETBooklet"]
    iict_rows = load_csv(iict_file)
    for r in iict_rows:
        r["mode"] = mode
    chatqa_rows_all.extend(canon_rows + iict_rows)

chatqa_summary_by_mode = {mode: summarize_generation([r for r in chatqa_rows_all if r["mode"] == mode]) for mode in MODES}
confusion_chatqa_no_duet = confusion(chatqa_rows_all)

# ---------- Write + print ----------
out = {
    "note": "Sensitivity analysis with DUETBooklet excluded and IICTNotice (1-page supplementary English doc) included instead. Reported alongside, not in place of, the canonical ten-document results.",
    "retrieval_summary_by_model_mode": retrieval_summary,
    "generation_summary_primary_qwen25_3b": {"overall": gen_summary_overall, "by_mode": gen_summary_by_mode},
    "confusion_matrix_primary_qwen25_3b": confusion_no_duet,
    "generation_summary_recommended_bgem3_chatqa": chatqa_summary_by_mode,
    "confusion_matrix_recommended_bgem3_chatqa": confusion_chatqa_no_duet,
}
out_path = RESULTS / "aggregate_full" / "no_duet_sensitivity.json"
out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nWrote {out_path}")
print(json.dumps(out, ensure_ascii=False, indent=2))
