#!/usr/bin/env python
"""Orchestrates everything still queued after the qwen2.5:3b vs llama3-chatqa
generation run: waits for that run to finish, scores it, adds
mistral:7b-instruct-q4_K_M as a third generator on the same BAAI/bge-m3
config (both modes), runs the original-proposal-vs-current captioning
ablation, and writes one consolidated summary at the end.

Runs as a single background process so nothing needs to be re-launched by
hand between stages. Safe to re-run: each stage is skipped if its output
file already has the expected row count.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

PY = sys.executable
GT = str(ROOT / "data" / "eval_ground_truth_full.csv")
RESULTS = ROOT / "results"
AGG = RESULTS / "aggregate_full"
AGG.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "BAAI/bge-m3"
LLMS = {
    "qwen3b": ("qwen2.5:3b-instruct", RESULTS / "generation_eval_baai-bge-m3_direct.csv", RESULTS / "generation_eval_baai-bge-m3_translated.csv"),
    "chatqa": ("llama3-chatqa:latest", RESULTS / "generation_eval_llama3-chatqa_baai-bge-m3_direct.csv", RESULTS / "generation_eval_llama3-chatqa_baai-bge-m3_translated.csv"),
    "mistral": ("mistral:7b-instruct-q4_K_M", RESULTS / "generation_eval_mistral7b_baai-bge-m3_direct.csv", RESULTS / "generation_eval_mistral7b_baai-bge-m3_translated.csv"),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8-sig") as f:
        return sum(1 for _ in f) - 1  # minus header


def wait_for(path: Path, expected: int, timeout_s: int, poll_s: int = 60) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        n = csv_rows(path)
        if n >= expected:
            return True
        time.sleep(poll_s)
    return csv_rows(path) >= expected


def run_generate(llm: str, mode: str, out_path: Path) -> None:
    if csv_rows(out_path) >= 150:
        log(f"[skip] {out_path.name} already complete")
        return
    log(f"Generating: llm={llm} mode={mode} -> {out_path.name}")
    t0 = time.time()
    result = subprocess.run(
        [PY, str(ROOT / "scripts" / "evaluate_metrics.py"), "generate",
         "--ground-truth", GT, "--out", str(out_path),
         "--embedding-model", EMBEDDING_MODEL, "--mode", mode, "--llm", llm],
        cwd=str(ROOT),
    )
    log(f"  -> {'OK' if result.returncode == 0 else 'FAILED'} in {time.time() - t0:.1f}s")


def run_significance(csv_a: Path, csv_b: Path, label_a: str, label_b: str, out: Path) -> None:
    log(f"Significance: {label_a} vs {label_b} -> {out.name}")
    subprocess.run(
        [PY, str(ROOT / "scripts" / "evaluate_metrics.py"), "significance",
         "--csv-a", str(csv_a), "--csv-b", str(csv_b),
         "--label-a", label_a, "--label-b", label_b, "--out", str(out)],
        cwd=str(ROOT),
    )


def mean_latency_ms(path: Path) -> float | None:
    if not path.exists():
        return None
    vals = []
    with path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            v = row.get("generation_latency_ms", "")
            if v:
                try:
                    vals.append(float(v))
                except ValueError:
                    pass
    return round(sum(vals) / len(vals), 1) if vals else None


def stage_1_wait_for_chatqa() -> None:
    d = LLMS["chatqa"][1]
    t = LLMS["chatqa"][2]
    if csv_rows(d) < 150:
        log("Waiting for llama3-chatqa direct run to finish...")
        wait_for(d, 150, timeout_s=4 * 3600)
    if csv_rows(t) < 150:
        log("Waiting for llama3-chatqa translated run to finish...")
        wait_for(t, 150, timeout_s=4 * 3600)
    log("llama3-chatqa generation complete.")


def stage_2_significance_qwen_vs_chatqa() -> None:
    run_significance(LLMS["qwen3b"][1], LLMS["chatqa"][1], "qwen2.5-3b_direct", "llama3-chatqa_direct", AGG / "significance_qwen3b_vs_chatqa_direct.json")
    run_significance(LLMS["qwen3b"][2], LLMS["chatqa"][2], "qwen2.5-3b_translated", "llama3-chatqa_translated", AGG / "significance_qwen3b_vs_chatqa_translated.json")


def stage_3_mistral_generation() -> None:
    llm, direct_out, translated_out = LLMS["mistral"]
    run_generate(llm, "direct", direct_out)
    run_generate(llm, "translated", translated_out)


def stage_4_significance_mistral() -> None:
    mdir, mtrans = LLMS["mistral"][1], LLMS["mistral"][2]
    run_significance(LLMS["qwen3b"][1], mdir, "qwen2.5-3b_direct", "mistral7b_direct", AGG / "significance_qwen3b_vs_mistral_direct.json")
    run_significance(LLMS["qwen3b"][2], mtrans, "qwen2.5-3b_translated", "mistral7b_translated", AGG / "significance_qwen3b_vs_mistral_translated.json")
    run_significance(LLMS["chatqa"][1], mdir, "llama3-chatqa_direct", "mistral7b_direct", AGG / "significance_chatqa_vs_mistral_direct.json")
    run_significance(LLMS["chatqa"][2], mtrans, "llama3-chatqa_translated", "mistral7b_translated", AGG / "significance_chatqa_vs_mistral_translated.json")


def stage_5_caption_ablation() -> None:
    ablation_out = RESULTS / "generation_eval_baseline_nocap_image_direct.csv"
    if csv_rows(ablation_out) >= 16:
        log("[skip] caption ablation already complete")
    else:
        log("Running caption ablation: build baseline (no-caption) projects + eval 16 image questions")
        import eval_caption_ablation as abl
        abl.build_baseline_projects()
        abl.run_ablation_eval()
    run_significance(
        LLMS["qwen3b"][1], ablation_out,
        "current_arch_image_qs", "baseline_nocap_image_qs",
        AGG / "significance_caption_ablation.json",
    )


def stage_6_final_summary() -> None:
    log("Building final consolidated summary...")
    summary = {
        "generators_compared": ["qwen2.5:3b-instruct", "llama3-chatqa:latest", "mistral:7b-instruct-q4_K_M"],
        "fixed_config": {"embedding_model": EMBEDDING_MODEL, "modes": ["direct", "translated"]},
        "mean_generation_latency_ms": {
            name: {"direct": mean_latency_ms(d), "translated": mean_latency_ms(t)}
            for name, (_, d, t) in LLMS.items()
        },
        "significance_files": sorted(str(p.relative_to(ROOT)) for p in AGG.glob("significance_*.json")),
    }
    out = AGG / "final_comparison_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out}")
    log("ALL STAGES COMPLETE.")


def main() -> None:
    stage_1_wait_for_chatqa()
    stage_2_significance_qwen_vs_chatqa()
    stage_3_mistral_generation()
    stage_4_significance_mistral()
    stage_5_caption_ablation()
    stage_6_final_summary()


if __name__ == "__main__":
    main()
