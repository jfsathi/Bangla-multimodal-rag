#!/usr/bin/env python
"""Run the full generation-eval matrix: 5 embedding models x 2 modes, each over
all 150 ground-truth questions, using qwen2.5:3b-instruct. Runs configs
sequentially (not in parallel - avoids GPU/CPU contention) as one process so
it can be launched as a single background job.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EMBEDDING_MODELS = [
    "nomic-embed-text-v2-moe:latest",
    "BAAI/bge-m3",
    "intfloat/multilingual-e5-base",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]
MODES = ["direct", "translated"]
LLM = "qwen2.5:3b-instruct"


def slugify(name: str) -> str:
    return name.replace(":", "-").replace("/", "-").replace(".", "-").lower()


def main() -> None:
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    total = len(EMBEDDING_MODELS) * len(MODES)
    done = 0
    for model in EMBEDDING_MODELS:
        for mode in MODES:
            done += 1
            out_path = results_dir / f"generation_eval_{slugify(model)}_{mode}.csv"
            print(f"\n=== [{done}/{total}] {model} / {mode} -> {out_path.name} ===", flush=True)
            t0 = time.time()
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "evaluate_metrics.py"), "generate",
                 "--ground-truth", str(ROOT / "data" / "eval_ground_truth_full.csv"),
                 "--out", str(out_path),
                 "--embedding-model", model,
                 "--mode", mode,
                 "--llm", LLM],
                cwd=str(ROOT),
            )
            elapsed = time.time() - t0
            status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
            print(f"=== [{done}/{total}] {model} / {mode}: {status} in {elapsed:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
