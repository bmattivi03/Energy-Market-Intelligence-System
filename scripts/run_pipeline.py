#!/usr/bin/env python3
"""End-to-end pipeline orchestrator with skip-if-exists caching.

Runs: ingest -> impute -> build_splits -> module_a -> module_b -> module_c.
Each stage is skipped when its output artifact already exists (unless --force or
--force-<stage>). Use --skip-<stage> to skip explicitly, or --only <stage> to run
a single stage.

Examples
--------
    python scripts/run_pipeline.py                 # full run, skip completed stages
    python scripts/run_pipeline.py --skip-ingest --skip-impute
    python scripts/run_pipeline.py --only module_b --force
    python scripts/run_pipeline.py --from splits   # start at splits, run onward
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = {
    **os.environ,
    "PYTHONPATH": str(ROOT / "src"),
    "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    # Module C loads CatBoost + torch in one process; their OpenMP runtimes
    # collide on macOS. KMP_DUPLICATE_LIB_OK silences the abort; OMP_NUM_THREADS=1
    # prevents the SIGSEGV in torch during sustained RL training. Both needed.
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "OMP_NUM_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
}


def _stages(seed: int):
    """Ordered (name, output-artifact, [commands]) tuples."""
    py = sys.executable
    return [
        ("ingest", ROOT / "data/raw/entsoe_prices_2019_2025.parquet",
         [[py, "src/ingestion/run_ingestion.py"]]),
        ("impute", ROOT / "data/processed/emis_imputed.parquet",
         [[py, "-m", "preprocessing.impute", "--seed", str(seed), "--apply-constraints"]]),
        ("splits", ROOT / "data/splits/train.parquet",
         [[py, "-m", "preprocessing.build_splits"]]),
        ("module_a", ROOT / "data/module_a/load_quantiles.parquet",
         [[py, "-m", "module_a.train", "--seed", str(seed)],
          [py, "-m", "module_a.calibrate"],
          [py, "-m", "module_a.export_parquet"]]),
        ("module_b", ROOT / "checkpoints/module_b/catboost/meta.json",
         [[py, "-m", "module_b.train", "--seed", str(seed)]]),
        ("module_c", ROOT / "checkpoints/module_c/ppo/model.zip",
         [[py, "-m", "module_c.run"]]),
    ]


def _run(name: str, commands: list[list[str]]) -> None:
    for cmd in commands:
        print(f"\n{'='*70}\n  STAGE {name}: {' '.join(cmd[1:]) or cmd}\n{'='*70}", flush=True)
        t0 = time.monotonic()
        r = subprocess.run(cmd, cwd=ROOT, env=ENV)
        if r.returncode != 0:
            sys.exit(f"\n[run_pipeline] stage '{name}' failed (exit {r.returncode}).")
        print(f"[run_pipeline] {name} step done in {time.monotonic() - t0:.0f}s")


def main() -> None:
    names = [s[0] for s in _stages(42)]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true", help="re-run every (non-skipped) stage even if its output exists")
    p.add_argument("--only", choices=names, help="run only this stage")
    p.add_argument("--from", dest="from_stage", choices=names, help="start at this stage, run onward")
    for n in names:
        p.add_argument(f"--skip-{n}", action="store_true", help=f"skip the {n} stage")
        p.add_argument(f"--force-{n}", action="store_true", help=f"re-run {n} even if output exists")
    args = p.parse_args()

    stages = _stages(args.seed)
    start = names.index(args.from_stage) if args.from_stage else 0

    for i, (name, artifact, commands) in enumerate(stages):
        if args.only and name != args.only:
            continue
        if not args.only and i < start:
            continue
        if getattr(args, f"skip_{name}"):
            print(f"[run_pipeline] skip {name} (--skip-{name})")
            continue
        force = args.force or getattr(args, f"force_{name}")
        if artifact.exists() and not force:
            print(f"[run_pipeline] skip {name}: artifact exists ({artifact.relative_to(ROOT)}). Use --force-{name} to re-run.")
            continue
        _run(name, commands)

    print("\n[run_pipeline] done.")


if __name__ == "__main__":
    main()
