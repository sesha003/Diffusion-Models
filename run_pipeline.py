#!/usr/bin/env python3
"""
MASTER RUNNER — Diffusion Lung Segmentation Pipeline
Runs in sequence:
  1. train_direct_segmentation.py   (3 models × 3 folds)
  2. generate_figures.py            (20 figures)
  3. generate_tables_and_volumetric.py (tables + 3D)

Usage:  python run_pipeline.py
"""

import subprocess, sys, time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
PY       = sys.executable

STEPS = [
    ("Training (3 models × 3 folds)",       CODE_DIR / "train_direct_segmentation.py"),
    ("Generating Figures",                    CODE_DIR / "generate_figures.py"),
    ("Generating Tables & Volumetric",        CODE_DIR / "generate_tables_and_volumetric.py"),
]

print("=" * 70)
print("  DIFFUSION SEGMENTATION — FULL PIPELINE")
print("=" * 70)
print(f"  Steps: {len(STEPS)}")
for i, (name, _) in enumerate(STEPS):
    print(f"  {i+1}. {name}")
print("=" * 70 + "\n")

overall_start = time.time()
for step_idx, (name, script) in enumerate(STEPS):
    print(f"\n{'='*70}")
    print(f"  STEP {step_idx+1}/{len(STEPS)}: {name}")
    print(f"  Script: {script.name}")
    print(f"{'='*70}\n")

    t0 = time.time()
    result = subprocess.run([PY, "-u", str(script)], check=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n[ERROR] Step {step_idx+1} failed (exit code {result.returncode})")
        print(f"  Script: {script}")
        print("  Pipeline stopped. Fix the error and re-run.")
        sys.exit(result.returncode)
    else:
        print(f"\n[OK] Step {step_idx+1} completed in {elapsed/60:.1f} min")

total = time.time() - overall_start
print(f"\n{'='*70}")
print(f"  PIPELINE COMPLETE in {total/60:.1f} min")
print(f"{'='*70}\n")
