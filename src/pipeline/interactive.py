#!/usr/bin/env python3
"""Interactive AutoNAS pipeline.

Prompts the user for:
  - Latency budget (ms)
  - Dataset path (default: repo's toy dataset)
  - Epochs (default: 2)
  - Output path (default: output/model_<timestamp>.pt)

Then runs:
  1. Detect classes from ImageFolder
  2. Enumerate best architecture under latency budget
  3. Train via the original timm_trainer.py recipe
  4. Export TorchScript model

All core logic is shared with pipeline.auto_nas.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

# _PROJ is src/ (this file's sibling packages live there); _REPO_ROOT is
# the true repo root one level up, where non-code assets (data/) live.
_PROJ = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PROJ.parent
for _p in (str(_PROJ), str(_PROJ / "model"), str(_PROJ / "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

# Reuse pipeline internals
from pipeline.auto_nas import (  # noqa: E402
    _detect_num_classes,
    _enumerate_best,
    _export_best_checkpoint,
    _train_via_timm_trainer,
    _write_structure,
)

TOY_DATASET = _REPO_ROOT / "data" / "toy_dataset"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "output"

# ---------------------------------------------------------------------------
# Interactive prompts
# ---------------------------------------------------------------------------


def _prompt_float(prompt: str, default: float | None = None) -> float:
    """Prompt for a positive float, re-ask on invalid input."""
    default_str = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{default_str}: ").strip()
        if not raw and default is not None:
            return default
        try:
            val = float(raw)
            if val > 0:
                return val
            print("  Must be positive.", file=sys.stderr)
        except ValueError:
            print("  Invalid number.", file=sys.stderr)


def _prompt_path(prompt: str, default: Path) -> Path:
    """Prompt for a dataset path; must exist and contain train/ ."""
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        path = Path(raw).resolve() if raw else default
        if not path.is_dir():
            print(f"  Directory not found: {path}", file=sys.stderr)
            continue
        if not (path / "train").is_dir():
            print(f"  Missing train/ subdirectory in: {path}", file=sys.stderr)
            continue
        return path


def _prompt_int(prompt: str, default: int) -> int:
    """Prompt for a non-negative integer."""
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            val = int(raw)
            if val >= 0:
                return val
            print("  Must be non-negative.", file=sys.stderr)
        except ValueError:
            print("  Invalid integer.", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main interactive flow
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("  AutoNAS — Automatic Neural Architecture Search Pipeline")
    print("=" * 60)
    print()
    print("This pipeline will:")
    print("  1. Search the IDW-fixfc space for the best architecture")
    print("     under your latency budget")
    print("  2. Train that architecture on your ImageFolder dataset")
    print("     (original recipe: auto_augment, random_erase, mixup, ")
    print("     label_smoothing, nesterov, cosine LR with warmup)")
    print("  3. Export the trained model as TorchScript (.pt)")
    print()

    # --- latency budget ---
    latency_ms = _prompt_float("Latency budget (ms)", default=20.0)
    print(f"  → {latency_ms} ms\n")

    # --- dataset path ---
    print("Dataset path (ImageFolder format with train/ val/ test/ subdirs).")
    print(f"  Toy dataset (100 classes, ~200 imgs/split): {TOY_DATASET}")
    data_dir = _prompt_path("Dataset path", default=TOY_DATASET)
    print(f"  → {data_dir}\n")

    # --- epochs ---
    epochs = _prompt_int("Training epochs (2 for quick test, 100+ for real)", default=2)
    print(f"  → {epochs}\n")

    # --- warmup ---
    default_warmup = 0 if epochs <= 5 else 5
    warmup = _prompt_int("LR warmup epochs (0 for short smoke runs)", default=default_warmup)
    print(f"  → {warmup}\n")

    # --- output ---
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = DEFAULT_OUTPUT_DIR / f"model_{ts}.pt"
    raw = input(f"Output path [{default_output}]: ").strip()
    output_path = Path(raw).resolve() if raw else default_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  → {output_path}\n")

    # --- GPU ---
    gpu = 0
    if not torch.cuda.is_available():
        print("CUDA not available — using CPU.")
        gpu = -1

    # --- Confirm ---
    print("-" * 60)
    print("Summary:")
    print(f"  Dataset:        {data_dir}")
    print(f"  Latency budget: {latency_ms} ms")
    print(f"  Epochs:         {epochs}")
    print(f"  Warmup epochs:  {warmup}")
    print(f"  Output:         {output_path}")
    print(f"  Device:         {'cuda:0' if gpu >= 0 else 'cpu'}")
    print("-" * 60)
    confirm = input("Proceed? [Y/n]: ").strip().lower()
    if confirm and confirm != "y":
        print("Aborted.")
        return

    # ===================================================================
    # Run pipeline
    # ===================================================================
    lookup_csv = _REPO_ROOT / "data" / "block_lookup" / "idw_block_lookup_jetson_gpu.csv"

    # Step 1: Detect classes
    print("\n[1/4] Detecting classes ...")
    num_classes = _detect_num_classes(data_dir)

    # Step 2: Enumerate
    print("\n[2/4] Searching architectures ...")
    structure_str = _enumerate_best(
        lookup_csv, num_classes, latency_ms,
        metric="analytic_ssr_exp_proj_dw_mean",
        aggregate="sum",
    )

    # Step 3: Train (original timm_trainer.py recipe)
    print(f"\n[3/4] Training ({epochs} epochs, original recipe) ...")
    save_dir = _REPO_ROOT / "artifacts" / f"autonas_{ts}"
    save_dir.mkdir(parents=True, exist_ok=True)
    structure_txt_path = _write_structure(save_dir, structure_str)

    _train_via_timm_trainer(
        structure_txt_path, num_classes, data_dir, save_dir,
        epochs=epochs,
        batch_size_per_gpu=64,
        workers_per_gpu=4,
        lr_per_256=0.1,
        warmup=warmup,
        gpu=gpu,
    )

    # Step 4: Export
    print("\n[4/4] Exporting TorchScript model ...")
    _export_best_checkpoint(
        structure_str, num_classes, save_dir, output_path,
    )

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print(f"  Architecture: {structure_str}")
    print(f"  Model saved to: {output_path}")
    print(f"  Checkpoints/logs kept at: {save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
