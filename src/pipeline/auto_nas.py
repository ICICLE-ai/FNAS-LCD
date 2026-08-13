#!/usr/bin/env python3
"""Automatic end-to-end NAS pipeline.

Single command: given a dataset (ImageFolder) and a latency budget (ms),
search the IDW-fixfc space for the best architecture, train it using the
repo's original timm_trainer.py recipe (auto_augment, random_erase, bn_momentum
override, nesterov, mixup+label-smoothing, instance-count cosine LR, atomic
best/latest checkpointing, auto-resume), and export a TorchScript model.

Usage:
    python -m pipeline.auto_nas \
        --data-dir /path/to/imagefolder \
        --latency-budget-ms 20 \
        --output model.pt \
        --epochs 100 \
        --gpu 0
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Path setup (must come before local imports)
# ---------------------------------------------------------------------------
# _PROJ is src/ (this file's sibling packages live there); _REPO_ROOT is
# the true repo root one level up, where non-code assets (data/) live.
_PROJ = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PROJ.parent
for _p in (str(_PROJ), str(_PROJ / "model"), str(_PROJ / "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.paths import MODEL_ROOT, REPO_ROOT, masternet_arch  # noqa: E402

TIMM_TRAINER = _PROJ / "training" / "timm_trainer.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_num_classes(data_dir: Path) -> int:
    """Count class subdirectories under data_dir/train/."""
    train_dir = data_dir / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Expected train/ subdirectory in {data_dir}")
    classes = sorted(
        d.name for d in train_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    if not classes:
        raise RuntimeError(f"No class subdirectories found in {train_dir}")
    print(f"Detected {len(classes)} classes in {train_dir}")
    return len(classes)


# ---------------------------------------------------------------------------
# Enumeration step
# ---------------------------------------------------------------------------

def _enumerate_best(
    lookup_csv: Path,
    num_classes: int,
    latency_budget_ms: float,
    metric: str = "analytic_ssr_exp_proj_dw_mean",
    aggregate: str = "sum",
) -> str:
    """Run frontier DP search, return the best structure string."""
    import io
    from enumeration.frontier_enum import (  # noqa: E402
        choose_best,
        read_fc_lookup,
        read_lookup,
        search,
    )

    print(f"Searching architectures (budget={latency_budget_ms}ms) ...", flush=True)

    # Suppress verbose output from enumeration internals
    saved_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        grouped = read_lookup(
            str(lookup_csv),
            "jetson_gpu_latency_ms_median",
            metric,
            candidates_per_in_ch=200,
        )

        # FC block costs from CSV (both SuperConvK1BNRELU and Linear rows)
        proj_cost, linear_cost = read_fc_lookup(
            str(lookup_csv), "jetson_gpu_latency_ms_median", num_classes
        )
        if linear_cost == 0.0 and num_classes != 100:
            _, fallback_linear = read_fc_lookup(
                str(lookup_csv), "jetson_gpu_latency_ms_median", 100
            )
            linear_cost = fallback_linear

        def fc_cost_fn(oc: int) -> float:
            return float(proj_cost.get(oc, 0.0)) + float(linear_cost)

        finals = search(
            grouped=grouped,
            metric=metric,
            agg=aggregate,
            target=latency_budget_ms,
            bin_size=0.1,
            max_factor=1.0,
            states_per_key=3,
            max_layers=14,
            fc_resolution=7,
            skip_stem_in_agg=True,
            fc_cost_fn=fc_cost_fn,
        )
    finally:
        sys.stdout = saved_stdout

    best_list = choose_best(finals, aggregate, latency_budget_ms, tolerance=0.0)
    if not best_list:
        raise RuntimeError(
            f"No architectures found within {latency_budget_ms}ms budget. "
            "Try increasing --latency-budget-ms."
        )

    best = best_list[0]
    structure_str = "".join(best.cfgs)
    print(f"  Best: score={aggregate_score(best, aggregate):.4f}  "
          f"latency={best.constraint:.4f}ms  flops={best.flops/1e6:.1f}M  "
          f"layers={best.total_sublayers}")
    print(f"  Structure: {structure_str}")
    return structure_str


# Need this import for the score display above
from scoring.cross_stage_aggregation import aggregate_score  # noqa: E402


# ---------------------------------------------------------------------------
# Training — subprocess into training/timm_trainer.py with the original recipe
# ---------------------------------------------------------------------------

def _write_structure(save_dir: Path, structure_str: str) -> Path:
    structure_txt_path = save_dir / "structure.txt"
    structure_txt_path.write_text(structure_str + "\n", encoding="utf-8")
    return structure_txt_path


def _timm_trainer_env() -> dict:
    import os
    env = os.environ.copy()
    pythonpath_parts = [str(MODEL_ROOT), str(REPO_ROOT), env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = ":".join(p for p in pythonpath_parts if p)
    return env


def _train_via_timm_trainer(
    structure_str: str,
    num_classes: int,
    data_dir: Path,
    save_dir: Path,
    epochs: int,
    batch_size_per_gpu: int,
    workers_per_gpu: int,
    lr_per_256: float,
    warmup: int,
    gpu: int,
    input_image_size: int = 224,
    max_iters: int | None = None,
) -> None:
    """Train via the original timm_trainer.py recipe: auto_augment, random_erase,
    bn_momentum override, nesterov, mixup+label_smoothing, cosine LR with warmup,
    atomic best/latest checkpointing, auto-resume.

    max_iters, if set, caps training to that many mini-batch iterations total
    (forces a single epoch) — for a fast demo run that still produces a real
    checkpoint to export."""
    use_cpu = gpu < 0 or not torch.cuda.is_available()
    if use_cpu and gpu >= 0:
        print("CUDA not available — falling back to CPU training.")

    if max_iters is not None:
        epochs = 1
        print(f"Demo mode: capping training to {max_iters} iterations.")

    # The cosine LR schedule needs at least one post-warmup epoch
    # (stop_instances must exceed warmup_instances); clamp so short runs
    # (e.g. epochs=1 for a smoke test) never crash on warmup epochs alone.
    effective_warmup = min(warmup, max(epochs - 1, 0))
    if effective_warmup != warmup:
        print(f"Warmup reduced from {warmup} to {effective_warmup} epochs "
              f"(cannot exceed epochs-1 for epochs={epochs}).")

    command = [
        sys.executable,
        str(TIMM_TRAINER),
        "--model_type", "zen",
        "--arch", masternet_arch(),
        "--plainnet_struct", structure_str,
        "--dataset", "auto_imagefolder",
        "--num_classes", str(num_classes),
        "--input_image_size", str(input_image_size),
        "--data_dir", str(data_dir),
        "--img_per_class", "-1",
        "--frac", "1",
        "--epfrac", "1",
        "--epochs", str(epochs),
        "--batch_size", "256",
        "--batch_size_per_gpu", str(batch_size_per_gpu),
        "--workers_per_gpu", str(workers_per_gpu),
        "--dist_mode", "cpu" if use_cpu else "single",
        "--optimizer", "sgd",
        "--lr_per_256", str(lr_per_256),
        "--target_lr_per_256", "0.0",
        "--lr_mode", "cosine",
        "--warmup", str(effective_warmup),
        "--bn_momentum", "0.01",
        "--wd", "5e-4",
        "--weight_init", "xavier",
        "--nesterov",
        "--label_smoothing",
        "--random_erase",
        "--mixup",
        "--auto_augment",
        "--save_dir", str(save_dir),
        "--summary_file", str(save_dir / "summary.csv"),
        "--completion_file", str(save_dir / "train.done"),
        "--expid", "1",
        "--print_freq", "10000000",
        "--auto_resume",
    ]
    if not use_cpu:
        command += ["--gpu", str(gpu)]
    if max_iters is not None:
        command += ["--max_iters", str(max_iters)]

    subprocess.run(command, cwd=str(MODEL_ROOT), check=True, env=_timm_trainer_env())


def _export_best_checkpoint(
    structure_str: str,
    num_classes: int,
    save_dir: Path,
    output_path: Path,
    input_image_size: int = 224,
) -> None:
    """Load the best checkpoint saved by timm_trainer.py's training run and
    export it directly to TorchScript (no ONNX — avoids the onnxscript
    dependency entirely)."""
    from Masternet import MasterNet  # noqa: E402

    best_path = save_dir / "best-params_rank0.pth"
    if not best_path.exists():
        best_path = save_dir / "best-params.pth"
    if not best_path.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found in {save_dir}. "
            "Expected best-params_rank0.pth or best-params.pth"
        )

    model = MasterNet(plainnet_struct=structure_str, num_classes=num_classes, no_create=False)
    checkpoint = torch.load(str(best_path), map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    dummy = torch.randn(1, 3, input_image_size, input_image_size)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output_path))
    print(f"Model exported to {output_path}")

    # Verify
    loaded = torch.jit.load(str(output_path))
    with torch.no_grad():
        out = loaded(dummy)
    print(f"  Verification: output shape = {tuple(out.shape)}")


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AutoNAS: automatic architecture search + training + export"
    )
    p.add_argument("--data-dir", "--data_dir", dest="data_dir", required=True,
                   help="Path to ImageFolder dataset (must contain train/ subdir)")
    p.add_argument("--latency-budget-ms", "--latency_budget_ms", dest="latency_budget_ms",
                   type=float, required=True,
                   help="Jetson GPU latency ceiling in milliseconds")
    p.add_argument("--output", required=True,
                   help="Output path for TorchScript model (.pt)")
    p.add_argument("--epochs", type=int, default=100,
                   help="Training epochs (default: 100)")
    p.add_argument("--warmup", type=int, default=5,
                   help="LR warmup epochs (default: 5; use 0 for short smoke runs)")
    p.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=64,
                   help="Batch size per GPU (default: 64)")
    p.add_argument("--workers", type=int, default=4,
                   help="DataLoader workers (default: 4)")
    p.add_argument("--lr", type=float, default=0.1,
                   help="Base learning rate per 256 batch size (default: 0.1)")
    p.add_argument("--gpu", type=int, default=0,
                   help="GPU device ID (default: 0, use -1 for CPU)")
    p.add_argument("--metric", default="analytic_ssr_exp_proj_dw_mean",
                   help="Inner scoring metric (default: analytic_ssr_exp_proj_dw_mean)")
    p.add_argument("--aggregate", default="sum",
                   choices=["sum", "min", "geomean", "harmonic"],
                   help="Cross-stage aggregation (default: sum)")
    p.add_argument("--save-dir", "--save_dir", dest="save_dir", default=None,
                   help="Directory for checkpoints (default: temp dir)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.gpu >= 0 and not torch.cuda.is_available():
        print("CUDA not available — falling back to CPU training.")
    print(f"GPU: {args.gpu if args.gpu >= 0 and torch.cuda.is_available() else 'cpu'}")

    lookup_csv = _REPO_ROOT / "data" / "block_lookup" / "idw_block_lookup_jetson_gpu.csv"

    # ---- Step 1: Detect classes ----
    num_classes = _detect_num_classes(data_dir)

    # ---- Step 2: Enumerate best architecture ----
    structure_str = _enumerate_best(
        lookup_csv, num_classes, args.latency_budget_ms,
        metric=args.metric, aggregate=args.aggregate,
    )

    # ---- Step 3: Train (original timm_trainer.py recipe) ----
    save_dir = Path(args.save_dir) if args.save_dir else Path(tempfile.mkdtemp(prefix="autonas_"))
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint dir: {save_dir}")

    print(f"Training for {args.epochs} epochs (original recipe: auto_augment, "
          f"random_erase, bn_momentum, nesterov, mixup, label_smoothing) ...")
    _train_via_timm_trainer(
        structure_str, num_classes, data_dir, save_dir,
        epochs=args.epochs,
        batch_size_per_gpu=args.batch_size,
        workers_per_gpu=args.workers,
        lr_per_256=args.lr,
        warmup=args.warmup,
        gpu=args.gpu,
    )

    # ---- Step 4: Export (TorchScript, direct from best checkpoint) ----
    _export_best_checkpoint(
        structure_str, num_classes, save_dir, output_path,
    )

    # Clean up temp dir
    if args.save_dir is None:
        shutil.rmtree(save_dir, ignore_errors=True)

    print(f"\nDone. Model saved to {output_path}")


if __name__ == "__main__":
    main()
