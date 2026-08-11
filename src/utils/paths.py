"""Central path configuration for the CAP-NAS pipeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# PROJECT_ROOT is the src/ directory (this file's sibling packages —
# model/, training/, etc. — live there). REPO_ROOT_DIR is the true repo
# root one level up, where non-code assets (data/) actually live.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT
REPO_ROOT_DIR = PROJECT_ROOT.parent
MODEL_ROOT = PROJECT_ROOT / "model"
DATA_DIR = REPO_ROOT_DIR / "data" / "block_lookup"
CONFIG_PATH = REPO_ROOT_DIR / "config" / "defaults.yaml"

LOOKUP_GPU_CSV = DATA_DIR / "idw_block_lookup_jetson_gpu.csv"
LOOKUP_CSV = LOOKUP_GPU_CSV
LOOKUP_CPU_CSV = DATA_DIR / "idw_block_lookup_cpu_x86.csv"

ORIGINAL_SEED_STRUCTURE = (
    "SuperConvK3BNRELU(3,8,2,1)"
    "SuperResIDWE6K3(8,32,2,8,1)"
    "SuperResIDWE6K3(32,48,2,32,1)"
    "SuperResIDWE6K3(48,96,2,48,1)"
    "SuperResIDWE6K3(96,128,2,96,1)"
    "SuperConvK1BNRELU(128,2048,1,1)"
)
MAX_LAYERS = 14

def setup_model_path() -> Path:
    """Ensure PlainNet/MasterNet/ModelLoader and global_utils are importable."""
    utils_root = str(PROJECT_ROOT / "utils")
    for root in (str(MODEL_ROOT), utils_root):
        if root not in sys.path:
            sys.path.insert(0, root)
    return MODEL_ROOT


def setup_repo_path() -> Path:
    """Add project root so top-level packages import when running scripts directly."""
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return PROJECT_ROOT


def ensure_lookup_symlinks() -> None:
    """No-op: full lookup CSV is shipped under data/block_lookup/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def masternet_arch() -> str:
    return f"{MODEL_ROOT / 'Masternet.py'}:MasterNet"
