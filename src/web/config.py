"""Central config for the FNAS-LCD web service."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# PROJECT_ROOT is src/ (this file's sibling packages — model/, training/,
# etc. — live there). REPO_ROOT_DIR is the true repo root one level up,
# where non-code assets (data/) actually live.
_WEB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _WEB_DIR.parent
REPO_ROOT_DIR = PROJECT_ROOT.parent

# ── storage paths ──────────────────────────────────────────────────────────
# uploads/checkpoints stay on local disk (need real POSIX paths for
# ImageFolder/DataLoader); models/ is used only as a transient local staging
# spot before upload to object storage (see storage_service.py).
STORAGE_DIR = _WEB_DIR / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
CHECKPOINTS_DIR = STORAGE_DIR / "checkpoints"
MODELS_DIR = STORAGE_DIR / "models"

# ── database (Postgres) ──────────────────────────────────────────────────────
# Defaults match the `db` service in docker-compose.yml; override via env
# vars for bare local dev against a different Postgres instance.
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "db")
POSTGRES_PORT = int(os.environ.get("POSTGRES_PORT", "5432"))
POSTGRES_USER = os.environ.get("POSTGRES_USER", "fnas_lcd")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "fnas_lcd")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "fnas_lcd")

# ── object storage (S3-compatible — MinIO locally) ──────────────────────────
# Defaults match the `storage` service in docker-compose.yml.
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL", "http://storage:9000")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "fnas_lcd")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "fnas_lcd_secret")
S3_BUCKET = os.environ.get("S3_BUCKET", "fnas-lcd-models")

# ── lookup table ───────────────────────────────────────────────────────────
LOOKUP_CSV = REPO_ROOT_DIR / "data" / "block_lookup" / "idw_block_lookup_jetson_gpu.csv"

# ── model path (for training subprocess) ───────────────────────────────────
MODEL_ROOT = PROJECT_ROOT / "model"
REPO_ROOT = PROJECT_ROOT

# ── training entrypoint ────────────────────────────────────────────────────
TIMM_TRAINER = PROJECT_ROOT / "training" / "timm_trainer.py"

# ── default device configs ─────────────────────────────────────────────────
DEVICE_CONFIGS = [
    {
        "name": "jetson_gpu",
        "lookup_csv": "data/block_lookup/idw_block_lookup_jetson_gpu.csv",
        "constraint_column": "jetson_gpu_latency_ms_median",
        "display_name": "Jetson GPU (latency ms)",
    },
    {
        "name": "flops",
        "lookup_csv": "data/block_lookup/idw_block_lookup_jetson_gpu.csv",
        "constraint_column": "flops",
        "display_name": "FLOPs-constrained",
    },
]

# ── pre-configured datasets ────────────────────────────────────────────────
PRECONFIGURED_DATASETS = [
    {
        "name": "toy_dataset",
        "path": str(REPO_ROOT_DIR / "data" / "toy_dataset"),
        "num_classes": 100,
        "is_upload": 0,
        "is_ready": 1,
        "description": "Built-in toy dataset (100 classes)",
    },
]


@dataclass
class AppSettings:
    """Runtime settings singleton."""

    storage_dir: Path = field(default_factory=lambda: STORAGE_DIR)
    uploads_dir: Path = field(default_factory=lambda: UPLOADS_DIR)
    checkpoints_dir: Path = field(default_factory=lambda: CHECKPOINTS_DIR)
    models_dir: Path = field(default_factory=lambda: MODELS_DIR)
    lookup_csv: Path = field(default_factory=lambda: LOOKUP_CSV)
    model_root: Path = field(default_factory=lambda: MODEL_ROOT)
    repo_root: Path = field(default_factory=lambda: REPO_ROOT)
    timm_trainer: Path = field(default_factory=lambda: TIMM_TRAINER)
    device_configs: list[dict] = field(default_factory=lambda: DEVICE_CONFIGS.copy())
    preconfigured_datasets: list[dict] = field(default_factory=lambda: PRECONFIGURED_DATASETS.copy())

    postgres_host: str = POSTGRES_HOST
    postgres_port: int = POSTGRES_PORT
    postgres_user: str = POSTGRES_USER
    postgres_password: str = POSTGRES_PASSWORD
    postgres_db: str = POSTGRES_DB

    s3_endpoint_url: str = S3_ENDPOINT_URL
    s3_access_key: str = S3_ACCESS_KEY
    s3_secret_key: str = S3_SECRET_KEY
    s3_bucket: str = S3_BUCKET

    host: str = "0.0.0.0"
    port: int = 8000


def ensure_storage_dirs() -> None:
    """Create storage subdirectories if they don't exist."""
    for d in (UPLOADS_DIR, CHECKPOINTS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


settings = AppSettings()
