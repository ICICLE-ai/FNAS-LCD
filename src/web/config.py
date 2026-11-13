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

# Largest dataset archive accepted, in bytes. Enforced before an upload is
# written to disk (see main.py), and stated on the upload page.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024 * 1024  # 10 GB

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

# ── Tapis (remote GPU training on SDSC Expanse) ────────────────────────────
# Training runs as a Tapis job on Expanse rather than in this container: the
# service submits, polls, and fetches the resulting model. Uploaded datasets
# are copied there by Tapis jobs too. Every Tapis call uses the logged-in
# user's own token (see services/user_tokens.py); the service holds no Tapis
# credentials. With TAPIS_ENABLED=0 the service keeps its local behaviour, so
# `docker compose up` works with no Tapis at all.
TAPIS_ENABLED = os.environ.get("TAPIS_ENABLED", "0") not in ("0", "", "false", "False")
# When hosted behind a platform that logs users in and hands the app their
# Tapis token as an `X-Tapis-Token` cookie (e.g. an ICICLE pod), require a
# valid one on every page and API route except /health, /static and the
# upload links. Off by default so local `docker compose up` works with no
# portal in front of it.
REQUIRE_TAPIS_TOKEN = os.environ.get("REQUIRE_TAPIS_TOKEN", "0") not in ("0", "", "false", "False")
TAPIS_BASE_URL = os.environ.get("TAPIS_BASE_URL", "https://icicleai.tapis.io")
# The two apps the jobs run through (definitions in tapis/): one trains a
# model, the other copies an uploaded dataset onto Expanse or deletes one.
TAPIS_APP_ID = os.environ.get("TAPIS_APP_ID", "fnas-lcd-expanse-train")
TAPIS_APP_VERSION = os.environ.get("TAPIS_APP_VERSION", "0.1.0")
TAPIS_UPLOAD_APP_ID = os.environ.get("TAPIS_UPLOAD_APP_ID", "fnas-lcd-expanse-upload")
TAPIS_UPLOAD_APP_VERSION = os.environ.get("TAPIS_UPLOAD_APP_VERSION", "0.1.0")
TAPIS_POLL_SECONDS = int(os.environ.get("TAPIS_POLL_SECONDS", "30"))
# Expanse's Tapis queues end any job after 800 minutes, so a training job
# can't ask for more. Shown to users on the search page.
TAPIS_MAX_MINUTES = min(int(os.environ.get("TAPIS_MAX_MINUTES", "800")), 800)
# Expanse runs at most this many jobs per user per queue at once; more wait.
# Not enforced here, only stated on the search page.
EXPANSE_JOBS_PER_USER = 10
# SDSC deletes files in Expanse's scratch space, where the datasets are kept,
# this many days after they were written. Shown to users, and an upload is
# treated as gone once this has passed.
DATASET_LIFETIME_DAYS = 90
# This service's public address. Tapis downloads each upload from a temporary
# link under it (see services/upload_links.py).
PUBLIC_URL = os.environ.get("PUBLIC_URL", "https://fnaslcd.pods.icicleai.tapis.io").rstrip("/")

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
        # Where the same dataset lives on Expanse: the dataset folder of an
        # upload of it (see tapis_service.dataset_path). Remote training needs
        # a path Expanse can see; this container's own paths are meaningless
        # there. With remote training on, a built-in dataset without one isn't
        # offered (see database.init_db). Unset by default because the location
        # is deployment-specific.
        "remote_path": os.environ.get("TOY_DATASET_REMOTE_PATH", "") or None,
        "num_classes": 100,
        "is_upload": 0,
        "is_ready": 1,
        "description": "Built-in toy dataset (100 classes)",
    },
    {
        "name": "Ohio Small Animals",
        # Remote-only dataset: no local copy ships with this repo, and
        # none is needed -- `path` is never read from disk for a
        # pre-configured (non-uploaded) dataset, only `remote_path` is.
        # Not on Expanse yet, so unset by default.
        "path": os.environ.get("OSA_DATASET_REMOTE_PATH", ""),
        "remote_path": os.environ.get("OSA_DATASET_REMOTE_PATH", "") or None,
        "num_classes": 100,
        "is_upload": 0,
        "is_ready": 1,
        "description": "Ohio Small Animals (OSA), center-cropped to 224px (100 classes)",
    },
]


@dataclass
class AppSettings:
    """Runtime settings singleton."""

    storage_dir: Path = field(default_factory=lambda: STORAGE_DIR)
    uploads_dir: Path = field(default_factory=lambda: UPLOADS_DIR)
    checkpoints_dir: Path = field(default_factory=lambda: CHECKPOINTS_DIR)
    models_dir: Path = field(default_factory=lambda: MODELS_DIR)
    max_upload_bytes: int = MAX_UPLOAD_BYTES
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

    tapis_enabled: bool = TAPIS_ENABLED
    require_tapis_token: bool = REQUIRE_TAPIS_TOKEN
    tapis_base_url: str = TAPIS_BASE_URL
    tapis_app_id: str = TAPIS_APP_ID
    tapis_app_version: str = TAPIS_APP_VERSION
    tapis_upload_app_id: str = TAPIS_UPLOAD_APP_ID
    tapis_upload_app_version: str = TAPIS_UPLOAD_APP_VERSION
    tapis_poll_seconds: int = TAPIS_POLL_SECONDS
    tapis_max_minutes: int = TAPIS_MAX_MINUTES
    expanse_jobs_per_user: int = EXPANSE_JOBS_PER_USER
    dataset_lifetime_days: int = DATASET_LIFETIME_DAYS
    public_url: str = PUBLIC_URL

    host: str = "0.0.0.0"
    port: int = 8000


def ensure_storage_dirs() -> None:
    """Create storage subdirectories if they don't exist."""
    for d in (UPLOADS_DIR, CHECKPOINTS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


settings = AppSettings()
