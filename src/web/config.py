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

# ── Tapis (remote GPU training on OSC) ─────────────────────────────────────
# Training runs as a Tapis job on an HPC system rather than in this container:
# the service submits, polls, and fetches the resulting model. With
# TAPIS_ENABLED=0 the service keeps its local behaviour, so `docker compose up`
# works with no credentials.
TAPIS_ENABLED = os.environ.get("TAPIS_ENABLED", "0") not in ("0", "", "false", "False")
TAPIS_BASE_URL = os.environ.get("TAPIS_BASE_URL", "https://icicleai.tapis.io")
TAPIS_USERNAME = os.environ.get("TAPIS_USERNAME", "")
TAPIS_PASSWORD = os.environ.get("TAPIS_PASSWORD", "")
TAPIS_JWT = os.environ.get("TAPIS_JWT", "")          # alternative to user/password
# Preferred credentials: an OAuth client plus a refresh token, so the service
# never stores a user's password. Obtain the refresh token once with
# tapis_service.bootstrap_refresh_token(). Tapis rotates refresh tokens on use,
# so the current one is persisted to the database, not read from here after the
# first refresh — this value only seeds it.
TAPIS_CLIENT_ID = os.environ.get("TAPIS_CLIENT_ID", "")
TAPIS_CLIENT_KEY = os.environ.get("TAPIS_CLIENT_KEY", "")
TAPIS_REFRESH_TOKEN = os.environ.get("TAPIS_REFRESH_TOKEN", "")
TAPIS_APP_ID = os.environ.get("TAPIS_APP_ID", "fnas-lcd-train")
TAPIS_APP_VERSION = os.environ.get("TAPIS_APP_VERSION", "0.1.3")
TAPIS_SYSTEM = os.environ.get("TAPIS_SYSTEM", "pitzer-tapis")
TAPIS_QUEUE = os.environ.get("TAPIS_QUEUE", "gpu-exp")
# How the dataset reaches the training container:
#   in_place -- bind-mount it from the execution system's filesystem (default).
#   copy     -- have Tapis transfer it into the job's input directory.
# in_place is the default because a dataset's remote_path already means "this
# exists on the execution system", so copying it is duplication -- and Tapis
# refuses any single transfer above 10,000 files, which real image datasets
# exceed easily (the OSA dataset is ~107,000).
TAPIS_STAGE_MODE = os.environ.get("TAPIS_STAGE_MODE", "in_place")
# Job working directories are created under here, e.g. /fs/ess/<project>/<user>.
# This must be set: a system's own default jobWorkingDir is typically owned by
# the system's owner and not writable by our effective user, so jobs that do not
# override it fail while processing inputs. Deliberately has no default, since
# the right path is deployment-specific.
TAPIS_WORK_BASE = os.environ.get("TAPIS_WORK_BASE", "")
TAPIS_POLL_SECONDS = int(os.environ.get("TAPIS_POLL_SECONDS", "30"))
TAPIS_MAX_MINUTES = int(os.environ.get("TAPIS_MAX_MINUTES", "1440"))

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
        # Where the same dataset lives on the Tapis execution system. Remote
        # training needs a path the HPC system can see; this container's own
        # paths are meaningless there. Datasets without a remote_path cannot be
        # trained remotely -- nas_runner fails them with an explicit message
        # rather than submitting a job that dies during staging. Unset by
        # default because the location is deployment-specific.
        "remote_path": os.environ.get("TOY_DATASET_REMOTE_PATH", "") or None,
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

    tapis_enabled: bool = TAPIS_ENABLED
    tapis_base_url: str = TAPIS_BASE_URL
    tapis_username: str = TAPIS_USERNAME
    tapis_password: str = TAPIS_PASSWORD
    tapis_jwt: str = TAPIS_JWT
    tapis_client_id: str = TAPIS_CLIENT_ID
    tapis_client_key: str = TAPIS_CLIENT_KEY
    tapis_refresh_token: str = TAPIS_REFRESH_TOKEN
    tapis_app_id: str = TAPIS_APP_ID
    tapis_app_version: str = TAPIS_APP_VERSION
    tapis_system: str = TAPIS_SYSTEM
    tapis_queue: str = TAPIS_QUEUE
    tapis_stage_mode: str = TAPIS_STAGE_MODE
    tapis_work_base: str = TAPIS_WORK_BASE
    tapis_poll_seconds: int = TAPIS_POLL_SECONDS
    tapis_max_minutes: int = TAPIS_MAX_MINUTES

    host: str = "0.0.0.0"
    port: int = 8000


def ensure_storage_dirs() -> None:
    """Create storage subdirectories if they don't exist."""
    for d in (UPLOADS_DIR, CHECKPOINTS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


settings = AppSettings()
