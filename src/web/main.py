"""FastAPI application — FNAS-LCD Web Service.

Start with (from the repo root):
    python -m src.web.main
    python -m src.web.main --port 8080
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# Ensure project root is importable
_PROJ = Path(__file__).resolve().parents[1]
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize DB, storage dirs, and object storage bucket on startup."""
    from web.config import settings, ensure_storage_dirs
    from web.database import init_db
    from web.services.storage_service import ensure_bucket

    ensure_storage_dirs()
    init_db()
    ensure_bucket()
    print(f"  Database: {settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}")
    print(f"  Storage:  {settings.storage_dir} (local) + {settings.s3_endpoint_url}/{settings.s3_bucket} (models)")
    if settings.tapis_enabled:
        print(f"  Training: {settings.tapis_base_url} "
              f"app={settings.tapis_app_id}:{settings.tapis_app_version} "
              f"system={settings.tapis_system} queue={settings.tapis_queue}")
        # Remote jobs outlive this process. Re-attach to any that were still
        # running when we last stopped, otherwise they finish unobserved and
        # their models are never collected.
        from web.services.job_manager import job_manager
        resumed = job_manager.resume_interrupted()
        if resumed:
            print(f"  Resumed:  {resumed} in-flight training job(s)")
    else:
        print("  Training: local (TAPIS_ENABLED=0 — models are not trained)")
    print(f"  Server:   http://{settings.host}:{settings.port}")
    yield


app = FastAPI(
    title="FNAS-LCD",
    description="Latency-Constrained Neural Architecture Search Web Service",
    version="0.1.0",
    lifespan=lifespan,
)


# ── platform health check (required by ICICLE's deploy platform) ───────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "fnas-lcd",
        "version": app.version,
        # Set by the ICICLE Dockerfile at build time; "unknown" outside that build.
        "build_sha": os.environ.get("BUILD_SHA", "unknown"),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


# ── static files ───────────────────────────────────────────────────────────
_STATIC = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

# ── routers ────────────────────────────────────────────────────────────────
from web.routers import api, ui

app.include_router(ui.router)    # HTML pages (no prefix)
app.include_router(api.router)   # JSON API (/api prefix)


# ── CLI ────────────────────────────────────────────────────────────────────
def main():
    from web.config import settings

    parser = argparse.ArgumentParser(description="FNAS-LCD Web Service")
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    args = parser.parse_args()

    settings.host = args.host
    settings.port = args.port

    import uvicorn
    uvicorn.run(
        "src.web.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
