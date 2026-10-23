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
        # Copies of uploads to the execution system aren't resumed: their
        # files were on this container's disk. Their uploaders are told to
        # upload again.
        from web.services.dataset_service import fail_interrupted_copies
        interrupted = fail_interrupted_copies()
        if interrupted:
            print(f"  Failed:   {interrupted} dataset copy(ies) interrupted by the restart")
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


# ── upload size limit ──────────────────────────────────────────────────────
# Starlette writes an uploaded file to a temporary file in full before any
# route code runs, so a route can only check the size after the disk space is
# already used. The limit is therefore enforced here, while the body arrives.
_UPLOAD_PATHS = {"/datasets/upload", "/api/datasets/upload"}
# Allowance for the multipart framing and the form's other fields.
_UPLOAD_OVERHEAD_BYTES = 1024 * 1024


class _UploadSizeLimit:
    """Refuse dataset uploads larger than settings.max_upload_bytes: up front
    when the request declares its length, otherwise as soon as the body
    passes the limit."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if not (scope["type"] == "http" and scope["method"] == "POST"
                and scope["path"] in _UPLOAD_PATHS):
            await self.app(scope, receive, send)
            return

        from fastapi.responses import JSONResponse
        from starlette.exceptions import HTTPException
        from web.config import settings
        from web.services.dataset_service import format_size

        limit = settings.max_upload_bytes + _UPLOAD_OVERHEAD_BYTES
        message = (f"File too large: the maximum upload size is "
                   f"{format_size(settings.max_upload_bytes)}.")

        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > limit:
            await JSONResponse({"detail": message}, status_code=413)(scope, receive, send)
            return

        received = 0

        async def counting_receive():
            nonlocal received
            event = await receive()
            if event["type"] == "http.request":
                received += len(event.get("body", b""))
                if received > limit:
                    raise HTTPException(413, message)
            return event

        await self.app(scope, counting_receive, send)


app.add_middleware(_UploadSizeLimit)


# ── static files ───────────────────────────────────────────────────────────
_STATIC = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

# ── routers ────────────────────────────────────────────────────────────────
from fastapi import Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse

from web.deps import AuthRequired, require_login
from web.routers import api, auth, ui


@app.exception_handler(AuthRequired)
async def _auth_required(request: Request, exc: AuthRequired):
    if request.url.path.startswith("/api"):
        return JSONResponse({"detail": exc.message}, status_code=401)
    import html
    return HTMLResponse(
        f"<!doctype html><title>Login required</title>"
        f"<body style='font-family:system-ui;max-width:32rem;margin:4rem auto'>"
        f"<h1>Login required</h1><p>{html.escape(exc.message)}</p></body>",
        status_code=401,
    )


# /health and /static are registered on `app` directly, above, so they stay
# open. The OAuth routes are the non-token way in and are left ungated.
_gate = [Depends(require_login)]
app.include_router(ui.router, dependencies=_gate)    # HTML pages (no prefix)
app.include_router(api.router, dependencies=_gate)   # JSON API (/api prefix)
app.include_router(auth.router)  # Tapis OAuth account-connection routes


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
