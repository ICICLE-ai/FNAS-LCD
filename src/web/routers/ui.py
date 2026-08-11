"""HTML page routes for the FNAS-LCD web service."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from ..database import get_db
from .api import _row_to_dataset, _row_to_job
from ..services import dataset_service

router = APIRouter()

_TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


# ── helpers ────────────────────────────────────────────────────────────────

def _get_datasets(conn) -> list:
    return conn.execute(
        "SELECT * FROM datasets WHERE is_ready=1 ORDER BY id"
    ).fetchall()


def _get_device_configs(conn) -> list:
    return conn.execute("SELECT * FROM device_configs ORDER BY id").fetchall()


def _get_jobs(conn, status=None, limit=50):
    if status:
        return conn.execute(
            "SELECT jobs.*, datasets.name AS dataset_name FROM jobs "
            "LEFT JOIN datasets ON datasets.id = jobs.dataset_id "
            "WHERE jobs.status=%s ORDER BY jobs.id DESC LIMIT %s",
            (status, limit),
        ).fetchall()
    return conn.execute(
        "SELECT jobs.*, datasets.name AS dataset_name FROM jobs "
        "LEFT JOIN datasets ON datasets.id = jobs.dataset_id "
        "ORDER BY jobs.id DESC LIMIT %s", (limit,)
    ).fetchall()


def _job_stats(conn) -> dict:
    rows = conn.execute(
        "SELECT status, COUNT(*) as cnt FROM jobs GROUP BY status"
    ).fetchall()
    stats = {r["status"]: r["cnt"] for r in rows}
    return {
        "total": sum(stats.values()),
        "completed": stats.get("completed", 0),
        "failed": stats.get("failed", 0),
        "running": stats.get("searching", 0) + stats.get("training", 0) + stats.get("exporting", 0),
        "pending": stats.get("pending", 0),
        "cancelled": stats.get("cancelled", 0),
    }


# ── pages ──────────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
def index(request: Request, submitted: int | None = Query(None)):
    with get_db() as conn:
        ds_count = conn.execute("SELECT COUNT(*) AS count FROM datasets WHERE is_ready=1").fetchone()["count"]
        stats = _job_stats(conn)
        recent = conn.execute(
            "SELECT jobs.*, datasets.name AS dataset_name FROM jobs "
            "LEFT JOIN datasets ON datasets.id = jobs.dataset_id "
            "ORDER BY jobs.id DESC LIMIT 10"
        ).fetchall()
    recent_jobs = [_row_to_job(r) for r in recent]
    active = {"pending", "searching", "training", "exporting"}
    return templates.TemplateResponse(request, "index.html", {
        "dataset_count": ds_count,
        "stats": stats,
        "recent_jobs": recent_jobs,
        "has_active": any(j.status in active for j in recent_jobs),
        "submitted": submitted,
    })


# ── datasets ───────────────────────────────────────────────────────────────

@router.get("/datasets", response_class=HTMLResponse)
def datasets_page(request: Request):
    with get_db() as conn:
        datasets = _get_datasets(conn)
    return templates.TemplateResponse(request, "datasets.html", {
        "datasets": [_row_to_dataset(r) for r in datasets],
    })


@router.get("/datasets/upload", response_class=HTMLResponse)
def dataset_upload_form(request: Request):
    return templates.TemplateResponse(request, "dataset_upload.html")


@router.post("/datasets/upload", response_class=HTMLResponse)
async def dataset_upload_handler(
    request: Request,
    name: str = Form(...),
    file: UploadFile = File(...),
):
    if not file.filename:
        return templates.TemplateResponse(request, "dataset_upload.html", {
            "error": "Filename is required",
        })

    allowed = (".zip", ".tar", ".tar.gz", ".tgz")
    if not any(file.filename.endswith(ext) for ext in allowed):
        return templates.TemplateResponse(request, "dataset_upload.html", {
            "error": f"Unsupported format. Allowed: {allowed}",
        })

    file_bytes = await file.read()
    try:
        info = dataset_service.add_uploaded_dataset(name, file_bytes, file.filename)
    except Exception as e:
        return templates.TemplateResponse(request, "dataset_upload.html", {
            "error": str(e),
        })

    return RedirectResponse(f"/datasets/{info['id']}", status_code=303)


@router.get("/datasets/{dataset_id}", response_class=HTMLResponse)
def dataset_detail(request: Request, dataset_id: int):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM datasets WHERE id=%s", (dataset_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "Dataset not found")
        jobs = conn.execute(
            "SELECT * FROM jobs WHERE dataset_id=%s ORDER BY id DESC LIMIT 20",
            (dataset_id,),
        ).fetchall()
    return templates.TemplateResponse(request, "dataset_detail.html", {
        "dataset": _row_to_dataset(row),
        "jobs": [_row_to_job(j) for j in jobs],
    })


# ── search ─────────────────────────────────────────────────────────────────

@router.get("/search", response_class=HTMLResponse)
def search_form(request: Request):
    with get_db() as conn:
        datasets = _get_datasets(conn)
        devices = _get_device_configs(conn)
    return templates.TemplateResponse(request, "search.html", {
        "datasets": [_row_to_dataset(r) for r in datasets],
        "devices": [dict(r) for r in devices],
    })


@router.post("/search", response_class=HTMLResponse)
def search_run(
    request: Request,
    dataset_id: int = Form(...),
    device_type: str = Form(...),
    budget: float = Form(...),
):
    """Submit a job: search + export run in the background (demo: training
    is skipped). Redirects home with a 'submitted' flash; track progress
    and download from the job's page."""
    from ..services.job_manager import job_manager

    with get_db() as conn:
        datasets = _get_datasets(conn)
        devices = _get_device_configs(conn)
        ds = conn.execute("SELECT * FROM datasets WHERE id=%s", (dataset_id,)).fetchone()
        dev = conn.execute("SELECT * FROM device_configs WHERE name=%s", (device_type,)).fetchone()

    if ds is None:
        raise HTTPException(404, "Dataset not found")
    if dev is None:
        raise HTTPException(400, f"Unknown device: {device_type}")

    try:
        job = job_manager.create_and_start(
            dataset_id=dataset_id,
            device_type=device_type,
            budget=budget,
        )
    except ValueError as e:
        return templates.TemplateResponse(request, "search.html", {
            "datasets": [_row_to_dataset(r) for r in datasets],
            "devices": [dict(r) for r in devices],
            "error": str(e),
        })

    return RedirectResponse(f"/?submitted={job.id}", status_code=303)


# ── jobs ───────────────────────────────────────────────────────────────────

@router.get("/jobs", response_class=HTMLResponse)
def jobs_page(request: Request, status: str | None = Query(None)):
    with get_db() as conn:
        jobs = _get_jobs(conn, status=status)
        stats = _job_stats(conn)
    job_list = [_row_to_job(j) for j in jobs]
    active = {"pending", "searching", "training", "exporting"}
    return templates.TemplateResponse(request, "jobs.html", {
        "jobs": job_list,
        "has_active": any(j.status in active for j in job_list),
        "stats": stats,
        "filter_status": status or "",
    })


@router.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_detail(request: Request, job_id: int):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=%s", (job_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "Job not found")
        ds = conn.execute("SELECT * FROM datasets WHERE id=%s", (row["dataset_id"],)).fetchone()
    return templates.TemplateResponse(request, "job_detail.html", {
        "job": _row_to_job(row),
        "dataset": _row_to_dataset(ds) if ds else None,
    })
