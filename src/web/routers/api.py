"""JSON API routes for the FNAS-LCD web service."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from ..config import settings
from ..database import get_db
from ..services import storage_service
from ..models import (
    DatasetListResponse,
    DatasetResponse,
    DeviceConfigResponse,
    JobCreateRequest,
    JobListResponse,
    JobResponse,
    JobStatusResponse,
    MessageResponse,
    SearchRequest,
    SearchResponse,
)
from ..services import dataset_service

router = APIRouter(prefix="/api")


# ── helpers ────────────────────────────────────────────────────────────────

def _row_to_dataset(row) -> DatasetResponse:
    return DatasetResponse(
        id=row["id"],
        name=row["name"],
        path=row["path"],
        num_classes=row["num_classes"],
        is_upload=bool(row["is_upload"]),
        is_ready=bool(row["is_ready"]),
        description=row["description"] or "",
        created_at=row["created_at"] or "",
    )


def _row_to_job(row) -> JobResponse:
    row_keys = row.keys()
    return JobResponse(
        id=row["id"],
        dataset_id=row["dataset_id"],
        dataset_name=row["dataset_name"] if "dataset_name" in row_keys else None,
        device_type=row["device_type"],
        budget=row["budget"],
        metric=row["metric"],
        aggregate=row["aggregate"],
        epochs=row["epochs"],
        warmup=row["warmup"],
        batch_size=row["batch_size"],
        workers=row["workers"],
        lr=row["lr"],
        gpu=row["gpu"],
        max_iters=row["max_iters"],
        structure_str=row["structure_str"],
        status=row["status"],
        error_message=row["error_message"],
        best_val_acc=row["best_val_acc"],
        output_path=row["output_path"],
        save_dir=row["save_dir"],
        search_score=row["search_score"],
        search_constraint=row["search_constraint"],
        search_flops=row["search_flops"],
        search_params=row["search_params"],
        created_at=row["created_at"] or "",
        started_at=row["started_at"],
        finished_at=row["finished_at"],
    )


# ── device configs ─────────────────────────────────────────────────────────

@router.get("/devices", response_model=list[DeviceConfigResponse])
def list_devices():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM device_configs ORDER BY id").fetchall()
    return [DeviceConfigResponse(**dict(r)) for r in rows]


# ── datasets ───────────────────────────────────────────────────────────────

@router.get("/datasets", response_model=DatasetListResponse)
def list_datasets():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM datasets WHERE is_ready=1 ORDER BY id"
        ).fetchall()
    return DatasetListResponse(datasets=[_row_to_dataset(r) for r in rows])


@router.get("/datasets/{dataset_id}", response_model=DatasetResponse)
def get_dataset(dataset_id: int):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM datasets WHERE id=%s", (dataset_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Dataset not found")
    return _row_to_dataset(row)


@router.post("/datasets/upload", response_model=DatasetResponse)
async def upload_dataset(
    name: str = Query(..., description="Dataset display name"),
    file: UploadFile = File(...),
):
    """Upload a zip/tar archive in ImageFolder format."""
    if not file.filename:
        raise HTTPException(400, "Filename is required")

    allowed_ext = (".zip", ".tar", ".tar.gz", ".tgz")
    if not any(file.filename.endswith(ext) for ext in allowed_ext):
        raise HTTPException(400, f"Unsupported format. Allowed: {allowed_ext}")

    file_bytes = await file.read()
    if len(file_bytes) > 10 * 1024 * 1024 * 1024:  # 10 GB
        raise HTTPException(400, "File too large (max 10 GB)")

    try:
        info = dataset_service.add_uploaded_dataset(name, file_bytes, file.filename)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {e}")

    with get_db() as conn:
        row = conn.execute("SELECT * FROM datasets WHERE id=%s", (info["id"],)).fetchone()
    return _row_to_dataset(row)


@router.delete("/datasets/{dataset_id}", response_model=MessageResponse)
def delete_dataset(dataset_id: int):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM datasets WHERE id=%s", (dataset_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Dataset not found")
    if not row["is_upload"]:
        raise HTTPException(400, "Cannot delete pre-configured datasets")

    import shutil
    ds_path = Path(row["path"])
    if ds_path.is_dir():
        shutil.rmtree(ds_path, ignore_errors=True)

    with get_db() as conn:
        conn.execute("DELETE FROM datasets WHERE id=%s", (dataset_id,))
    return MessageResponse(message=f"Dataset {dataset_id} deleted")


# ── search stub (full impl in Phase 3) ─────────────────────────────────────

@router.post("/search", response_model=SearchResponse)
def run_search(req: SearchRequest):
    """Run architecture search under a latency/FLOPs budget."""
    from ..services.search_service import run_search as do_search

    # Lookup dataset info
    with get_db() as conn:
        ds = conn.execute("SELECT * FROM datasets WHERE id=%s", (req.dataset_id,)).fetchone()
        if ds is None:
            raise HTTPException(404, "Dataset not found")
        dev = conn.execute(
            "SELECT * FROM device_configs WHERE name=%s", (req.device_type,)
        ).fetchone()
        if dev is None:
            raise HTTPException(400, f"Unknown device type: {req.device_type}")

    try:
        result = do_search(
            lookup_csv=settings.lookup_csv,
            constraint_column=dev["constraint_column"],
            num_classes=ds["num_classes"],
            budget=req.budget,
        )
    except RuntimeError as e:
        raise HTTPException(400, str(e))

    from ..models import SearchResult
    return SearchResponse(
        result=SearchResult(**result),
        device_type=req.device_type,
        budget=req.budget,
        num_classes=ds["num_classes"],
    )


# ── jobs stub (full impl in Phase 4) ───────────────────────────────────────

@router.post("/jobs", response_model=JobResponse)
def create_job(req: JobCreateRequest):
    """Create and start a training job."""
    from ..services.job_manager import job_manager

    with get_db() as conn:
        ds = conn.execute("SELECT * FROM datasets WHERE id=%s", (req.dataset_id,)).fetchone()
        if ds is None:
            raise HTTPException(404, "Dataset not found")
        dev = conn.execute(
            "SELECT * FROM device_configs WHERE name=%s", (req.device_type,)
        ).fetchone()
        if dev is None:
            raise HTTPException(400, f"Unknown device type: {req.device_type}")

    try:
        job = job_manager.create_and_start(
            dataset_id=req.dataset_id,
            device_type=req.device_type,
            budget=req.budget,
            epochs=req.epochs,
            warmup=req.warmup,
            batch_size=req.batch_size,
            workers=req.workers,
            lr=req.lr,
            gpu=req.gpu,
            max_iters=req.max_iters,
            structure_str=req.structure_str,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    return job


@router.get("/jobs", response_model=JobListResponse)
def list_jobs(
    status: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    with get_db() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE status=%s ORDER BY id DESC LIMIT %s OFFSET %s",
                (status, limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY id DESC LIMIT %s OFFSET %s",
                (limit, offset),
            ).fetchall()
    return JobListResponse(jobs=[_row_to_job(r) for r in rows])


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: int):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=%s", (job_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Job not found")
    return _row_to_job(row)


@router.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
def get_job_status(job_id: int):
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, status, best_val_acc, error_message FROM jobs WHERE id=%s",
            (job_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(404, "Job not found")
    return JobStatusResponse(**dict(row))


@router.post("/jobs/{job_id}/cancel", response_model=MessageResponse)
def cancel_job(job_id: int):
    from ..services.job_manager import job_manager

    success = job_manager.cancel(job_id)
    if success:
        return MessageResponse(message=f"Job {job_id} cancelled")
    else:
        raise HTTPException(400, "Job cannot be cancelled in its current state")


# ── download ───────────────────────────────────────────────────────────────

@router.get("/download/{job_id}")
def download_model(job_id: int):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=%s", (job_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Job not found")
    if row["status"] != "completed":
        raise HTTPException(400, f"Job is not completed (status: {row['status']})")

    object_key = row["output_path"]
    if not object_key or not storage_service.object_exists(object_key):
        raise HTTPException(404, "Model file not found. It may have been cleaned up.")

    filename = Path(object_key).name
    data = storage_service.download_object(object_key)
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
