"""JSON API routes for the FNAS-LCD web service."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from ..config import settings
from ..database import get_db
from ..deps import get_session
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
from ..services import dataset_service, resumable_uploads, user_tokens

router = APIRouter(prefix="/api")


# ── helpers ────────────────────────────────────────────────────────────────

def _dataset_status(row) -> str:
    """See DatasetResponse.status."""
    if row.get("copy_status") in ("uploading", "checking", "copying", "deleting", "failed"):
        return row["copy_status"]
    if dataset_service.is_expired(row):
        return "expired"
    return "ready" if row["is_ready"] else "uploading"


def _dataset_for_jobs(conn, dataset_id: int, user: str | None):
    """The dataset a search or job asks for, if `user` may use it."""
    row = conn.execute("SELECT * FROM datasets WHERE id=%s", (dataset_id,)).fetchone()
    if row is None or not dataset_service.can_use(row, user):
        raise HTTPException(404, "Dataset not found")
    return row


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
        uploaded_by=row.get("uploaded_by"),
        status=_dataset_status(row),
        status_detail=row.get("copy_error") if row.get("copy_status") == "failed" else None,
        expires_on=dataset_service.expires_on(row),
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
        submitted_by=row["submitted_by"] if "submitted_by" in row_keys else None,
    )


# ── device configs ─────────────────────────────────────────────────────────

@router.get("/devices", response_model=list[DeviceConfigResponse])
def list_devices():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM device_configs ORDER BY id").fetchall()
    return [DeviceConfigResponse(**dict(r)) for r in rows]


# ── datasets ───────────────────────────────────────────────────────────────

@router.get("/datasets", response_model=DatasetListResponse)
def list_datasets(session: dict = Depends(get_session)):
    """The datasets you can search and train with: the built-in ones and
    your own uploads (an admin's: every upload)."""
    with get_db() as conn:
        rows = dataset_service.usable_datasets(conn, session.get("tapis_username"))
    return DatasetListResponse(datasets=[_row_to_dataset(r) for r in rows])


@router.get("/datasets/{dataset_id}", response_model=DatasetResponse)
def get_dataset(dataset_id: int, session: dict = Depends(get_session)):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM datasets WHERE id=%s", (dataset_id,)).fetchone()
    if row is None or not dataset_service.can_use(row, session.get("tapis_username")):
        raise HTTPException(404, "Dataset not found")
    return _row_to_dataset(row)


# A plain `def` (not async) so FastAPI runs it in a worker thread: extracting
# a multi-GB archive would otherwise block every other request meanwhile.
@router.post(
    "/datasets/upload",
    response_model=DatasetResponse,
    description="Upload a zip/tar archive in ImageFolder format, at most "
                f"{dataset_service.format_size(settings.max_upload_bytes)}.",
)
def upload_dataset(
    name: str = Query(..., description="Dataset display name"),
    file: UploadFile = File(...),
    session: dict = Depends(get_session),
):
    # Uploads are attributed, like jobs: only the uploader may delete one.
    uploaded_by = session.get("tapis_username")
    if not uploaded_by:
        raise HTTPException(
            403, "Log in through the ICICLE portal before uploading a dataset."
        )

    if not file.filename:
        raise HTTPException(400, "Filename is required")

    if not file.filename.endswith(dataset_service.ARCHIVE_TYPES):
        raise HTTPException(400, f"Unsupported format. Allowed: "
                                 f"{', '.join(dataset_service.ARCHIVE_TYPES)}")

    if file.size is not None and file.size > settings.max_upload_bytes:
        raise HTTPException(413, "File too large: the maximum upload size is "
                                 f"{dataset_service.format_size(settings.max_upload_bytes)}.")

    try:
        info = dataset_service.add_uploaded_dataset(
            name, file.file, file.filename, uploaded_by=uploaded_by)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {e}")

    with get_db() as conn:
        row = conn.execute("SELECT * FROM datasets WHERE id=%s", (info["id"],)).fetchone()
    return _row_to_dataset(row)


@router.delete("/datasets/{dataset_id}", response_model=MessageResponse)
def delete_dataset(dataset_id: int, session: dict = Depends(get_session)):
    """Delete an uploaded dataset. Only its uploader may, or an admin
    (settings.admin_users), and only while no job has used it: jobs keep a
    reference to their dataset, so deleting it would mean erasing those
    jobs' records.

    Its copy on Expanse is deleted by a job run with the caller's own login,
    which can take a few minutes; the dataset is shown as 'deleting' and
    disappears once that job finishes."""
    username = session.get("tapis_username")
    delete_job = None
    with get_db() as conn:
        # Locked until the delete commits, so no job can start using it meanwhile.
        row = conn.execute(
            "SELECT * FROM datasets WHERE id=%s FOR UPDATE", (dataset_id,)
        ).fetchone()
        if row is None or not dataset_service.can_use(row, username):
            raise HTTPException(404, "Dataset not found")
        if not row["is_upload"]:
            raise HTTPException(400, "Cannot delete pre-configured datasets")
        busy = {"checking": "checked", "copying": "copied to Expanse",
                "deleting": "deleted from Expanse"}
        if row["copy_status"] in busy:
            raise HTTPException(
                409, f"This dataset is still being {busy[row['copy_status']]}. "
                     f"Try again once that has finished."
            )
        used = conn.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE dataset_id=%s", (dataset_id,)
        ).fetchone()["n"]
        if used:
            raise HTTPException(
                409, f"This dataset can't be deleted: {used} job(s) used it, and "
                     f"deleting it would erase their records."
            )

        if dataset_service.has_expanse_copy(row):
            token = user_tokens.current(username)
            if not token:
                raise HTTPException(401, "Your login is about to expire. Reload the page and try again.")
            try:
                delete_job = dataset_service.submit_delete(row, token)
            except Exception as e:  # noqa: BLE001
                raise HTTPException(
                    502, f"Couldn't start deleting the dataset's copy on Expanse, "
                         f"so nothing was deleted: {e}"
                )
            conn.execute(
                "UPDATE datasets SET copy_status='deleting', copy_error=NULL, is_ready=0, "
                "remote_job_uuid=%s WHERE id=%s", (delete_job, dataset_id),
            )
        else:
            resumable_uploads.cancel(dataset_id)  # if it is still arriving
            dataset_service.delete_local_files(row["id"], row["name"])
            conn.execute("DELETE FROM datasets WHERE id=%s", (dataset_id,))
    if delete_job:
        dataset_service.follow_delete(dataset_id, delete_job, username)
        return MessageResponse(message=f"Deleting dataset {dataset_id} from Expanse. "
                                       f"It disappears from the list once that's done.")
    return MessageResponse(message=f"Dataset {dataset_id} deleted")


# ── search stub (full impl in Phase 3) ─────────────────────────────────────

@router.post("/search", response_model=SearchResponse)
def run_search(req: SearchRequest, session: dict = Depends(get_session)):
    """Run architecture search under a latency/FLOPs budget."""
    from ..services.search_service import run_search as do_search

    # Lookup dataset info
    with get_db() as conn:
        ds = _dataset_for_jobs(conn, req.dataset_id, session.get("tapis_username"))
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
def create_job(req: JobCreateRequest, session: dict = Depends(get_session)):
    """Create and start a training job.

    Requires the caller's Tapis login (the token the ICICLE pod passes in):
    the training job runs on Expanse with it, and the submitter is recorded.
    """
    from ..services.job_manager import job_manager

    # The user of the validated Tapis token, if the request sent one
    # (see deps.get_session).
    submitted_by = session.get("tapis_username")
    if not submitted_by:
        raise HTTPException(
            403, "Log in through the ICICLE portal before submitting a job."
        )

    with get_db() as conn:
        _dataset_for_jobs(conn, req.dataset_id, submitted_by)
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
            submitted_by=submitted_by,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    return job


@router.get("/jobs", response_model=JobListResponse)
def list_jobs(
    status: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    session: dict = Depends(get_session),
):
    tapis_username = session.get("tapis_username")
    with get_db() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE status=%s AND submitted_by=%s "
                "ORDER BY id DESC LIMIT %s OFFSET %s",
                (status, tapis_username, limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE submitted_by=%s ORDER BY id DESC LIMIT %s OFFSET %s",
                (tapis_username, limit, offset),
            ).fetchall()
    return JobListResponse(jobs=[_row_to_job(r) for r in rows])


def _owned_job_row(conn, job_id: int, session: dict):
    """Fetch a job row, but only if the current session actually owns it.

    404 rather than 403 for someone else's job -- don't reveal that a given
    id belongs to another user.
    """
    row = conn.execute("SELECT * FROM jobs WHERE id=%s", (job_id,)).fetchone()
    if row is None or not session.get("tapis_username") \
            or row["submitted_by"] != session["tapis_username"]:
        raise HTTPException(404, "Job not found")
    return row


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: int, session: dict = Depends(get_session)):
    with get_db() as conn:
        row = _owned_job_row(conn, job_id, session)
    return _row_to_job(row)


@router.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
def get_job_status(job_id: int, session: dict = Depends(get_session)):
    with get_db() as conn:
        row = _owned_job_row(conn, job_id, session)
    return JobStatusResponse(
        id=row["id"], status=row["status"],
        best_val_acc=row["best_val_acc"], error_message=row["error_message"],
    )


@router.post("/jobs/{job_id}/cancel", response_model=MessageResponse)
def cancel_job(job_id: int, session: dict = Depends(get_session)):
    from ..services.job_manager import job_manager

    with get_db() as conn:
        _owned_job_row(conn, job_id, session)

    success = job_manager.cancel(job_id)
    if success:
        return MessageResponse(message=f"Job {job_id} cancelled")
    else:
        raise HTTPException(400, "Job cannot be cancelled in its current state")


# ── download ───────────────────────────────────────────────────────────────

@router.get("/download/{job_id}")
def download_model(job_id: int, session: dict = Depends(get_session)):
    with get_db() as conn:
        row = _owned_job_row(conn, job_id, session)
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
