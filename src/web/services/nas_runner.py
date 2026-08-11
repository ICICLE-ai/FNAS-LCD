"""NAS pipeline runner — executes a job end-to-end (search → train → export).

Called from a background thread by job_manager. Each step updates the job
row in Postgres so the web UI can poll for progress.
"""

from __future__ import annotations

import datetime
import sys
import threading
import time
from pathlib import Path

# ── path setup (must come before local imports that touch PlainNet) ────────
_PROJ = Path(__file__).resolve().parents[2]
for _p in (str(_PROJ), str(_PROJ / "model"), str(_PROJ / "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.auto_nas import _write_structure  # noqa: E402

from ..config import settings
from ..database import get_connection


def _export_fresh_model(structure_str: str, num_classes: int, output_path: Path,
                         input_image_size: int = 224) -> None:
    """Build the searched architecture with freshly-initialized weights and
    export it directly to TorchScript — no training (demo mode)."""
    import torch
    from Masternet import MasterNet  # noqa: E402

    model = MasterNet(plainnet_struct=structure_str, num_classes=num_classes, no_create=False)
    model.init_parameters(method="xavier")
    model.eval()

    dummy = torch.randn(1, 3, input_image_size, input_image_size)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output_path))


def _update_status(job_id: int, **kwargs):
    """Update job row columns. Thread-safe: each call opens its own connection."""
    if not kwargs:
        return

    set_parts = []
    values = []
    for k, v in kwargs.items():
        set_parts.append(f"{k}=%s")
        values.append(v)

    values.append(job_id)

    conn = get_connection()
    try:
        conn.execute(
            f"UPDATE jobs SET {', '.join(set_parts)} WHERE id=%s",
            values,
        )
        conn.commit()
    finally:
        conn.close()


def _load_job(job_id: int) -> dict | None:
    """Load a single job row as a dict."""
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM jobs WHERE id=%s", (job_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def execute_job(job_id: int, cancel_event: threading.Event):
    """Run all steps: search, train, export.

    Args:
        job_id: Job row ID.
        cancel_event: Set to signal cancellation.
    """
    now = datetime.datetime.now().isoformat()
    _update_status(job_id, status="searching", started_at=now)

    job = _load_job(job_id)
    if job is None:
        return

    # ── Step 1: Search (if no structure provided) ──────────────────────
    if not job["structure_str"]:
        if cancel_event.is_set():
            return

        try:
            from .search_service import run_search

            # Determine constraint column from device_configs
            conn = get_connection()
            dev_row = conn.execute(
                "SELECT * FROM device_configs WHERE name=%s",
                (job["device_type"],),
            ).fetchone()
            ds_row = conn.execute(
                "SELECT * FROM datasets WHERE id=%s", (job["dataset_id"],)
            ).fetchone()
            conn.close()

            if dev_row is None:
                raise RuntimeError(f"Unknown device: {job['device_type']}")
            if ds_row is None:
                raise RuntimeError(f"Dataset not found: {job['dataset_id']}")

            result = run_search(
                lookup_csv=settings.lookup_csv,
                constraint_column=dev_row["constraint_column"],
                num_classes=ds_row["num_classes"],
                budget=job["budget"],
            )

            structure_str = result["structure_str"]
            _update_status(
                job_id,
                structure_str=structure_str,
                search_score=result["score"],
                search_constraint=result["constraint"],
                search_flops=result["flops"],
                search_params=result["params"],
            )
            job["structure_str"] = structure_str

        except Exception as e:
            _update_status(
                job_id,
                status="failed",
                error_message=f"Search failed: {e}",
                finished_at=datetime.datetime.now().isoformat(),
            )
            return

    # ── Step 2: Train (demo mode — shown for UX continuity, does no work) ──
    if cancel_event.is_set():
        return

    _update_status(job_id, status="training")
    time.sleep(1.5)  # otherwise this status is too brief to ever be seen

    # ── Step 3: Export ───────────────────────────────────────────────────
    if cancel_event.is_set():
        return

    _update_status(job_id, status="exporting")

    # Refresh job after status update
    job = _load_job(job_id)
    save_dir = Path(job["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    conn = get_connection()
    ds_row = conn.execute(
        "SELECT * FROM datasets WHERE id=%s", (job["dataset_id"],)
    ).fetchone()
    conn.close()
    if ds_row is None:
        raise RuntimeError(f"Dataset not found: {job['dataset_id']}")

    num_classes = ds_row["num_classes"]
    structure_str = job["structure_str"]

    # Keep the structure on disk for reference alongside the export.
    _write_structure(save_dir, structure_str)

    # Export locally first (models_dir is a transient staging spot), then
    # upload to object storage — the DB's output_path stores the object key,
    # not a local path, once storage-backed download is live everywhere.
    local_output_path = settings.models_dir / f"job_{job_id}.pt"
    object_key = f"models/job_{job_id}.pt"

    try:
        _export_fresh_model(
            structure_str=structure_str,
            num_classes=num_classes,
            output_path=local_output_path,
        )
        from .storage_service import upload_object
        upload_object(local_output_path, object_key)
        local_output_path.unlink(missing_ok=True)
    except Exception as e:
        _update_status(
            job_id,
            status="failed",
            error_message=f"Export failed: {e}",
            finished_at=datetime.datetime.now().isoformat(),
        )
        return

    # ── Done ──────────────────────────────────────────────────────────
    _update_status(
        job_id,
        status="completed",
        output_path=object_key,
        finished_at=datetime.datetime.now().isoformat(),
    )
