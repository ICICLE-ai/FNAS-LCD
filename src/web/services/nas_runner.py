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

from ..config import settings
from ..database import get_connection


def _write_structure(save_dir: Path, structure_str: str) -> Path:
    """Record the architecture next to the exported model, for reference.

    Duplicates pipeline.auto_nas._write_structure rather than importing it:
    auto_nas imports torch at module scope, and pulling that into the web
    application's import graph for a three-line file write is not worth it now
    that training happens remotely. torch is only needed on the legacy export
    fallback below, where it is imported inside the function.
    """
    structure_txt_path = save_dir / "structure.txt"
    structure_txt_path.write_text(structure_str + "\n", encoding="utf-8")
    return structure_txt_path


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


def _export_trained_model(structure_str: str, num_classes: int, checkpoint: Path,
                          output_path: Path, input_image_size: int = 224) -> None:
    """Load trained weights into the searched architecture and TorchScript it.

    Used when the training job returns only a raw checkpoint. Container builds
    that support `--export` emit a ready-made model.pt and this is skipped
    entirely; keeping both paths means the service works against either.
    """
    import torch
    from Masternet import MasterNet  # noqa: E402

    model = MasterNet(plainnet_struct=structure_str, num_classes=num_classes, no_create=False)
    ck = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state_dict = ck.get("state_dict", ck)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        # Not fatal — but it means the checkpoint and the architecture disagree,
        # which would silently yield a partly-random model.
        print(f"WARNING: checkpoint/architecture mismatch: "
              f"{len(missing)} missing, {len(unexpected)} unexpected keys")
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


def _fetch_trained_model(uuid: str, structure_str: str, num_classes: int,
                         save_dir: Path, output_path: Path) -> float | None:
    """Retrieve the trained model from a finished Tapis job.

    Prefers a TorchScript model exported by the training job itself, which
    needs nothing from us. Container builds without that support return only a
    checkpoint, so fall back to rebuilding the architecture and loading the
    weights here.

    Returns the best validation accuracy if the job reported it.
    """
    import json

    from . import tapis_service

    available = tapis_service.list_outputs(uuid)

    # Preferred: the job exported a ready-to-serve model.
    if tapis_service.MODEL_FILE in available:
        data = tapis_service.download_output(uuid, tapis_service.MODEL_FILE)
        if data:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(data)
        else:
            raise RuntimeError(f"could not download {tapis_service.MODEL_FILE}")
    else:
        # Fallback: raw checkpoint -> rebuild + trace locally.
        data = tapis_service.download_output(uuid, tapis_service.CHECKPOINT_FILE)
        if not data:
            raise RuntimeError(
                f"training job {uuid} produced neither {tapis_service.MODEL_FILE} "
                f"nor {tapis_service.CHECKPOINT_FILE}"
            )
        ckpt_path = save_dir / tapis_service.CHECKPOINT_FILE
        ckpt_path.write_bytes(data)
        _export_trained_model(
            structure_str=structure_str,
            num_classes=num_classes,
            checkpoint=ckpt_path,
            output_path=output_path,
        )

    # Metrics are optional: older training containers do not emit them.
    metrics = tapis_service.download_output(uuid, tapis_service.METRICS_FILE)
    if not metrics:
        return None
    try:
        parsed = json.loads(metrics)
    except (ValueError, TypeError):
        return None
    (save_dir / tapis_service.METRICS_FILE).write_bytes(metrics)
    acc = parsed.get("best_acc1")
    return float(acc) if acc is not None else None


def _fail(job_id: int, message: str) -> None:
    _update_status(
        job_id,
        status="failed",
        error_message=message,
        finished_at=datetime.datetime.now().isoformat(),
    )


def _run_remote_training(job_id: int, job: dict, cancel_event: threading.Event) -> str | None:
    """Submit a Tapis training job and wait for it. Returns the job UUID, or
    None if it failed (in which case the job row is already marked failed)."""
    from . import tapis_service

    conn = get_connection()
    try:
        ds_row = conn.execute(
            "SELECT * FROM datasets WHERE id=%s", (job["dataset_id"],)
        ).fetchone()
    finally:
        conn.close()

    if ds_row is None:
        _fail(job_id, f"Dataset not found: {job['dataset_id']}")
        return None

    remote_path = ds_row["remote_path"]
    if not remote_path:
        # Uploaded datasets live only in this container. Submitting anyway
        # would fail later during input staging with a far less obvious error.
        _fail(job_id,
              f"Dataset '{ds_row['name']}' is not available for remote training: "
              f"it has no path on the training system. Only pre-registered "
              f"datasets can be trained remotely.")
        return None

    uuid = job.get("tapis_job_uuid")
    if not uuid:
        try:
            uuid = tapis_service.submit_training(
                name=f"fnas-lcd-job-{job_id}",
                structure_str=job["structure_str"],
                num_classes=ds_row["num_classes"],
                remote_dataset_path=remote_path,
                epochs=job["epochs"],
                max_iters=job["max_iters"],
                batch_size=job["batch_size"],
                workers=job["workers"],
                lr=job["lr"],
                warmup=job["warmup"],
            )
        except Exception as e:  # noqa: BLE001
            _fail(job_id, f"Training submission failed: {e}")
            return None
        _update_status(job_id, tapis_job_uuid=uuid)

    # Poll until the remote job stops progressing. Queue waits can be long, so
    # this deliberately has no timeout of its own -- the job's own walltime
    # bounds it.
    while True:
        if cancel_event.is_set():
            try:
                tapis_service.cancel(uuid)
            except Exception:  # noqa: BLE001
                pass
            return None

        try:
            state = tapis_service.get_status(uuid)
        except Exception as e:  # noqa: BLE001
            _fail(job_id, f"Lost contact with the training job {uuid}: {e}")
            return None

        if state in tapis_service.TERMINAL_STATES:
            break
        time.sleep(settings.tapis_poll_seconds)

    if state != "FINISHED":
        detail = tapis_service.get_last_message(uuid) or state
        log = tapis_service.download_output(uuid, tapis_service.LOG_FILE)
        if log:
            detail += "\n" + log.decode(errors="replace")[-2000:]
        _fail(job_id, f"Training job {state}: {detail}")
        return None

    return uuid


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

    # ── Step 2: Train ──────────────────────────────────────────────────
    # With Tapis enabled this submits a real GPU training job to the remote
    # HPC system and waits for it. Without it, training is a no-op and the
    # export below produces an untrained model (local/demo behaviour).
    if cancel_event.is_set():
        return

    _update_status(job_id, status="training")

    tapis_uuid = None
    if settings.tapis_enabled:
        tapis_uuid = _run_remote_training(job_id, job, cancel_event)
        if tapis_uuid is None:
            return  # failure already recorded
    else:
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

    best_val_acc = None
    try:
        if tapis_uuid is not None:
            best_val_acc = _fetch_trained_model(
                tapis_uuid, structure_str, num_classes, save_dir, local_output_path
            )
        else:
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
        best_val_acc=best_val_acc,
        finished_at=datetime.datetime.now().isoformat(),
    )
