"""Job manager — background thread lifecycle and Postgres updates."""

from __future__ import annotations

import datetime
import threading
import time
from pathlib import Path
from typing import Optional

from ..config import settings
from ..database import get_db
from ..models import JobResponse


class JobManager:
    """Manages background training jobs with Postgres persistence."""

    def __init__(self):
        self._cancel_events: dict[int, threading.Event] = {}
        self._threads: dict[int, threading.Thread] = {}
        self._lock = threading.Lock()

    # ── public API ─────────────────────────────────────────────────────

    def create_and_start(
        self,
        dataset_id: int,
        device_type: str,
        budget: float,
        metric: str = "analytic_ssr_exp_proj_dw_mean",
        aggregate: str = "sum",
        epochs: int = 100,
        warmup: int = 5,
        batch_size: int = 64,
        workers: int = 4,
        lr: float = 0.1,
        gpu: int = 0,
        max_iters: Optional[int] = None,
        structure_str: Optional[str] = None,
        submitted_by: Optional[str] = None,
    ) -> JobResponse:
        """Create a job row and launch it in a background thread.

        `submitted_by` is the Tapis username of the logged-in user (see
        `deps.get_session`), persisted on the row. The training job runs on
        Expanse with that user's own login -- see
        `nas_runner._run_remote_training`.
        """
        import datetime as dt

        save_dir = settings.checkpoints_dir / f"job_{0}"  # placeholder

        with get_db() as conn:
            cursor = conn.execute(
                """INSERT INTO jobs
                   (dataset_id, device_type, budget, metric, aggregate,
                    epochs, warmup, batch_size, workers, lr, gpu, max_iters,
                    structure_str, status, save_dir, submitted_by)
                   VALUES (%s,%s,%s,%s,%s, %s,%s,%s,%s,%s,%s,%s, %s, 'pending', '', %s)
                   RETURNING id""",
                (dataset_id, device_type, budget, metric, aggregate,
                 epochs, warmup, batch_size, workers, lr, gpu, max_iters,
                 structure_str, submitted_by),
            )
            job_id = cursor.fetchone()["id"]

            # Now set proper save_dir with the real job_id
            save_dir = settings.checkpoints_dir / f"job_{job_id}"
            save_dir.mkdir(parents=True, exist_ok=True)
            conn.execute(
                "UPDATE jobs SET save_dir=%s WHERE id=%s",
                (str(save_dir), job_id),
            )

        cancel_event = threading.Event()
        with self._lock:
            self._cancel_events[job_id] = cancel_event

        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, cancel_event),
            daemon=True,
            name=f"job-{job_id}",
        )
        thread.start()

        with self._lock:
            self._threads[job_id] = thread

        with get_db() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=%s", (job_id,)).fetchone()

        from ..routers.api import _row_to_job
        return _row_to_job(row)

    def cancel(self, job_id: int) -> bool:
        """Request cancellation. Returns True if the job was cancellable."""
        with self._lock:
            evt = self._cancel_events.get(job_id)
        if evt is None:
            return False

        # Check current status
        with get_db() as conn:
            row = conn.execute(
                "SELECT status FROM jobs WHERE id=%s",
                (job_id,),
            ).fetchone()
        if row is None:
            return False

        # 'training' is cancellable too: with remote execution it can sit in a
        # scheduler queue or run for hours, and abandoning it silently would
        # keep consuming the compute allocation.
        cancellable = {"pending", "searching", "training"}
        if row["status"] not in cancellable:
            return False

        # The job's thread notices at once and stops its job on Expanse too
        # (nas_runner._run_remote_training).
        evt.set()
        with get_db() as conn:
            conn.execute(
                "UPDATE jobs SET status='cancelled', finished_at=%s WHERE id=%s",
                (datetime.datetime.now().isoformat(), job_id),
            )
        return True

    # ── internal job runner ────────────────────────────────────────────

    def resume_interrupted(self) -> int:
        """Re-attach to jobs that were still running when the service stopped.

        A restart kills the polling threads but not the remote training jobs
        they were watching. Their UUIDs are in Postgres, so pick them back up
        rather than leaving them to run to completion unobserved. A restart
        also forgets every user's token, so each waits for its owner's next
        visit (see user_tokens). Returns the number of jobs resumed.
        """
        with get_db() as conn:
            rows = conn.execute(
                "SELECT id FROM jobs WHERE status IN ('training', 'exporting') "
                "AND tapis_job_uuid IS NOT NULL"
            ).fetchall()

        resumed = 0
        for row in rows:
            job_id = row["id"]
            with self._lock:
                if job_id in self._threads:
                    continue
                cancel_event = threading.Event()
                self._cancel_events[job_id] = cancel_event

            thread = threading.Thread(
                target=self._run_job,
                args=(job_id, cancel_event),
                daemon=True,
                name=f"job-{job_id}-resumed",
            )
            thread.start()
            with self._lock:
                self._threads[job_id] = thread
            resumed += 1
        return resumed

    def _run_job(self, job_id: int, cancel_event: threading.Event):
        """Background job worker. Steps: search → train → export."""
        from .nas_runner import execute_job

        try:
            execute_job(job_id, cancel_event)
        except Exception:
            # Error already recorded by execute_job
            pass
        finally:
            with self._lock:
                self._cancel_events.pop(job_id, None)
                self._threads.pop(job_id, None)


# Singleton
job_manager = JobManager()
