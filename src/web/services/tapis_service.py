"""Tapis calls: training jobs, dataset copies, and their results.

Training does not run in this container. The service searches for an
architecture, submits a Tapis job that trains it on SDSC Expanse, polls until
the job finishes, then downloads the resulting model. Uploaded datasets reach
Expanse the same way, as jobs. This module owns every Tapis interaction;
nas_runner and dataset_service drive it.

Every call is made with the user's own login: the access token passed in,
which the ICICLE pod gives this service (see user_tokens). This module holds
no credentials of its own.

The jobs run through two apps in tapis/, owned by one ICICLE account and
shared with all ICICLE users. Each app fixes where and how its jobs run --
system, folders, allocation, queue, image -- so a request only names the app
and gives the job's own arguments, and Tapis lends the app owner's folder
permissions to whoever runs it.
"""

from __future__ import annotations

import hashlib
import re
import threading
from typing import Callable, Optional

from ..config import settings
from . import user_tokens

# Tapis job states that mean "no longer progressing".
TERMINAL_STATES = {"FINISHED", "FAILED", "CANCELLED"}
# States before Tapis has copied a job's input files into the job's folder.
INPUT_STATES = {"PENDING", "PROCESSING_INPUTS", "STAGING_INPUTS"}

# Files a finished training job leaves in its output folder.
MODEL_FILE = "model.pt"                      # TorchScript, when the job exports it
CHECKPOINT_FILE = "best-params_rank0.pth"    # always present
METRICS_FILE = "best_metrics.json"
LOG_FILE = "tapisjob.out"

# Where the upload app keeps each upload: a folder per upload job, named after
# the job, holding the unpacked dataset (tapis/fnas-lcd-expanse-upload-*.json).
# On expanse-tapis-static, Tapis paths start at harvest's scratch space.
EXPANSE_ROOT = "/expanse/lustre/scratch/harvest/temp_project"
UPLOADS_DIR = f"{EXPANSE_ROOT}/jobs/fnas-lcd/uploads"
_JOB_UUID = r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}-[0-9]{3}"
_DATASET_PATH = re.compile(rf"^{re.escape(UPLOADS_DIR)}/({_JOB_UUID})/dataset$")


class TapisNotConfigured(RuntimeError):
    """Raised when remote training is requested but cannot be done."""


def dataset_path(upload_job_uuid: str) -> str:
    """Where the upload job with this id left its dataset on Expanse."""
    return f"{UPLOADS_DIR}/{upload_job_uuid}/dataset"


def dataset_job_uuid(remote_path: Optional[str]) -> Optional[str]:
    """The id of the upload job that holds a dataset on Expanse, or None if
    `remote_path` isn't such a dataset (e.g. a path on another system)."""
    match = _DATASET_PATH.match(remote_path or "")
    return match.group(1) if match else None


def get_userinfo(access_token: str) -> str:
    """Return the Tapis username an access token belongs to."""
    import requests

    resp = requests.get(
        f"{settings.tapis_base_url}/v3/oauth2/userinfo",
        headers={"X-Tapis-Token": access_token},
        timeout=30,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Tapis userinfo request failed "
                           f"({resp.status_code}): {resp.text[:300]}")
    return resp.json()["result"]["username"]


# One tapipy client per token (keyed by its hash). Building one is slow-ish and, without a tenant
# id, asks the Tenants API for every tenant first; tokens change only every
# few hours per user, so a small cache saves both.
_clients: dict[str, object] = {}
_clients_lock = threading.Lock()


def _call(fn_name: str, token: str, **kwargs):
    """Invoke one Jobs API method with the given user's access token."""
    from tapipy.tapis import Tapis

    if not token:
        raise TapisNotConfigured("no Tapis login to act with")
    key = hashlib.sha256(token.encode()).hexdigest()
    with _clients_lock:
        client = _clients.get(key)
    if client is None:
        client = Tapis(base_url=settings.tapis_base_url,
                       tenant_id=user_tokens.token_claims(token).get("tapis/tenant_id"),
                       access_token=token)
        with _clients_lock:
            if len(_clients) >= 64:
                _clients.clear()
            _clients[key] = client
    return getattr(client.jobs, fn_name)(**kwargs)


# ── submit ─────────────────────────────────────────────────────────────────

def build_job_body(
    *,
    name: str,
    structure_str: str,
    num_classes: int,
    remote_dataset_path: str,
    epochs: int,
    max_iters: Optional[int] = None,
    batch_size: Optional[int] = None,
    workers: Optional[int] = None,
    lr: Optional[float] = None,
    warmup: Optional[int] = None,
    max_minutes: Optional[int] = None,
    submitted_by: Optional[str] = None,
) -> dict:
    """Assemble a training job request.

    Several details here are load-bearing and were established by testing
    against the live system:

    * The architecture must be **single-quoted** inside the arg string. Tapis
      does not quote appArgs (applications define their own conventions), and
      the structure string contains parentheses, which are a bash syntax error
      unquoted in the generated tapisjob.sh.
    * max_iters truncates each *epoch*, it does not bound the run, so demo
      mode must also force epochs=1 — mirroring auto_nas._train_via_timm_trainer.
    * warmup must then be clamped below the epoch count. The cosine schedule
      asserts stop_instances > warmup_instances, so warming up for longer than
      the whole run aborts training partway through the first epoch.

    The job's folders, allocation, queue and container settings are fixed in
    the app, and the job script in the image (src/training/expanse_job.py)
    accepts only these arguments.
    """
    upload_uuid = dataset_job_uuid(remote_dataset_path)
    if upload_uuid is None:
        raise TapisNotConfigured(
            f"the dataset isn't on Expanse ({remote_dataset_path}), so it can't be trained there")
    if max_iters is not None:
        epochs = 1
    if warmup is not None:
        warmup = min(warmup, max(epochs - 1, 0))

    app_args = [
        {"name": "dataset", "arg": f"--dataset {upload_uuid}"},
        {"name": "plainnet_struct", "arg": f"--plainnet_struct '{structure_str}'"},
        {"name": "num_classes", "arg": f"--num_classes {num_classes}"},
        {"name": "epochs", "arg": f"--epochs {epochs}"},
    ]
    if max_iters is not None:
        app_args.append({"name": "max_iters", "arg": f"--max_iters {max_iters}"})
    if batch_size is not None:
        app_args.append({"name": "batch_per_gpu", "arg": f"--batch_size_per_gpu {batch_size}"})
    if workers is not None:
        app_args.append({"name": "workers", "arg": f"--workers_per_gpu {workers}"})
    if lr is not None:
        app_args.append({"name": "lr", "arg": f"--lr_per_256 {lr}"})
    if warmup is not None:
        app_args.append({"name": "warmup", "arg": f"--warmup {warmup}"})

    return {
        "name": name,
        "appId": settings.tapis_app_id,
        "appVersion": settings.tapis_app_version,
        # Attribution independent of our own database -- so a job's
        # submitter is visible from Tapis's own records too, not only ours.
        "description": f"FNAS-LCD training ({num_classes} classes)"
                       + (f" -- submitted via fnas-lcd by {submitted_by}" if submitted_by else ""),
        "maxMinutes": max_minutes or settings.tapis_max_minutes,
        "parameterSet": {"appArgs": app_args},
    }


def submit_training(token: str, **kwargs) -> str:
    """Submit a training job (see build_job_body) as the user whose token
    this is. Returns the Tapis job UUID."""
    return _call("submitJob", token, **build_job_body(**kwargs)).uuid


def submit_upload(token: str, *, source_url: str, name: str,
                  uploaded_by: Optional[str] = None) -> str:
    """Submit the job that copies one dataset archive onto Expanse: Tapis
    downloads it from `source_url` into the job's folder, and the job unpacks
    it there. Returns the job UUID, which then names the dataset's folder."""
    return _call("submitJob", token, **{
        "name": name,
        "appId": settings.tapis_upload_app_id,
        "appVersion": settings.tapis_upload_app_version,
        "description": "FNAS-LCD dataset upload"
                       + (f" -- uploaded via fnas-lcd by {uploaded_by}" if uploaded_by else ""),
        "fileInputs": [{"name": "archive", "sourceUrl": source_url}],
    }).uuid


def submit_delete(token: str, *, upload_job_uuid: str, name: str) -> str:
    """Submit the job that deletes one uploaded dataset from Expanse. The job
    deletes only a dataset that the same Tapis user uploaded. Returns the job
    UUID."""
    if not re.fullmatch(_JOB_UUID, upload_job_uuid):
        raise ValueError(f"not an upload job id: {upload_job_uuid!r}")
    return _call("submitJob", token, **{
        "name": name,
        "appId": settings.tapis_upload_app_id,
        "appVersion": settings.tapis_upload_app_version,
        "description": f"FNAS-LCD dataset delete ({upload_job_uuid})",
        "parameterSet": {"appArgs": [{"name": "mode", "arg": f"delete {upload_job_uuid}"}]},
    }).uuid


# ── monitor ────────────────────────────────────────────────────────────────

def get_status(token: str, uuid: str) -> str:
    return _call("getJobStatus", token, jobUuid=uuid).status


def get_last_message(token: str, uuid: str) -> str:
    try:
        return _call("getJob", token, jobUuid=uuid).lastMessage or ""
    except Exception:  # noqa: BLE001 - diagnostics only
        return ""


def cancel(token: str, uuid: str) -> None:
    _call("cancelJob", token, jobUuid=uuid)


# Consecutive failed status checks after which a job counts as lost.
_MAX_STATUS_FAILURES = 5


def watch(uuid: str, username: str, done: Callable[[str], bool] = TERMINAL_STATES.__contains__,
          stop: Optional[threading.Event] = None) -> Optional[str]:
    """Check a job's state with its owner's newest token until `done(state)`,
    and return that state. Queue waits can be long, so this has no timeout of
    its own -- the job's own time limit bounds it.

    When the owner's token has expired, this waits for them to open
    FNAS-LCD again (user_tokens.wait_for). Returns None if `stop` is set
    first. Raises if Tapis can't be reached several times in a row.
    """
    stop = stop or threading.Event()
    failures = 0
    while True:
        token = user_tokens.wait_for(username, stop)
        if token is None:
            return None
        try:
            state = get_status(token, uuid)
            failures = 0
        except Exception:  # noqa: BLE001
            failures += 1
            if failures >= _MAX_STATUS_FAILURES:
                raise
            state = None
        if state is not None and done(state):
            return state
        if stop.wait(settings.tapis_poll_seconds):
            return None


# ── outputs ────────────────────────────────────────────────────────────────

def list_outputs(token: str, uuid: str) -> list[str]:
    """Names of files in the job's output directory (empty on failure)."""
    try:
        # The API path is /output/list/{outputPath} and outputPath must end
        # with '/'; it is relative to the job's output directory.
        listing = _call("getJobOutputList", token, jobUuid=uuid, outputPath="/")
    except Exception:  # noqa: BLE001
        return []
    return [getattr(f, "name", str(f)) for f in listing]


def download_output(token: str, uuid: str, name: str) -> Optional[bytes]:
    """Fetch one output file. Returns None if it is absent or unreadable."""
    try:
        data = _call("getJobOutputDownload", token, jobUuid=uuid, outputPath=name)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(data, str):
        data = data.encode()
    return data


def failure_detail(uuid: str, username: str, state: str, log_chars: int = 2000) -> str:
    """Why a job ended without finishing: Tapis's last message plus the end
    of the job's log, read with its owner's newest token."""
    token = user_tokens.wait_for(username)
    detail = get_last_message(token, uuid) or state
    log = download_output(token, uuid, LOG_FILE)
    if log:
        detail += "\n" + log.decode(errors="replace")[-log_chars:]
    return detail
