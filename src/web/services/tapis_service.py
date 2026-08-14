"""Tapis client — submits training jobs to a remote HPC system.

Training does not run in this container. The service searches for an
architecture, submits a Tapis job that trains it on GPU hardware, polls until
the job finishes, then downloads the resulting model. This module owns every
Tapis interaction; nas_runner drives it.

Deliberately thin and module-level, matching storage_service.py: config comes
from settings, no class hierarchy, and each function maps to one Tapis call.
"""

from __future__ import annotations

import threading
from typing import Optional

from ..config import settings

# Tapis job states that mean "no longer progressing".
TERMINAL_STATES = {"FINISHED", "FAILED", "CANCELLED"}

# Files a finished job leaves in its output directory.
MODEL_FILE = "model.pt"                      # TorchScript, when the job exports it
CHECKPOINT_FILE = "best-params_rank0.pth"    # always present
METRICS_FILE = "best_metrics.json"
LOG_FILE = "tapisjob.out"

_client = None
_client_lock = threading.Lock()


class TapisNotConfigured(RuntimeError):
    """Raised when remote training is requested but credentials are missing."""


# ── refresh-token credentials (preferred) ──────────────────────────────────
#
# Tapis v3 has no client_credentials/service-account grant: tokens always
# represent a user. The next best thing is an OAuth client plus a refresh token,
# so the service holds a revocable credential instead of someone's password.
#
# A bare password grant returns only an access token (4h, no refresh token).
# Passing client_id/client_key with the grant is what makes Tapis issue a
# refresh token as well.

_REFRESH_TOKEN_KEY = "tapis_refresh_token"


def _load_stored_refresh_token() -> Optional[str]:
    """Read the current refresh token from the database, if available.

    Falls back to None (not an error) when there is no database — this module is
    also used by standalone scripts that have no service database.
    """
    try:
        from ..database import get_db
        with get_db() as conn:
            row = conn.execute(
                "SELECT value FROM service_credentials WHERE key=%s",
                (_REFRESH_TOKEN_KEY,),
            ).fetchone()
        return row["value"] if row else None
    except Exception:  # noqa: BLE001 - absence of a DB is not fatal here
        return None


def _store_refresh_token(token: str) -> bool:
    """Persist a rotated refresh token. Returns True if it was stored."""
    try:
        from ..database import get_db
        with get_db() as conn:
            conn.execute(
                "INSERT INTO service_credentials (key, value) VALUES (%s, %s) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, "
                "updated_at = to_char(now(), 'YYYY-MM-DD HH24:MI:SS')",
                (_REFRESH_TOKEN_KEY, token),
            )
        return True
    except Exception as e:  # noqa: BLE001
        # Losing a rotated token means the next refresh fails, so make it loud.
        print(f"WARNING: could not persist rotated Tapis refresh token: {e}")
        return False


def _post_token_request(payload: dict) -> dict:
    import requests

    resp = requests.post(f"{settings.tapis_base_url}/v3/oauth2/tokens",
                         json=payload, timeout=30)
    if resp.status_code >= 400:
        raise RuntimeError(f"Tapis token request failed "
                           f"({resp.status_code}): {resp.text[:300]}")
    return resp.json()["result"]


def _unwrap(result: dict, key: str) -> Optional[str]:
    """Pull a token out of Tapis's nested token response shape."""
    node = result.get(key)
    if isinstance(node, dict):
        return node.get(key)
    return node if isinstance(node, str) else None


def bootstrap_refresh_token(username: str, password: str) -> str:
    """Exchange a password for a refresh token, once, and persist it.

    Run this a single time when setting the service up. Afterwards the password
    is not needed and should be removed from the service's environment.
    """
    if not settings.tapis_client_id or not settings.tapis_client_key:
        raise TapisNotConfigured(
            "TAPIS_CLIENT_ID/TAPIS_CLIENT_KEY are required to obtain a refresh "
            "token. Register an OAuth client via POST /v3/oauth2/clients."
        )
    result = _post_token_request({
        "grant_type": "password",
        "username": username,
        "password": password,
        "client_id": settings.tapis_client_id,
        "client_key": settings.tapis_client_key,
    })
    refresh = _unwrap(result, "refresh_token")
    if not refresh:
        raise RuntimeError(
            "Tapis did not return a refresh token. This happens when the grant "
            "is made without client credentials."
        )
    _store_refresh_token(refresh)
    return refresh


def _access_token_from_refresh() -> str:
    """Trade the stored refresh token for an access token, persisting rotation."""
    token = _load_stored_refresh_token() or settings.tapis_refresh_token
    if not token:
        raise TapisNotConfigured("no Tapis refresh token available")

    result = _post_token_request({
        "grant_type": "refresh_token",
        "refresh_token": token,
        "client_id": settings.tapis_client_id,
        "client_key": settings.tapis_client_key,
    })
    access = _unwrap(result, "access_token")
    if not access:
        raise RuntimeError(f"no access token in refresh response: {list(result)}")

    # Tapis rotates refresh tokens: the one we just used may now be dead, so the
    # replacement must be persisted or the next refresh will fail.
    rotated = _unwrap(result, "refresh_token")
    if rotated and rotated != token:
        _store_refresh_token(rotated)
    return access


# ── connection ─────────────────────────────────────────────────────────────

def _build_client():
    from tapipy.tapis import Tapis

    # 1. OAuth client + refresh token — no user password held by the service.
    if settings.tapis_client_id and settings.tapis_client_key and (
            _load_stored_refresh_token() or settings.tapis_refresh_token):
        return Tapis(base_url=settings.tapis_base_url,
                     access_token=_access_token_from_refresh())

    # 2. A pre-issued JWT. Expires in ~4h with no way to renew, so this suits
    #    short-lived scripts rather than a long-running service.
    if settings.tapis_jwt:
        return Tapis(base_url=settings.tapis_base_url, access_token=settings.tapis_jwt)

    # 3. Password grant. Works, but means the service stores a user's password;
    #    prefer (1) and run bootstrap_refresh_token() to get there.
    if not settings.tapis_username or not settings.tapis_password:
        raise TapisNotConfigured(
            "Tapis credentials missing. Provide TAPIS_CLIENT_ID + "
            "TAPIS_CLIENT_KEY (+ a bootstrapped refresh token), or TAPIS_JWT, "
            "or TAPIS_USERNAME/TAPIS_PASSWORD. Set TAPIS_ENABLED=0 to run "
            "without remote training."
        )

    t = Tapis(base_url=settings.tapis_base_url,
              username=settings.tapis_username,
              password=settings.tapis_password)
    t.get_tokens()
    return t


def get_client(force_refresh: bool = False):
    """Return a cached Tapis session, creating or refreshing it as needed.

    Access tokens are short-lived (hours), and a job can outlive one, so
    callers re-authenticate on an auth failure rather than assuming the token
    obtained at startup is still valid.
    """
    global _client
    with _client_lock:
        if _client is None or force_refresh:
            _client = _build_client()
        return _client


def _call(fn_name: str, **kwargs):
    """Invoke a jobs API method, retrying once with a fresh token on failure.

    tapipy surfaces auth failures as assorted exception types depending on the
    layer that rejects, so rather than pattern-matching messages we simply
    retry once with a new session; a genuine error fails the same way twice.
    """
    try:
        return getattr(get_client().jobs, fn_name)(**kwargs)
    except Exception:
        return getattr(get_client(force_refresh=True).jobs, fn_name)(**kwargs)


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
    stage_mode: Optional[str] = None,
) -> dict:
    """Assemble a Tapis job request.

    Several details here are load-bearing and were established by testing
    against the live system:

    * The architecture must be **single-quoted** inside the arg string. Tapis
      does not quote appArgs (applications define their own conventions), and
      the structure string contains parentheses, which are a bash syntax error
      unquoted in the generated tapisjob.sh.
    * The exec/archive directories must be overridden. The system's default
      jobWorkingDir is not writable by our effective user, and jobs fail while
      processing inputs.
    * max_iters truncates each *epoch*, it does not bound the run, so demo
      mode must also force epochs=1 — mirroring auto_nas._train_via_timm_trainer.
    * warmup must then be clamped below the epoch count. The cosine schedule
      asserts stop_instances > warmup_instances, so warming up for longer than
      the whole run aborts training partway through the first epoch.
    """
    if max_iters is not None:
        epochs = 1
    if warmup is not None:
        warmup = min(warmup, max(epochs - 1, 0))

    if not settings.tapis_work_base:
        raise TapisNotConfigured(
            "TAPIS_WORK_BASE must be set to a directory writable by the "
            "execution system's effective user (e.g. /fs/ess/<project>/<user>). "
            "Without it, jobs inherit the system's default working directory, "
            "which is typically not writable and fails during input processing."
        )
    work = settings.tapis_work_base.rstrip("/")
    job_dir = f"{work}/tapis-jobs/" + "${JobUUID}"

    app_args = [
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
    if warmup is not None:
        app_args.append({"name": "lr", "arg": (
            f"--lr_per_256 {lr if lr is not None else 0.1} --target_lr_per_256 0.0 "
            f"--lr_mode cosine --warmup {warmup}")})

    # How the dataset reaches the container.
    #
    # in_place: bind-mount it straight off the execution system's filesystem.
    #   Nothing is transferred, so this is unaffected by Tapis's 10,000-file
    #   limit on a single transfer -- real image datasets blow past that (the
    #   OSA dataset is ~107,000 files) and the job fails during staging.
    #
    # copy: let Tapis stage it into the job's input directory. Only viable for
    #   small datasets, and duplicates the data once per job.
    mode = (stage_mode or settings.tapis_stage_mode or "in_place").lower()
    container_args: list[dict] = []
    file_inputs: list[dict] = []

    if mode == "copy":
        file_inputs.append({
            "name": "dataset",
            # rootDir on the execution system is "/", so an absolute path maps
            # directly into a tapis:// URL.
            "sourceUrl": f"tapis://{settings.tapis_system}{remote_dataset_path}",
            "targetPath": "data",
        })
        # The app's own default --data_dir /workspace/data then applies.
    else:
        container_args.append({
            "name": "dataset_bind",
            "arg": f"--bind {remote_dataset_path}:/dataset",
        })
        app_args.append({"name": "data_dir", "arg": "--data_dir /dataset"})

    parameter_set: dict = {"appArgs": app_args}
    if container_args:
        parameter_set["containerArgs"] = container_args

    body = {
        "name": name,
        "appId": settings.tapis_app_id,
        "appVersion": settings.tapis_app_version,
        "description": f"FNAS-LCD training ({num_classes} classes)",
        "maxMinutes": max_minutes or settings.tapis_max_minutes,
        "execSystemInputDir": job_dir,
        "execSystemExecDir": job_dir,
        "execSystemOutputDir": f"{job_dir}/output",
        "archiveSystemId": settings.tapis_system,
        "archiveSystemDir": f"{work}/tapis-archive/" + "${JobUUID}",
        "parameterSet": parameter_set,
    }
    if file_inputs:
        body["fileInputs"] = file_inputs
    return body


def submit_training(**kwargs) -> str:
    """Submit a training job. Returns the Tapis job UUID."""
    body = build_job_body(**kwargs)
    resp = _call("submitJob", **body)
    return resp.uuid


# ── monitor ────────────────────────────────────────────────────────────────

def get_status(uuid: str) -> str:
    return _call("getJobStatus", jobUuid=uuid).status


def get_last_message(uuid: str) -> str:
    try:
        return _call("getJob", jobUuid=uuid).lastMessage or ""
    except Exception:  # noqa: BLE001 - diagnostics only
        return ""


def cancel(uuid: str) -> None:
    _call("cancelJob", jobUuid=uuid)


# ── outputs ────────────────────────────────────────────────────────────────

def list_outputs(uuid: str) -> list[str]:
    """Names of files in the job's output directory (empty on failure)."""
    try:
        # The API path is /output/list/{outputPath} and outputPath must end
        # with '/'; it is relative to the job's output directory.
        listing = _call("getJobOutputList", jobUuid=uuid, outputPath="/")
    except Exception:  # noqa: BLE001
        return []
    return [getattr(f, "name", str(f)) for f in listing]


def download_output(uuid: str, name: str) -> Optional[bytes]:
    """Fetch one output file. Returns None if it is absent or unreadable."""
    try:
        data = _call("getJobOutputDownload", jobUuid=uuid, outputPath=name)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(data, str):
        data = data.encode()
    return data
