# Target architecture

This describes how FNAS-LCD is put together: the web service, how it talks
to Tapis for remote training, and the identity model that governs who can
submit and see jobs. For how to run it, see the main `README.md`; for what
ICICLE's platform needs to host it, see `deployment.md` in this folder.

## Components

```
Browser ── FastAPI app (src/web/) ── Postgres (job/dataset metadata)
              │                   └─ S3-compatible storage (exported models)
              └── Tapis API ── Pitzer (OSC) / Vista (TACC) GPU job
```

- **FastAPI app** (`src/web/`) — server-rendered Jinja2 templates (no
  frontend build step) plus a JSON API under `/api`. Runs as a single
  process; background threads track jobs submitted to Tapis.
- **Postgres** — job records, dataset registrations, device configs,
  sessions, and stored Tapis credentials (see below). No training data or
  model weights live here.
- **Object storage** (MinIO locally; any S3-compatible endpoint in
  production) — holds exported model files once a job completes.
- **Tapis** — the actual job execution. The app never runs training
  itself; it submits a job description to Tapis, which runs it on a real
  GPU system (currently OSC Pitzer; TACC Vista is planned) and reports
  status back.

## Why identity is a first-class concern

Tapis has no generic "service account" concept — every API call is
authenticated as a specific person's account. That constraint shapes most
of the identity-related code in this service: it isn't optional plumbing,
it's a direct consequence of how the platform this app is built on works.
Three ways this service establishes "whose Tapis account should this call
run as" coexist in `src/web/services/tapis_service.py`:

1. **Shared service credential.** One Tapis identity's OAuth client +
   refresh token, used as a fallback for jobs with no more specific owner
   (e.g. one resumed on restart with no session behind it anymore).
2. **Per-user OAuth.** A user connects their own Tapis account once
   (`authorization_code` grant, `src/web/routers/auth.py`); their jobs then
   run under their own identity. Credentials are stored per user
   (`user_tapis_credentials`) and rotated automatically — Tapis rotates the
   refresh token on every use.
3. **Pass-through token.** If the request already carries a Tapis access
   token (an `X-Tapis-Token` cookie — e.g. injected by a hosting platform
   that already logged the user in), that token is used directly, with no
   redirect needed. Enabling `REQUIRE_TAPIS_TOKEN=1` makes this mandatory
   for every route except `/health` and `/static`, for deployments that
   sit behind such a platform.

All three funnel through one function (`tapis_service._call()`), which
prefers an explicit access token over a stored one when both are present.
See `src/web/deps.py` for how a request's identity is resolved
(`get_session`, `get_tapis_token`, `require_login`).

## Job lifecycle

1. **Search** (`src/enumeration/`, `src/scoring/`) runs locally — fast,
   no GPU needed.
2. **Train.** `src/web/services/nas_runner.py` submits a job via
   `tapis_service.submit_training()`, then polls
   (`TAPIS_POLL_SECONDS`) until it reaches a terminal state. This is a
   background thread per job; on restart, `job_manager.resume_interrupted()`
   re-attaches to anything still `training`/`exporting` in Postgres. A job
   submitted via a pass-through token (point 3 above) cannot be resumed
   this way — that credential only ever lives in memory, never stored.
3. **Export.** The finished model (and metrics, if the training container
   emits them) is downloaded from Tapis and uploaded to object storage;
   the job row is marked `completed`.

Every job row records `submitted_by` — the Tapis username responsible for
it — and every job-reading/acting endpoint (list, detail, status, cancel,
download) checks that the caller owns the job before doing anything,
returning 404 rather than 403 for someone else's job so as not to reveal
that a given id belongs to another user.

## What's deliberately not solved here

- **`TAPIS_WORK_BASE` is one global setting, not per-user.** Every job
  writes into the same configured directory on the execution system,
  regardless of whose Tapis identity submitted it. This works as long as
  everyone submitting jobs shares write access to that directory; it has
  not been tested with a genuinely different second account.
- **Whether this needs to be a continuously-running service at all is an
  open question**, not a settled design. The background-thread/resume
  approach exists purely to promptly react when a job finishes (download
  the model, flip status) — the training job itself keeps running on the
  execution system regardless of whether this process is up. A
  check-on-read design (only reconcile a job's status when someone asks)
  would avoid needing continuous uptime, at the cost of not reacting to
  completion until someone actually looks.
