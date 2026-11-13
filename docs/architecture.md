# Target architecture

This describes how FNAS-LCD is put together: the web service, how it talks
to Tapis for remote training, and the identity model that governs who can
submit and see jobs. For how to run it, see the main `README.md`; for what
ICICLE's platform needs to host it, see `deployment.md` in this folder.

## Components

```
Browser ── FastAPI app (src/web/) ── Postgres (job/dataset metadata)
              │                   └─ S3-compatible storage (exported models)
              └── Tapis API ── SDSC Expanse GPU job
```

- **FastAPI app** (`src/web/`) — server-rendered Jinja2 templates (no
  frontend build step) plus a JSON API under `/api`. Runs as a single
  process; background threads track jobs submitted to Tapis.
- **Postgres** — job records, dataset registrations and device configs.
  No credentials, training data or model weights live here.
- **Object storage** (MinIO locally; any S3-compatible endpoint in
  production) — holds exported model files once a job completes.
- **Tapis** — the actual job execution. The app never runs training
  itself; it submits a job to Tapis, which runs it on a GPU node of SDSC
  Expanse and reports status back. Uploaded datasets reach Expanse the same
  way, as jobs.

## Identity

Every Tapis call runs as a specific user's account. FNAS-LCD makes each call
with the logged-in user's own Tapis access token, and holds no Tapis
credentials of its own.

- **Who the user is.** The hosting platform logs users in and passes their
  token to the app as an `X-Tapis-Token` cookie; the ICICLE pod does this.
  The app checks the token with Tapis (`/v3/oauth2/userinfo`) and uses the
  username Tapis returns (`src/web/deps.py`). `REQUIRE_TAPIS_TOKEN=1` makes
  a valid token mandatory for every route except `/health`, `/static` and
  the upload links. Without a token, a browser gets an anonymous session
  that can look around but not upload datasets or submit jobs.
- **Background work.** Each user's newest token is kept in memory only
  (`src/web/services/user_tokens.py`), and the threads that track the
  user's jobs use it. When it has expired, tracking pauses until the user
  opens FNAS-LCD again; the job itself keeps running on Expanse meanwhile.
- **The apps.** Jobs run through two Tapis apps (`tapis/`), owned by one
  ICICLE account and shared with every ICICLE user. Each app fixes the
  system, job folders, queue, allocation and container options. Because the
  folders come from the app, Tapis lets each job use the app owner's
  permission on them, so users need no folder access of their own. On
  Expanse, every job runs as the shared Linux account `harvest` and is
  charged to the `uot260` allocation.

## Job lifecycle

1. **Search** (`src/enumeration/`, `src/scoring/`) runs locally — fast,
   no GPU needed.
2. **Train.** `src/web/services/nas_runner.py` submits a job with the
   user's token via `tapis_service.submit_training()`, then polls
   (`TAPIS_POLL_SECONDS`) until it reaches a terminal state. This is a
   background thread per job; on restart, `job_manager.resume_interrupted()`
   re-attaches to anything still `training`/`exporting` in Postgres, and
   carries on once the job's owner has opened FNAS-LCD again.
3. **Export.** The finished model (and metrics, if the training container
   emits them) is downloaded from Tapis and uploaded to object storage;
   the job row is marked `completed`.

Every job row records `submitted_by` — the Tapis username responsible for
it — and every job-reading/acting endpoint (list, detail, status, cancel,
download) checks that the caller owns the job before doing anything,
returning 404 rather than 403 for someone else's job so as not to reveal
that a given id belongs to another user.

## Datasets on Expanse

- **Upload.** The service checks and repacks an uploaded archive, then
  submits an upload job with the uploader's token. The job's input is a
  temporary link that the pod serves without login
  (`/upload-links/<secret>`, `src/web/services/upload_links.py`). Tapis
  downloads the archive from it into the job's folder before the job is
  queued, and the service then revokes the link. The job unpacks the
  dataset in its own folder and records who uploaded it.
- **Training** jobs see every upload, read-only, and are told which one to
  use.
- **Delete.** Deleting an uploaded dataset runs the upload app's delete mode
  with the uploader's token. It deletes only a dataset that the same user
  uploaded.
- **Expiry.** The datasets live in Expanse's scratch space, which SDSC
  purges 90 days after files are written. The site shows each upload's
  expiry date and stops offering it after that.

## What's deliberately not solved here

- **Data on Expanse isn't private.** Every job runs as the same `harvest`
  account, and `expanse-tapis-static` is public, so other ICICLE users can
  read the uploaded datasets and trained models there. The upload and
  search pages say so.
- **Whether this needs to be a continuously-running service at all is an
  open question**, not a settled design. The background-thread/resume
  approach exists purely to promptly react when a job finishes (download
  the model, flip status) — the training job itself keeps running on the
  execution system regardless of whether this process is up. A
  check-on-read design (only reconcile a job's status when someone asks)
  would avoid needing continuous uptime, at the cost of not reacting to
  completion until someone actually looks.
