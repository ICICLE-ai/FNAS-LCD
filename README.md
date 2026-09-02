# FNAS-LCD

Fast Neural Architecture Synthesizer for Latency-constrained Deployment: an analytic, zero-shot proxy score
combined with beam search over a precomputed block lookup table selects an architecture under a target latency budget, then trains and exports it. Ships as
a web service (search, live job tracking, model download).

**Tags:** AI4CI <!-- TODO -->

### License

<!-- TODO: license not yet chosen. Badge/link to be filled in once decided. --> 

## References
- [Diátaxis](https://diataxis.fr/) — the documentation framework this README follows
- Precomputed per-block latency/analytic-score lookup table (`data/block_lookup/`), measured on a Jetson GPU target

## Acknowledgements

<!-- Please include other funding sources above this line. -->

*National Science Foundation (NSF) funded AI institute for Intelligent Cyberinfrastructure with Computational Learning in the Environment (ICICLE) (OAC 2112606)*

## Issue reporting

<!-- TODO -->
[GitHub Issues](https://github.com/ICICLE-ai/FNAS-LCD/issues)

---

# Tutorials

## Run your first architecture search (web service, via Docker Compose)

**Prerequisites:** Docker installed and running (`docker --version` works).

1. From the repo root, start the full stack (app + Postgres + object storage):
   ```bash
   docker compose up --build -d
   docker compose ps   # all three services should show Up / healthy
   ```
2. Open `http://localhost:8000` in a browser.
3. If remote training is enabled (`TAPIS_ENABLED=1` — see
   [Configure remote training](#configure-remote-training-tapis) below), go to
   **Account** and connect your Tapis account first; job submission is
   rejected otherwise. With `TAPIS_ENABLED=0` (the default), skip this step.
4. On the **Search** page, pick the bundled toy dataset (100 classes), a target device
   (e.g. "Jetson GPU"), and a latency budget in milliseconds (e.g. `20`), then submit.
5. You'll land back on the dashboard with a "Submitted job #N" confirmation. Click
   through to the job page to watch it move live through
   `searching → training → exporting → completed` (status updates every second).
6. Once completed, click **Download Model (.pt)** to get the exported TorchScript model.

## Run your first architecture search (CLI, no web service)

**Prerequisites:** [`uv`](https://docs.astral.sh/uv/) installed.

```bash
uv sync
uv run python -m src.pipeline.interactive
```
This prompts for a dataset path (defaults to the bundled toy dataset), a latency
budget, epoch count, and output path, then runs search → train → export end to end.

---

# How-To Guides

## Run the web service outside Docker
```bash
uv sync
POSTGRES_HOST=localhost POSTGRES_PORT=5432 ... \
S3_ENDPOINT_URL=http://localhost:9000 ... \
uv run python -m src.web.main --port 8000
```
The app reads all database and object-storage connection info from environment
variables (`POSTGRES_HOST/PORT/USER/PASSWORD/DB`, `S3_ENDPOINT_URL/ACCESS_KEY/SECRET_KEY/BUCKET`)
— point them at any reachable Postgres and S3-compatible endpoint. See `.env.example`
for the full list and the defaults used by `docker-compose.yml`.

## Run a search non-interactively (CLI, scriptable)
```bash
uv run python -m src.pipeline.auto_nas \
    --data-dir /path/to/imagefolder \
    --latency-budget-ms 20 \
    --output model.pt \
    --epochs 100 \
    --gpu 0
```
Dataset must be in **ImageFolder format**: a directory with a `train/` (and optionally
`val/`, `test/`) subfolder, one class per subdirectory.

## Upload your own dataset (web service)
Go to **Datasets → Upload Dataset**, provide a name and a `.zip`/`.tar`/`.tar.gz`/`.tgz`
archive in ImageFolder format (max 10GB). The service extracts it, validates the
`train/` structure, and detects the class count automatically.

## Check or cancel a running job
The **Jobs** page lists every job with live status and filtering by status. A job can
be cancelled from its detail page while still `pending`/`searching`.

## Configure remote training (Tapis)
Training runs as a real GPU job on an HPC system (OSC Pitzer) via
[Tapis](https://tapis-project.org/), not in this container. With
`TAPIS_ENABLED=0` (the default) the service needs no Tapis credentials at
all. To enable it:

1. Set `TAPIS_ENABLED=1`, `TAPIS_CLIENT_ID`/`TAPIS_CLIENT_KEY` (a registered
   Tapis OAuth client), and `TAPIS_WORK_BASE` (a directory writable by the
   execution system's user — required, no default) in `.env`. See
   `.env.example` for the full list and defaults.
2. Each web-service user connects their **own** Tapis account from the
   **Account** page (`/account`) — a standard OAuth `authorization_code`
   login redirect. The service never sees or stores a password; jobs
   submitted from a connected session run and are billed under that user's
   own Tapis identity. A session with no connected account cannot submit
   jobs.
3. (Optional, for jobs with no connected owner — e.g. re-attached on a
   restart) seed a *shared* fallback credential once via
   `TAPIS_REFRESH_TOKEN`, or call
   `tapis_service.bootstrap_refresh_token(username, password)` a single
   time — the password is not needed again afterward, and Tapis rotates the
   refresh token on every use, so the current one lives in the database, not
   the environment.

Datasets must already exist on the execution system — see `remote_path` in
`src/web/config.py`; this container's own dataset paths mean nothing there,
and uploaded datasets are rejected for remote training by design.

## Persist data across container restarts
`docker compose down` stops containers but keeps the named volumes (Postgres data,
object storage, dataset uploads/checkpoints) — `docker compose up` afterward picks up
right where you left off. Use `docker compose down -v` for a full wipe.

---

# Explanation

## Why an analytic proxy score instead of training-based NAS
Traditional neural architecture search trains many candidate networks to compare
them — expensive. FNAS-LCD instead scores each candidate block using an analytic,
zero-shot proxy (a stable-rank-based signal, computed without any forward/backward
pass) precomputed and stored in `data/block_lookup/`, alongside each block's measured
latency on the target device (a Jetson GPU) and its FLOPs/parameter count. Search
becomes a fast dynamic-programming frontier enumeration over this lookup table rather
than an expensive training loop.

## Pipeline stages
1. **Search** (`src/enumeration/`, `src/scoring/`) — given a latency budget and a
   dataset's class count, enumerate architectures stage by stage, keeping a frontier of
   the best-scoring candidates per latency bucket, until a full architecture (stem →
   4 stages → classifier head) is assembled under budget.
2. **Train** — the original recipe (auto_augment, random_erase, mixup, label smoothing,
   cosine LR with warmup) trains the selected architecture. The CLI
   (`src.pipeline.auto_nas`) trains locally; the web service submits a real job to a
   Tapis-managed HPC system and polls it to completion (see
   [Configure remote training](#configure-remote-training-tapis)) — with
   `TAPIS_ENABLED=0` it instead exports a freshly-initialized (untrained) model, for
   demo/deployment-validation purposes.
3. **Export** — the trained (or freshly-initialized, with `TAPIS_ENABLED=0`) model is
   traced and saved as a TorchScript `.pt` file.

## Web service architecture
FastAPI app (`src/web/`), server-rendered Jinja2 templates (no frontend build step),
Postgres for job/dataset metadata, S3-compatible object storage for exported models
(MinIO locally; a config change, not a code change, points it at real S3/ICICLE
storage later). Jobs run in a background thread per submission, with the UI polling
`/api/jobs/{id}/status` every second for live updates. See `docs/` for the ICICLE
deployment requirements and target architecture in more detail.

## Repository layout
- `src/pipeline/` — CLI entrypoints (`auto_nas.py`, `interactive.py`)
- `src/enumeration/`, `src/scoring/` — frontier search and analytic scoring core
- `src/model/` — PlainNet/MasterNet architecture definitions
- `src/DataLoader/` — ImageFolder dataset loading
- `src/training/` — trainer (`timm_trainer.py`)
- `src/web/` — the FastAPI web service (UI + JSON API)
- `data/block_lookup/` — precomputed per-block cost/score lookup table
- `data/toy_dataset/` — small bundled ImageFolder dataset for smoke testing
- `docs/` — ICICLE deployment requirements and architecture notes

---
