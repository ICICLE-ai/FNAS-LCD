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
3. Submitting a job needs a Tapis login, which the app reads from an
   `X-Tapis-Token` cookie (on an ICICLE pod, its login sets it). Locally, log in
   at https://icicleai.tapis.io/v3/oauth2/webapp, copy the access token it shows,
   and add it in your browser as a cookie named `X-Tapis-Token` for `localhost`.
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
Dataset must be in **ImageFolder format**: a directory with `train/` and `val/`
subfolders (and optionally `test/`), one class per subdirectory.

## Upload your own dataset (web service)
Go to **Datasets → Upload Dataset**, provide a name and a `.zip`/`.tar`/`.tar.gz`/`.tgz`
archive in ImageFolder format (max 10 GB). The page sends it in pieces with
[tus-js-client](https://github.com/tus/tus-js-client), so an upload survives dropped
connections; one that stops can be continued for a day by choosing the same file again
in the same browser. The service then checks the archive (no absolute paths, `..`, links
or special files, and it must fit in the free disk space), extracts it, and checks that
`train/`, `val/` and (if present) `test/` hold the same class folders, each with images,
as training needs; the dataset's page shows the outcome, and failed uploads are removed
after two days. Hidden files and folders (such as macOS's `._*` and `.DS_Store`) and
`__MACOSX/` are skipped. Scripts can upload in one request instead, with
`POST /api/datasets/upload`. Uploading requires a Tapis login, the same as submitting a
job; only the uploader can delete the dataset (`DELETE /api/datasets/{id}`), and only
while no job has used it.

With remote training enabled, an upload is then copied to SDSC Expanse in the
background (status **copying** on its page) and becomes usable once it is **ready**.
Deleting it also deletes that copy (status **deleting** until done). SDSC deletes the
copy 90 days after the upload; the dataset's page shows the date, and the dataset is
then **expired** — see [Configure remote training](#configure-remote-training-tapis).

## Check or cancel a running job
The **Jobs** page lists every job with live status and filtering by status. A job can
be cancelled from its detail page while still `pending`/`searching`.

## Configure remote training (Tapis)
Training runs as a real GPU job on SDSC Expanse via
[Tapis](https://tapis-project.org/), not in this container. With
`TAPIS_ENABLED=0` (the default), nothing runs on Expanse: jobs skip training
and export an untrained model. With `TAPIS_ENABLED=1` (see `.env.example` for
the related settings):

- Each user's jobs are submitted with their own Tapis login, the token in
  their `X-Tapis-Token` cookie. The service holds no Tapis credentials of its
  own. It keeps each user's newest token in memory only, to track their jobs;
  if the token expires while a job runs, the job carries on, and the service
  collects its result the next time that user opens FNAS-LCD.
- Jobs run through the two Tapis apps in `tapis/`, which must be registered
  once (see `tapis/README.md`). On Expanse they run as a shared account, so
  other ICICLE users can read the data there.
- Tapis downloads each uploaded dataset from a temporary link under
  `PUBLIC_URL` (`/upload-links/...`), so that path must be reachable without
  logging in.
- Expanse ends a job after 800 minutes (`TAPIS_MAX_MINUTES` can only lower
  this) and runs at most 10 of a user's jobs at a time per queue, and SDSC
  deletes datasets 90 days after they're uploaded. The site shows these limits.

**Hosted behind a login portal (e.g. an ICICLE pod).** If the platform
logs users in and passes their Tapis token to the app as an `X-Tapis-Token`
cookie, set `REQUIRE_TAPIS_TOKEN=1`. Every page and API route except
`/health`, `/static` and the upload links then requires a valid token
(checked against Tapis). Leave it `0` (the default) for local use.

Datasets must exist on Expanse: this container's own dataset paths mean
nothing there. Uploaded datasets are copied there automatically. A built-in
dataset is offered for training only if it names its location on Expanse
with `remote_path` in `src/web/config.py`; for the toy dataset, set
`TOY_DATASET_REMOTE_PATH` to the dataset folder of an upload of it.

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
`/api/jobs/{id}/status` every second for live updates. See
`docs/architecture.md` for the target architecture in more detail, and
`docs/deployment.md` for ICICLE deployment requirements and status.

## Repository layout
- `src/pipeline/` — CLI entrypoints (`auto_nas.py`, `interactive.py`)
- `src/enumeration/`, `src/scoring/` — frontier search and analytic scoring core
- `src/model/` — PlainNet/MasterNet architecture definitions
- `src/DataLoader/` — ImageFolder dataset loading
- `src/training/` — trainer (`timm_trainer.py`) and the entry point of the jobs on
  Expanse (`expanse_job.py`)
- `src/web/` — the FastAPI web service (UI + JSON API)
- `tapis/` — the Tapis apps the jobs on Expanse run through, and their registration
  script
- `data/block_lookup/` — precomputed per-block cost/score lookup table
- `data/toy_dataset/` — small bundled ImageFolder dataset for smoke testing
- `docs/architecture.md` — target architecture and identity model
- `docs/deployment.md` — ICICLE deployment requirements and current status

---
