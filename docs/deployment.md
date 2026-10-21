# ICICLE deployment requirements

What's needed to run FNAS-LCD as an ICICLE pod, and the current state of
that deployment. For the system design, see `architecture.md` in this
folder; for local development, see the main `README.md`.

## How deployment works

Pushing to `main` triggers `.github/workflows/deploy.yaml`
(`on: push: branches: [main]`, also runnable manually via
`workflow_dispatch`). That workflow calls ICICLE's shared
`icicle-ai/cicd-templates` pipeline, which:

1. Parses and validates `icicle-service.yaml` (service identity, runtime,
   system package requirements — not a Dockerfile; ICICLE builds the
   image itself from this plus an official base Dockerfile it fetches).
2. Builds the image and pushes it to ICICLE's container registry.
3. Sets the image of the pod named in `icicle-service.yaml`'s `pod-name`
   to the new build, restarts the pod, and checks that its `/health`
   reports the new build (this step currently fails; see Current status).

`component-info.yaml` is separate — ICICLE catalog metadata (owner,
classification, license, etc.), not part of the build or deploy mechanics.
It still has several unresolved `CONFIRM`/`TODO` fields (license,
`publicAccess`, `codeReviewConducted`) that were deliberately left for the
maintainer/team to confirm rather than guessed.

## Required configuration

The repo's `.env.example` documents every setting the running app reads.
Locally, `docker compose` supplies these directly. On the ICICLE pod they
are set on the pod itself, in TapisUI (environment variables, with secrets
in its secret map); the deploy pipeline doesn't touch them. At minimum, a
running pod needs:

- Postgres connection info (`POSTGRES_HOST`/`PORT`/`USER`/`PASSWORD`/`DB`)
- Object storage (`S3_ENDPOINT_URL`/`ACCESS_KEY`/`SECRET_KEY`/`BUCKET`)
- If remote training is enabled (`TAPIS_ENABLED=1`): a Tapis OAuth client
  (`TAPIS_CLIENT_ID`/`KEY`); the refresh token of the shared service
  account that every training job runs under (seeded once with
  `TAPIS_REFRESH_TOKEN` or `bootstrap_refresh_token()`; Tapis rotates it on
  every use, so the service keeps the current one in its database); and
  `TAPIS_WORK_BASE`, a directory on the execution system that account can
  write to. Users are identified separately: by the `X-Tapis-Token` cookie
  a hosting platform's login sets, or by connecting an account on the
  Account page (see `architecture.md`).

## Tapis apps

Training jobs run the `fnas-lcd-train` app (`tapis/fnas-lcd-train-0.1.3.app.json`).
Uploaded datasets additionally need the small `fnas-lcd-unpack` app
(`tapis/fnas-lcd-unpack-0.1.0.app.json`), which unpacks an uploaded dataset in
place on the execution system. Until it is registered and shared, uploads fail
at the copying step, and the dataset's page shows Tapis's error. To register
it, once:

1. Create the app from the JSON file, under any Tapis account in the tenant
   (TapisUI: Apps → create, pasting the JSON; or `POST /v3/apps`).
2. Share it with the account the service runs jobs as (the one whose
   refresh token the pod holds), so that account may launch it:
   `POST /v3/apps/share/fnas-lcd-unpack` with `{"users": ["<that account>"]}`.

Its paths (`/fs/ess/PAS2699/...`), queue (`cpu`) and `--account=PAS2699` are
Pitzer-specific, like the training app's. The service sends its own job
directories (from `TAPIS_WORK_BASE`) with every job; the app and version it
uses can be changed with `TAPIS_UNPACK_APP_ID`/`TAPIS_UNPACK_APP_VERSION`.

## Current status (October 2026)

**The pod runs** at https://fnaslcd.pods.icicleai.tapis.io, behind the
pod's Tapis login. That login sets the `X-Tapis-Token` cookie the app uses
to identify users (`REQUIRE_TAPIS_TOKEN` isn't set). ICICLE staff
configured the database, object storage and Tapis settings on the pod. Its
logs are visible in TapisUI.

**Training** runs on OSC Pitzer under one shared service account, whose
refresh token the pod holds, on the PAS2699 allocation
(`TAPIS_WORK_BASE=/fs/ess/PAS2699/jhyan`). Uploaded datasets are copied
there before they can be trained (see "Tapis apps" above).

**The deploy pipeline stops before restarting the pod.** `deploy.yaml`
uses the shared pipeline's `@main`, which became v2.2.1 in October 2026.
Its "Deploy image and verify rollout" step sets the pod's image to
`ghcr.io/icicle-ai/fnas-lcd:<service-version>` and then restarts the pod.
While `service-version` stays at 0.1.0 the image name never changes, so
Tapis rejects the update as making no change, and the step fails before
the restart. The new image is still built and pushed under the same tag,
so restarting the pod in TapisUI deploys it. Even once that's fixed, the
health check would fail, because `/health` sits behind the pod's Tapis
login. Both need a fix on ICICLE's side.
