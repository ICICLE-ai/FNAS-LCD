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
- If remote training is enabled (`TAPIS_ENABLED=1`): no Tapis credentials.
  Each user's jobs are submitted with their own token, from the
  `X-Tapis-Token` cookie the pod's login sets (see `architecture.md`).
  `PUBLIC_URL` must be the pod's public address, and the pod's networking
  must let `/upload-links/` through without login (Pods'
  `tapis_auth_excluded_paths`), because Tapis downloads uploaded datasets
  from temporary links there.

## Tapis apps

Training jobs, and the jobs that copy uploaded datasets onto Expanse or
delete them, run through the two apps in `tapis/`. They must be registered
before first use, and again whenever they change; see `tapis/README.md`.
Until then, uploads and training fail, and the dataset's or job's page
shows Tapis's error.

## Current status (November 2026)

**The pod runs** at https://fnaslcd.pods.icicleai.tapis.io, behind the
pod's Tapis login. That login sets the `X-Tapis-Token` cookie the app uses
to identify users (`REQUIRE_TAPIS_TOKEN` isn't set). ICICLE staff
configured the database, object storage and Tapis settings on the pod. Its
logs are visible in TapisUI.

**Training** runs on SDSC Expanse (`expanse-tapis-static`), with each
user's own ICICLE login, through the apps in `tapis/`. Every job runs as
Expanse's shared `harvest` account and is charged to the `uot260`
allocation, like other ICICLE services' jobs there. Uploaded datasets are
copied there before they can be trained. Expanse ends a job after 800
minutes and runs at most 10 of a user's jobs at a time per queue; SDSC
deletes the datasets 90 days after they're written. The site shows all
three limits.

Training used to run on OSC Pitzer under one shared account. The pod
still has that setup's settings (`TAPIS_CLIENT_ID`, `TAPIS_CLIENT_KEY`,
`TAPIS_REFRESH_TOKEN`, `TAPIS_WORK_BASE`); this version ignores them.

**The deploy pipeline stops before restarting the pod.** `deploy.yaml`
uses v2.2.2 of the shared pipeline. Its "Deploy image and verify rollout"
step points the pod at the image it just built, by digest
(`ghcr.io/icicle-ai/fnas-lcd@sha256:…`), and then restarts the pod. Tapis
Pods currently rejects digest image references (HTTP 400), so the step
fails before the restart. The image is also pushed as
`ghcr.io/icicle-ai/fnas-lcd:<service-version>`, which is what the pod
runs, so restarting the pod in TapisUI deploys it. Even once that's fixed,
the health check would fail, because `/health` sits behind the pod's Tapis
login. Both need a fix on ICICLE's side.
