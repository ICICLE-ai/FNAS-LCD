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
3. Triggers a Tapis Pods deployment for the pod named in
   `icicle-service.yaml`'s `pod-name`.

`component-info.yaml` is separate — ICICLE catalog metadata (owner,
classification, license, etc.), not part of the build or deploy mechanics.
It still has several unresolved `CONFIRM`/`TODO` fields (license,
`publicAccess`, `codeReviewConducted`) that were deliberately left for the
maintainer/team to confirm rather than guessed.

## Required configuration

The repo's `.env.example` documents every setting the running app reads.
Locally, `docker compose` supplies these directly. **On a pod, nothing
currently configures them** — this is the open item below. At minimum, a
running pod needs:

- Postgres connection info (`POSTGRES_HOST`/`PORT`/`USER`/`PASSWORD`/`DB`)
- Object storage (`S3_ENDPOINT_URL`/`ACCESS_KEY`/`SECRET_KEY`/`BUCKET`)
- If remote training is enabled (`TAPIS_ENABLED=1`): a Tapis OAuth client
  (`TAPIS_CLIENT_ID`/`KEY`) and a way to obtain credentials — either
  `TAPIS_REFRESH_TOKEN` (a shared fallback identity) or
  `REQUIRE_TAPIS_TOKEN=1` (identity comes from the hosting platform
  instead, per `architecture.md`)

## Current status

**The pod does not start.** `icicle-service.yaml`'s `pod-name` is
`fnaslcd` (a required fix — a value containing a hyphen fails), and with
that fix the deploy pipeline itself succeeds end to end: it builds,
pushes, and successfully triggers the Tapis Pods deployment. But the pod
itself crashes immediately after starting
(`https://fnaslcd.pods.icicleai.tapis.io/health` returns the platform's
own "this pod failed to start" page, not a response from this app).

The most likely cause is the configuration gap above: the deploy pipeline
builds and pushes the *image*, but nothing in it configures the running
container's environment variables, and this app cannot start at all
without at least a database to connect to. **How ICICLE wants runtime
secrets/config injected into a pod is, as of this writing, an open
question** — not something resolved in this repo. Resolving it is the
prerequisite for everything else here.

Pod logs (referenced by the platform's own error page) have not been
accessible — pod-level access on the Tapis Pods platform appears to be
separate from repository access and was not available while diagnosing
this.
