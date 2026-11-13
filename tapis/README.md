# Tapis apps

FNAS-LCD's jobs on SDSC Expanse run through two Tapis apps:

- `fnas-lcd-expanse-train-0.1.0.app.json` trains one searched architecture
  on one GPU (queue `tapisGPUshared`).
- `fnas-lcd-expanse-upload-0.1.0.app.json` copies an uploaded dataset onto
  Expanse, or deletes one (queue `tapisShared`). Tapis downloads the dataset
  from a temporary link the service gives it.

Both run on `expanse-tapis-static`, where every job runs as Expanse's shared
`harvest` account and is charged to the `uot260` allocation. Each upload is
kept in `/jobs/fnas-lcd/uploads/<upload job id>/dataset`, in Expanse's
scratch space, which SDSC purges 90 days after files are written.

Both apps run the service's own image,
`ghcr.io/icicle-ai/fnas-lcd:<service-version>`. Its `entrypoint.sh` hands the
job's mode (`train`, `upload` or `delete`) to `src/training/expanse_job.py`,
which accepts only the arguments the service sends. The apps fix everything
else: the system, the job folders, the queue, the allocation and the
container options.

The apps are owned by one ICICLE account and shared with every ICICLE user.
Users submit jobs with their own login. Because the job folders are set in
the app, Tapis lets each job use the app owner's permission on them, so users
need no setup of their own.

## Registering or updating the apps

Run this as the account that owns the apps:

    .venv/bin/python tapis/register_apps.py

It asks for that account's Tapis username and password, which are never
printed or saved. It creates each app, or updates it to match its file, and
shares it with every ICICLE user.

The image tag in both files must match `service-version` in
`icicle-service.yaml`, so that the jobs run the same code as the website; the
script checks this. When `service-version` changes, change the tag in both
files and run the script again.
