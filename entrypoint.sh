#!/usr/bin/env sh
set -eu
# FNAS-LCD's jobs on SDSC Expanse run this image with a mode as the first
# argument (see tapis/ and src/training/expanse_job.py). The ICICLE pod runs
# it with none, to serve the website.
case "${1:-}" in
    train|upload|delete)
        exec /app/.venv/bin/python -I /app/src/training/expanse_job.py "$@" ;;
esac
exec uvicorn src.web.main:app --host 0.0.0.0 --port "${PORT:-8000}"
