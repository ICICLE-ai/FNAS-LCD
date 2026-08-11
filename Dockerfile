# Local/compose Dockerfile — self-authored, NOT a copy of the ICICLE-managed
# Dockerfile from icicle-ai/cicd-templates (that one is fetched fresh by
# their CI and explicitly shouldn't be forked into this repo). This one is
# for `docker compose up` locally, to validate app + Postgres + storage
# before ICICLE wires the real deployment onto Pods.

FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# libgomp1 — required by torch's OpenMP-based CPU kernels.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so this layer is cached across code-only changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project

COPY . .
RUN uv sync --locked

EXPOSE 8000
ENTRYPOINT ["sh", "entrypoint.sh"]
