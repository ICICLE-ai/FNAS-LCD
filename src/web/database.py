"""Postgres database connection and initialization."""

from __future__ import annotations

from contextlib import contextmanager

import psycopg
import psycopg.rows

from .config import settings

# ── connection management ──────────────────────────────────────────────────


def _dsn() -> str:
    return (
        f"host={settings.postgres_host} port={settings.postgres_port} "
        f"user={settings.postgres_user} password={settings.postgres_password} "
        f"dbname={settings.postgres_db}"
    )


def get_connection() -> psycopg.Connection:
    """Get a new Postgres connection (for use in context managers).

    Uses dict_row so the `row["col"]` access patterns used throughout the
    app keep working unchanged.
    """
    return psycopg.connect(_dsn(), row_factory=psycopg.rows.dict_row)


@contextmanager
def get_db():
    """Context manager yielding a connection with auto-commit on success."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── schema ─────────────────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS datasets (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    path            TEXT NOT NULL,
    num_classes     INTEGER NOT NULL DEFAULT 0,
    is_upload       INTEGER NOT NULL DEFAULT 0,
    is_ready        INTEGER NOT NULL DEFAULT 0,
    description     TEXT DEFAULT '',
    created_at      TEXT NOT NULL DEFAULT to_char(now(), 'YYYY-MM-DD HH24:MI:SS')
);

CREATE TABLE IF NOT EXISTS device_configs (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    lookup_csv      TEXT,
    constraint_column TEXT,
    display_name    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id              SERIAL PRIMARY KEY,
    dataset_id      INTEGER NOT NULL REFERENCES datasets(id),
    device_type     TEXT NOT NULL DEFAULT 'jetson_gpu',
    budget          REAL NOT NULL,
    metric          TEXT NOT NULL DEFAULT 'analytic_ssr_exp_proj_dw_mean',
    aggregate       TEXT NOT NULL DEFAULT 'sum',
    epochs          INTEGER NOT NULL DEFAULT 100,
    warmup          INTEGER NOT NULL DEFAULT 5,
    batch_size      INTEGER NOT NULL DEFAULT 64,
    workers         INTEGER NOT NULL DEFAULT 4,
    lr             REAL NOT NULL DEFAULT 0.1,
    gpu             INTEGER NOT NULL DEFAULT 0,
    max_iters       INTEGER,
    structure_str   TEXT,
    status          TEXT NOT NULL DEFAULT 'pending',
    error_message   TEXT,
    best_val_acc    REAL,
    output_path     TEXT,
    save_dir        TEXT,
    search_score    REAL,
    search_constraint REAL,
    search_flops    REAL,
    search_params   REAL,
    created_at      TEXT NOT NULL DEFAULT to_char(now(), 'YYYY-MM-DD HH24:MI:SS'),
    started_at      TEXT,
    finished_at     TEXT
);
"""

# Columns added after the initial schema; applied to pre-existing databases
# that were created before these columns existed.
_JOBS_MIGRATION_COLUMNS = {
    "search_score": "REAL",
    "search_constraint": "REAL",
    "search_flops": "REAL",
    "search_params": "REAL",
    "max_iters": "INTEGER",
}


def _migrate_jobs_table(conn: psycopg.Connection) -> None:
    existing = {
        row["column_name"]
        for row in conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
            ("jobs",),
        )
    }
    for column, col_type in _JOBS_MIGRATION_COLUMNS.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE jobs ADD COLUMN {column} {col_type}")


def init_db() -> None:
    """Create tables and seed initial data.  Idempotent."""
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    for d in (settings.uploads_dir, settings.checkpoints_dir, settings.models_dir):
        d.mkdir(parents=True, exist_ok=True)

    with get_db() as conn:
        conn.execute(_SCHEMA_SQL)
        _migrate_jobs_table(conn)

        for dc in settings.device_configs:
            conn.execute(
                "INSERT INTO device_configs (name, lookup_csv, constraint_column, display_name) "
                "VALUES (%s, %s, %s, %s) ON CONFLICT (name) DO NOTHING",
                (dc["name"], dc["lookup_csv"], dc["constraint_column"], dc["display_name"]),
            )

        for ds in settings.preconfigured_datasets:
            conn.execute(
                "INSERT INTO datasets (name, path, num_classes, is_upload, is_ready, description) "
                "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (name) DO NOTHING",
                (ds["name"], ds["path"], ds["num_classes"], ds["is_upload"], ds["is_ready"], ds["description"]),
            )
