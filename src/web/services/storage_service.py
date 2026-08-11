"""Generic S3-compatible object storage wrapper (MinIO locally, real S3/ICICLE
storage later — swapping S3_ENDPOINT_URL/credentials is a config change, not
a rewrite).

Deliberately generic (upload_object/download_object keyed by an arbitrary
object key), not model-specific: today only exported models go through
this, but datasets are expected to move to S3-backed storage eventually
too (keyed under e.g. "datasets/<name>.zip"). Datasets will still need a
pull-and-extract-to-local-disk step at use time regardless, since
PyTorch's ImageFolder/DataLoader need real filesystem paths, not an S3 API
— this module is where that download step would plug in.
"""

from __future__ import annotations

from pathlib import Path

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from ..config import settings


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        config=Config(signature_version="s3v4"),
    )


def ensure_bucket() -> None:
    """Create the configured bucket if it doesn't already exist. Idempotent."""
    client = get_s3_client()
    try:
        client.head_bucket(Bucket=settings.s3_bucket)
    except ClientError:
        client.create_bucket(Bucket=settings.s3_bucket)


def upload_object(local_path: Path, key: str) -> None:
    """Upload a local file to object storage under `key`."""
    get_s3_client().upload_file(str(local_path), settings.s3_bucket, key)


def download_object(key: str) -> bytes:
    """Fetch an object's full contents as bytes."""
    obj = get_s3_client().get_object(Bucket=settings.s3_bucket, Key=key)
    return obj["Body"].read()


def object_exists(key: str) -> bool:
    try:
        get_s3_client().head_object(Bucket=settings.s3_bucket, Key=key)
        return True
    except ClientError:
        return False
