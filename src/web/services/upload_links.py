"""Temporary links from which Tapis downloads uploaded datasets.

A user's own Tapis login can't write into FNAS-LCD's folders on Expanse
directly; only a job run through the upload app can, because Tapis lends that
job the app owner's folder permissions. So an upload goes in as that job's
input file: Tapis downloads it from a link to this service, into the job's
folder, before the job waits in Expanse's queue.

Each link is a long random secret for one file, valid for a few hours at
most, and removed as soon as Tapis has copied the file (see
dataset_service.stage_upload). Tapis has no login to give, so the pod lets
requests under PATH_PREFIX through without one, and the route (main.py)
sits outside the login gate and serves only files registered here. Links are
kept in memory only: a restart drops them all.
"""

from __future__ import annotations

import secrets
import threading
import time
from pathlib import Path
from typing import Optional

from ..config import settings

PATH_PREFIX = "/upload-links/"
# Long enough for Tapis to start the job and copy a 10 GB upload.
_LIFETIME_SECONDS = 6 * 3600

_links: dict[str, tuple[Path, float]] = {}
_lock = threading.Lock()


def create(path: Path) -> str:
    """Register a file and return the full URL Tapis can download it from."""
    secret = secrets.token_urlsafe(32)
    now = time.time()
    with _lock:
        for key in [k for k, (_, until) in _links.items() if until <= now]:
            del _links[key]
        _links[secret] = (path, now + _LIFETIME_SECONDS)
    return f"{settings.public_url}{PATH_PREFIX}{secret}"


def revoke(url: str) -> None:
    """Remove a link made by create(). Unknown links are ignored."""
    with _lock:
        _links.pop(url.rsplit("/", 1)[-1], None)


def lookup(secret: str) -> Optional[Path]:
    """The file a live link serves, or None."""
    with _lock:
        held = _links.get(secret)
    if held is None or held[1] <= time.time():
        return None
    return held[0]
