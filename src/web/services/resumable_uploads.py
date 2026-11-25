"""Resumable dataset uploads, with the tus protocol (https://tus.io).

The upload page sends an archive with tus-js-client (static/), in pieces:
one request starts the upload, and each piece is then appended where the
server says the upload stands (routers/uploads.py). So a dropped
connection costs at most the piece in flight, and an upload that stopped
continues where it stopped. Once the whole archive has arrived, it is
checked, unpacked and copied to Expanse in the background
(dataset_service.check_upload).

Unfinished uploads are kept in this process and on this container's disk,
which a restart wipes (see dataset_service.forget_interrupted_uploads),
and are removed once no data has arrived for settings.upload_resume_days.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import settings
from . import dataset_service

_CLEANUP_SECONDS = 3600  # how often old uploads are looked for


@dataclass
class Upload:
    ds_id: int
    name: str
    owner: str
    filename: str
    length: int
    path: Path
    touched: float = field(default_factory=time.time)  # when data last arrived
    busy: bool = False  # a piece is arriving
    done: bool = False  # all of it arrived, and it's being checked

    def offset(self) -> int:
        """How many bytes of the archive have arrived."""
        if self.done:
            return self.length
        try:
            return self.path.stat().st_size
        except FileNotFoundError:  # no data yet
            return 0


class Busy(Exception):
    """A piece of the same upload is still arriving."""


_uploads: dict[int, Upload] = {}
_lock = threading.Lock()


def start(name: str, filename: str, length: int, owner: str) -> Upload:
    """Start an upload of `length` bytes as a new dataset. Raises ValueError,
    with a message for the user, if it can't be accepted."""
    free = dataset_service.free_space()
    if length > free:
        raise ValueError(
            f"This file is larger than the {dataset_service.format_size(free)} of disk space "
            f"the server has available right now. Please try again later."
        )
    ds_id, safe_name = dataset_service.reserve_dataset(name, owner, status="uploading")
    upload = Upload(ds_id, safe_name, owner, filename, length,
                    dataset_service.archive_path(ds_id, safe_name))
    with _lock:
        _uploads[ds_id] = upload
    return upload


def find(ds_id: int, owner: str) -> Upload | None:
    """The owner's upload `ds_id`, if it is unfinished or has just finished."""
    with _lock:
        upload = _uploads.get(ds_id)
    return upload if upload is not None and upload.owner == owner else None


def claim(ds_id: int, owner: str) -> Upload:
    """Take an unfinished upload for one arriving piece; release() it after.
    Raises LookupError if there is no such upload, Busy if a piece is
    already arriving."""
    with _lock:
        upload = _uploads.get(ds_id)
        if upload is None or upload.owner != owner or upload.done:
            raise LookupError(ds_id)
        if upload.busy:
            raise Busy(ds_id)
        upload.busy = True
    return upload


def release(upload: Upload) -> None:
    """After a piece: once the whole archive is in, have it checked."""
    with _lock:
        upload.busy = False
        upload.touched = time.time()
        # Unless the upload was cancelled meanwhile.
        upload.done = _uploads.get(upload.ds_id) is upload and upload.offset() == upload.length
    if upload.done:
        dataset_service.check_upload(upload.ds_id, upload.name, upload.path,
                                     upload.filename, upload.owner)


def cancel(ds_id: int) -> None:
    """Forget an upload, because its dataset is being deleted."""
    with _lock:
        _uploads.pop(ds_id, None)


def remove_stale() -> None:
    """Remove uploads that got no data for settings.upload_resume_days. Those
    that finished are only forgotten: their datasets live on."""
    cutoff = time.time() - settings.upload_resume_days * 24 * 3600
    with _lock:
        stale = [u for u in _uploads.values() if not u.busy and u.touched < cutoff]
        for upload in stale:
            del _uploads[upload.ds_id]
    for upload in stale:
        if not upload.done:
            dataset_service.delete_unfinished(upload.ds_id, upload.name)


def keep_clean() -> None:
    """Remove stale uploads, and failed ones (see
    dataset_service.remove_failed_uploads), every hour in the background."""
    def run() -> None:
        while True:
            time.sleep(_CLEANUP_SECONDS)
            try:
                remove_stale()
                dataset_service.remove_failed_uploads()
            except Exception as e:  # noqa: BLE001 - tried again in an hour
                print(f"Removing old uploads failed: {e}", flush=True)

    threading.Thread(target=run, daemon=True, name="upload-cleanup").start()
