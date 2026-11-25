"""Dataset service: upload extraction, validation, class detection."""

from __future__ import annotations

import datetime
import gzip
import os
import re
import shutil
import stat
import tarfile
import threading
import zipfile
import zlib
from pathlib import Path
from typing import BinaryIO

from ..config import settings
from ..database import get_db
from . import tapis_service, upload_links, user_tokens

# Disk space left free when deciding whether an upload fits, so a dataset
# never fills the disk completely.
_FREE_SPACE_MARGIN = 1024 * 1024 * 1024  # 1 GB

# The archive types an upload may be.
ARCHIVE_TYPES = (".zip", ".tar", ".tar.gz", ".tgz")
# The image types training reads (torchvision's ImageFolder).
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
# Training always loads both of these folders, and test/ too if there is one.
_SPLITS = ("train", "val")
# What reading a cut-off or corrupted archive raises.
_UNREADABLE = (EOFError, tarfile.ReadError, zipfile.BadZipFile, zlib.error, gzip.BadGzipFile)
_UNREADABLE_MESSAGE = (
    "This archive is incomplete or damaged, so it can't be read. This usually means the "
    "file wasn't completely downloaded or copied. Check that it opens on your computer, "
    "then upload it again."
)


class ArchiveRejected(ValueError):
    """An upload that can't be accepted; the message tells the user why."""


def validate_imagefolder(data_dir: Path) -> int:
    """Check that training can load an unpacked dataset, as torchvision's
    ImageFolder sees it: train/ and val/, and test/ if there is one, hold
    the same class folders, each with an image in it or in its subfolders.
    Returns the number of classes; raises ArchiveRejected saying what to fix."""
    splits = _SPLITS + (("test",) if (data_dir / "test").is_dir() else ())
    classes = None  # train/'s
    empty = []
    for split in splits:
        folder = data_dir / split
        optional = " test/ is optional, so you can also leave it out." if split == "test" else ""
        names = sorted(d.name for d in folder.iterdir() if d.is_dir()) if folder.is_dir() else []
        if not names:
            raise ArchiveRejected(
                f"The {split}/ folder has no class folders. Put each class's images in its "
                f"own folder, for example {split}/cats/ and {split}/dogs/.{optional}"
            )
        if classes is None:
            classes = names
        elif names != classes:
            only_train = [n for n in classes if n not in names]
            only_here = [n for n in names if n not in classes]
            raise ArchiveRejected(
                f"{split}/ must have the same class folders as train/."
                + (f" Only in train/: {_some(only_train)}." if only_train else "")
                + (f" Only in {split}/: {_some(only_here)}." if only_here else "")
                + optional
            )
        missing = [n for n in names if not _has_image(folder / n)]
        if len(missing) == len(names):
            raise ArchiveRejected(
                f"No images found in the class folders under {split}/. "
                f"Supported types: {', '.join(IMAGE_EXTENSIONS)}.{optional}"
            )
        empty += [f"{split}/{n}/" for n in missing]
    if empty:
        raise ArchiveRejected(f"These class folders have no images: {_some(empty)}. "
                              f"Each class folder needs at least one image.")
    return len(classes)


def _has_image(folder: Path) -> bool:
    """Whether training finds an image in a class folder, anywhere in it."""
    return any(name.lower().endswith(IMAGE_EXTENSIONS)
               for _, _, names in os.walk(folder) for name in names)


def _some(names: list[str], shown: int = 5) -> str:
    """The first few names, for a message, and how many more there are."""
    more = f" (and {len(names) - shown} more)" if len(names) > shown else ""
    return ", ".join(names[:shown]) + more


def _sanitize_name(name: str) -> str:
    """Keep only alphanumeric, dash, underscore."""
    import re
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def format_size(num_bytes: float) -> str:
    """Human-readable size in binary units, e.g. "512 bytes", "1.5 MB", "10 GB"."""
    size = float(num_bytes)
    for unit in ("bytes", "KB", "MB", "GB"):
        if size < 1024:
            break
        size /= 1024
    else:
        unit = "TB"
    text = f"{size:.0f}" if unit == "bytes" else f"{size:.1f}".removesuffix(".0")
    return f"{text} {unit}"


def _unsafe_path(name: str) -> str | None:
    """Why an archive entry's path could escape the dataset folder, or None."""
    # Some unpackers treat backslashes as separators, so check both.
    path = name.replace("\\", "/")
    if path.startswith("/") or re.match(r"[A-Za-z]:", path):
        return "an absolute path"
    if ".." in path.split("/"):
        return 'a ".." path'
    return None


def _is_skipped(name: str) -> bool:
    """Archive entries left out of a dataset, which hold no images: hidden
    files and folders (names starting with "."), such as macOS's "._" files
    and .DS_Store or Jupyter's .ipynb_checkpoints, and Finder's __MACOSX/
    folder. Left in, training would take a hidden folder for a class, or a
    "._x.jpg" for an image, and fail."""
    parts = [p for p in name.replace("\\", "/").split("/") if p not in ("", ".")]
    return any(p == "__MACOSX" or (p.startswith(".") and p != "..") for p in parts)


def inspect_archive(archive: BinaryIO, filename: str, max_total: int) -> tuple[int, int]:
    """Check every entry of an uploaded archive without extracting anything.
    Returns the total size of its files once unpacked, and how many hidden
    or macOS metadata files will be skipped (see _is_skipped).

    Rejects entries that could write outside the dataset folder (absolute
    paths, ".." components, symbolic or hard links) and special files such as
    devices: an image dataset only needs folders and regular files. Also
    rejects the archive as soon as its contents exceed `max_total` bytes, the
    disk space available for unpacking it, and an archive without train/ and
    val/ folders where extract_archive_to looks for them (see _check_layout).
    """
    total = 0
    skipped = 0
    top: dict[str, bool] = {}  # top-level names -> whether each is a folder
    nested: set[tuple[str, str]] = set()  # folders one level down, as (parent, name)
    archive.seek(0)

    def check(name: str, problem: str | None, size: int, is_dir: bool) -> None:
        nonlocal total
        if problem:
            raise ArchiveRejected(
                f'The archive contains {problem} ("{name}"), which isn\'t allowed. '
                f"A dataset archive may only contain folders and regular files."
            )
        total += size
        if total > max_total:
            raise ArchiveRejected(
                f"This archive unpacks to more than the {format_size(max_total)} of "
                f"disk space the server has available right now. Please upload a "
                f"smaller dataset, or try again later."
            )
        parts = [p for p in name.split("/") if p not in ("", ".")]
        if parts:
            top[parts[0]] = top.get(parts[0], False) or is_dir or len(parts) > 1
            if len(parts) > 2 or (len(parts) == 2 and is_dir):
                nested.add((parts[0], parts[1]))

    if filename.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if _is_skipped(info.filename):
                    skipped += not info.is_dir()
                    continue
                # Unix file type bits, when the zip records them.
                kind = stat.S_IFMT(info.external_attr >> 16)
                problem = _unsafe_path(info.filename)
                if not problem and kind == stat.S_IFLNK:
                    problem = "a link"
                elif not problem and kind not in (0, stat.S_IFREG, stat.S_IFDIR):
                    problem = "a special file"
                check(info.filename, problem, 0 if info.is_dir() else info.file_size,
                      info.is_dir())
    else:
        mode = "r:gz" if filename.endswith((".tar.gz", ".tgz")) else "r:"
        with tarfile.open(fileobj=archive, mode=mode) as tf:
            for member in tf:
                if _is_skipped(member.name):
                    skipped += not member.isdir()
                    continue
                problem = _unsafe_path(member.name)
                if not problem and (member.issym() or member.islnk()):
                    problem = "a link"
                elif not problem and not (member.isreg() or member.isdir()):
                    problem = "a special file"
                check(member.name, problem, member.size if member.isreg() else 0,
                      member.isdir())
    _check_layout(top, nested)
    return total, skipped


def _check_layout(top: dict[str, bool], nested: set[tuple[str, str]]) -> None:
    """Reject an archive without train/ and val/ folders where
    extract_archive_to looks for them: at its top level, or inside its only
    top-level folder. `top` and `nested` are as collected by inspect_archive."""
    folders = {name for name, is_dir in top.items() if is_dir}
    if "train" not in folders and len(top) == 1 and folders:
        folders = {name for parent, name in nested if parent in folders}
    if "train" not in folders:
        inner = sorted(name for name, is_dir in top.items()
                       if not is_dir and name.lower().endswith(ARCHIVE_TYPES + (".gz",)))
        if inner:
            raise ArchiveRejected(
                f"This archive contains another archive, {inner[0]}, instead of the "
                f"dataset's folders. Upload that file itself, or archive the dataset "
                f"folder (the one that contains train/)."
            )
        listing = sorted(name + "/" if is_dir else name for name, is_dir in top.items())
        raise ArchiveRejected(
            f"The archive has no train/ folder. It needs train/ and val/ at its top level, "
            f"or inside one top-level folder. At its top level it has: "
            f"{_some(listing) or 'nothing'}."
        )
    if "val" not in folders:
        raise ArchiveRejected(
            "The archive has no val/ folder next to its train/ folder. Training needs both: "
            "train/ to learn from, and val/ to measure accuracy (test/ is optional)."
        )


def _extraction_filter(member: tarfile.TarInfo, dest_path: str) -> tarfile.TarInfo | None:
    """Skip what _is_skipped says; otherwise Python's "data" filter, which refuses
    anything that would land outside the destination -- a second line of
    defense behind inspect_archive."""
    if _is_skipped(member.name):
        return None
    return tarfile.data_filter(member, dest_path)


def extract_archive_to(target_dir: Path, archive: BinaryIO, filename: str) -> Path:
    """Extract uploaded zip/tar/tgz archive into target_dir, flattening
    a single top-level directory if present.

    `archive` is an open, seekable binary file. It is read in place, never
    loaded into memory as a whole, so an upload doesn't need RAM of its size.

    Returns the path to the extracted dataset root (should contain train/).
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    archive.seek(0)
    if filename.endswith(".zip"):
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(target_dir, members=[
                i for i in zf.infolist() if not _is_skipped(i.filename)])
    elif filename.endswith((".tar.gz", ".tgz", ".tar")):
        mode = "r:gz" if filename.endswith((".tar.gz", ".tgz")) else "r:"
        with tarfile.open(fileobj=archive, mode=mode) as tf:
            tf.extractall(target_dir, filter=_extraction_filter)
    else:
        raise ValueError(f"Unsupported archive format: {filename}")

    # Flatten if everything is inside a single top-level directory
    entries = list(target_dir.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        inner = entries[0]
        # Check that inner has train/
        if (inner / "train").is_dir():
            # Move contents up one level
            for item in inner.iterdir():
                dest = target_dir / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(dest))
            inner.rmdir()

    return target_dir


def local_upload_dir(ds_id: int, name: str) -> Path:
    """Where an upload is unpacked on this container's disk."""
    return settings.uploads_dir / f"{ds_id}_{name}"


def archive_path(ds_id: int, name: str) -> Path:
    """Where a resumable upload's archive is written as it arrives."""
    return settings.uploads_dir / f"{ds_id}_{name}.upload"


def delete_local_files(ds_id: int, name: str) -> None:
    """Remove whatever an upload left on this container's disk: its archive
    as uploaded (resumable uploads), the folder it was unpacked into, and the
    archive re-packed from that for copying (see stage_upload)."""
    folder = local_upload_dir(ds_id, name)
    shutil.rmtree(folder, ignore_errors=True)
    for suffix in (".upload", ".tar"):
        folder.with_name(folder.name + suffix).unlink(missing_ok=True)


def free_space() -> int:
    """Disk space an upload may use right now, in bytes."""
    return max(shutil.disk_usage(settings.uploads_dir).free - _FREE_SPACE_MARGIN, 0)


def reserve_dataset(name: str, uploaded_by: str, status: str | None = None) -> tuple[int, str]:
    """Add a new upload's dataset entry, before its archive is checked, which
    claims its name. Returns its id and its name as stored (characters other
    than letters, digits, - and _ become _); raises ValueError if that name
    is taken. `uploaded_by` is the uploader's Tapis username; only they may
    delete it."""
    import psycopg

    safe_name = _sanitize_name(name)
    try:
        with get_db() as conn:
            ds_id = conn.execute(
                "INSERT INTO datasets (name, path, num_classes, is_upload, is_ready, description, "
                "uploaded_by, copy_status) VALUES (%s, '', 0, 1, 0, 'Uploading...', %s, %s) "
                "RETURNING id",
                (safe_name, uploaded_by, status),
            ).fetchone()["id"]
    except psycopg.errors.UniqueViolation:
        raise ValueError(f'A dataset named "{safe_name}" already exists. Please choose another name.')
    return ds_id, safe_name


def _unpack(ds_id: int, name: str, archive: BinaryIO, filename: str) -> tuple[int, int]:
    """Check an uploaded archive and unpack it into its dataset's folder.
    Returns the number of classes and how many hidden or macOS metadata
    files were skipped; raises ArchiveRejected saying what's wrong."""
    upload_dir = local_upload_dir(ds_id, name)
    try:
        _, skipped = inspect_archive(archive, filename, max_total=free_space())
        extract_archive_to(upload_dir, archive, filename)
    except _UNREADABLE as e:
        raise ArchiveRejected(_UNREADABLE_MESSAGE) from e
    return validate_imagefolder(upload_dir), skipped


def _accept(ds_id: int, name: str, filename: str, num_classes: int, skipped: int) -> bool:
    """Record an upload that passed the checks. With remote training on, it
    becomes ready only once it has been copied to Expanse (stage_upload,
    started here); until then it isn't offered for search or training.
    Returns whether it is ready now."""
    description = f"Uploaded: {filename}"
    if skipped:
        description += (f"; skipped {skipped} hidden or macOS metadata "
                        f"file{'s' if skipped > 1 else ''}")
    staged = settings.tapis_enabled
    _set_dataset(ds_id, path=str(local_upload_dir(ds_id, name)), num_classes=num_classes,
                 is_ready=0 if staged else 1, description=description,
                 copy_status="copying" if staged else None)
    if staged:
        threading.Thread(target=stage_upload, args=(ds_id,), daemon=True,
                         name=f"stage-dataset-{ds_id}").start()
    return not staged


def add_uploaded_dataset(name: str, archive: BinaryIO, filename: str, uploaded_by: str) -> dict:
    """Check, unpack and add a dataset uploaded in one request (the API's
    /datasets/upload); `archive` is the uploaded file, open for reading.
    A rejected upload is removed again, and the error raised for the caller
    to report. Returns dict with dataset info."""
    ds_id, safe_name = reserve_dataset(name, uploaded_by)
    try:
        num_classes, skipped = _unpack(ds_id, safe_name, archive, filename)
        ready = _accept(ds_id, safe_name, filename, num_classes, skipped)
    except Exception as e:
        delete_local_files(ds_id, safe_name)
        with get_db() as conn:
            conn.execute("DELETE FROM datasets WHERE id = %s", (ds_id,))
        _log_failure(ds_id, safe_name, uploaded_by, _describe(e, filename))
        raise
    return {
        "id": ds_id,
        "name": safe_name,
        "path": str(local_upload_dir(ds_id, safe_name)),
        "num_classes": num_classes,
        "is_ready": ready,
    }


def check_upload(ds_id: int, name: str, archive: Path, filename: str, uploader: str) -> None:
    """Check, unpack and add a resumable upload once its whole archive has
    arrived (see resumable_uploads), in the background. A rejected upload
    stays listed as failed, with the reason, for a while (see
    remove_failed_uploads)."""
    _set_dataset(ds_id, copy_status="checking")
    threading.Thread(target=_check_upload, args=(ds_id, name, archive, filename, uploader),
                     daemon=True, name=f"check-dataset-{ds_id}").start()


def _check_upload(ds_id: int, name: str, archive: Path, filename: str, uploader: str) -> None:
    try:
        with open(archive, "rb") as f:
            num_classes, skipped = _unpack(ds_id, name, f, filename)
        archive.unlink()  # unpacked, so no longer needed
        _accept(ds_id, name, filename, num_classes, skipped)
    except Exception as e:  # noqa: BLE001
        reason = str(e) if isinstance(e, ArchiveRejected) else f"Checking the upload failed: {e}"
        _fail_upload(ds_id, name, uploader, reason, detail=_describe(e, filename))


def _describe(e: Exception, filename: str) -> str:
    """An upload's error for the log, with what caused it."""
    cause = f" ({type(e.__cause__).__name__}: {e.__cause__})" if e.__cause__ else ""
    return f"{filename!r}: {type(e).__name__}: {e}{cause}"


def _log_failure(ds_id: int, name: str, uploader: str, reason: str) -> None:
    """Note a failed upload in the service's log, so what went wrong can be
    looked up later."""
    print(f"Upload of dataset {ds_id} ({name}) by {uploader} failed: {reason}", flush=True)


def _fail_upload(ds_id: int, name: str, uploader: str, reason: str,
                 detail: str | None = None) -> None:
    """Mark an upload failed, with the reason shown to users (and `detail`,
    if given, logged instead), and delete its local files. It is removed
    after settings.failed_upload_days (remove_failed_uploads)."""
    _set_dataset(ds_id, copy_status="failed", copy_error=reason, is_ready=0,
                 failed_at=datetime.datetime.now(datetime.timezone.utc))
    delete_local_files(ds_id, name)
    _log_failure(ds_id, name, uploader, detail or reason)


# ── copying uploads to Expanse ─────────────────────────────────────────────
#
# Training runs on Expanse, which can't see this container's disk. So once an
# upload is validated here, stage_upload copies it there in the background,
# with the uploader's own Tapis login: it re-packs the validated folder into a
# plain tar and runs the upload app. Tapis downloads the tar into that job's
# folder from a temporary link to this service (upload_links), and the job
# unpacks it there. Then the dataset is ready for training, and nothing of it
# is kept on this container's disk (which a pod restart wipes anyway).

_INTERRUPTED = ("Copying to Expanse was interrupted by a service restart. "
                "Please upload the dataset again.")


def _set_dataset(ds_id: int, **columns) -> None:
    assignments = ", ".join(f"{column}=%s" for column in columns)
    with get_db() as conn:
        conn.execute(f"UPDATE datasets SET {assignments} WHERE id=%s", (*columns.values(), ds_id))


def _plain_entry(info: tarfile.TarInfo) -> tarfile.TarInfo:
    """Neutral ownership and permissions for the copy: whatever the
    container's own files carry means nothing on the execution system."""
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mode = 0o755 if info.isdir() else 0o644
    return info


def _repack(src_dir: Path, tar_path: Path) -> None:
    """Pack a validated dataset folder into an uncompressed tar whose top
    level is the folder's contents (train/, val/, ...). Each file is deleted
    as soon as it is in the archive, so this needs little extra disk."""
    with tarfile.open(tar_path, "w", format=tarfile.GNU_FORMAT) as tf:
        for dirpath, dirnames, filenames in os.walk(src_dir):
            dirnames.sort()
            here = Path(dirpath)
            rel = here.relative_to(src_dir)
            if rel != Path("."):
                tf.add(here, arcname=str(rel), recursive=False, filter=_plain_entry)
            for fname in sorted(filenames):
                tf.add(here / fname, arcname=str(rel / fname), recursive=False, filter=_plain_entry)
                (here / fname).unlink()
    shutil.rmtree(src_dir, ignore_errors=True)


def _run_to_end(uuid: str, uploader: str, what: str) -> None:
    """Wait for an upload or delete job to end; raise unless it finished."""
    state = tapis_service.watch(uuid, uploader)
    if state != "FINISHED":
        detail = tapis_service.failure_detail(uuid, uploader, state, 1000)
        raise RuntimeError(f"the {what} job ({uuid}) ended {state}: {detail}")


def _finish_upload(ds_id: int, uuid: str, uploader: str) -> None:
    """Wait for an upload job to end, then mark its dataset ready (or raise)."""
    _run_to_end(uuid, uploader, "upload")
    path = tapis_service.dataset_path(uuid)
    _set_dataset(ds_id, remote_path=path, path=path, copy_status="ready", copy_error=None, is_ready=1)


def stage_upload(ds_id: int) -> None:
    """Copy one validated upload onto Expanse and unpack it there, then mark
    it ready for training -- or failed, with the reason."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM datasets WHERE id=%s", (ds_id,)).fetchone()
    name, uploader = row["name"], row["uploaded_by"]
    local_dir = local_upload_dir(ds_id, name)
    tar_path = local_dir.with_name(local_dir.name + ".tar")

    try:
        _repack(local_dir, tar_path)
        link = upload_links.create(tar_path)
        try:
            uuid = tapis_service.submit_upload(
                user_tokens.wait_for(uploader), source_url=link,
                name=f"fnas-lcd-upload-dataset-{ds_id}", uploaded_by=uploader)
            _set_dataset(ds_id, remote_job_uuid=uuid)
            # Once Tapis has copied the archive into the job's folder, neither
            # the link nor this container's copy is needed.
            tapis_service.watch(uuid, uploader, done=lambda s: s not in tapis_service.INPUT_STATES)
        finally:
            upload_links.revoke(link)
        tar_path.unlink(missing_ok=True)
        _finish_upload(ds_id, uuid, uploader)
    except Exception as e:  # noqa: BLE001
        _fail_upload(ds_id, name, uploader, f"Copying to Expanse failed: {e}")


# ── deleting uploads from Expanse ──────────────────────────────────────────
#
# A user's own login can't delete FNAS-LCD's files on Expanse either, so an
# upload's copy there is deleted by a job too: the upload app's delete mode,
# which deletes only a dataset uploaded by the same Tapis user.

def has_expanse_copy(row) -> bool:
    """Whether an upload still has a copy on Expanse to delete: it got there,
    and SDSC hasn't deleted it yet."""
    return tapis_service.dataset_job_uuid(row["remote_path"]) is not None and not is_expired(row)


def submit_delete(row, token: str) -> str:
    """Submit the job that deletes an upload's copy on Expanse, as the user
    whose token this is. Returns the job's id. The caller marks the dataset
    'deleting' and then calls follow_delete."""
    return tapis_service.submit_delete(
        token, upload_job_uuid=tapis_service.dataset_job_uuid(row["remote_path"]),
        name=f"fnas-lcd-delete-dataset-{row['id']}")


def follow_delete(ds_id: int, uuid: str, uploader: str) -> None:
    """Remove the dataset once its delete job finishes, in the background."""
    threading.Thread(target=_finish_delete, args=(ds_id, uuid, uploader),
                     daemon=True, name=f"delete-dataset-{ds_id}").start()


def _finish_delete(ds_id: int, uuid: str, uploader: str) -> None:
    """Wait for a delete job to end, then remove the dataset -- or mark it
    failed, with the reason, so its uploader can try again."""
    try:
        _run_to_end(uuid, uploader, "delete")
    except Exception as e:  # noqa: BLE001
        _set_dataset(ds_id, copy_status="failed",
                     copy_error=f"Deleting this dataset from Expanse failed, so it may still "
                                f"be there. Try deleting it again. ({e})")
        return
    with get_db() as conn:
        row = conn.execute("SELECT name FROM datasets WHERE id=%s", (ds_id,)).fetchone()
        conn.execute("DELETE FROM datasets WHERE id=%s", (ds_id,))
    if row:
        delete_local_files(ds_id, row["name"])


# ── restarts ───────────────────────────────────────────────────────────────

def _resume(row) -> None:
    uuid, uploader = row["remote_job_uuid"], row["uploaded_by"]
    if row["copy_status"] == "deleting":
        _finish_delete(row["id"], uuid, uploader)
        return
    if not uuid:
        _fail_upload(row["id"], row["name"], uploader, _INTERRUPTED)
        return
    try:
        # The archive's link died with the previous process. A job that
        # hadn't fetched it yet never will.
        state = tapis_service.watch(uuid, uploader, done=lambda s: True)
        if state in tapis_service.INPUT_STATES:
            try:
                tapis_service.cancel(user_tokens.wait_for(uploader), uuid)
            except Exception:  # noqa: BLE001 - it fails on its own without the file
                pass
            _fail_upload(row["id"], row["name"], uploader, _INTERRUPTED)
            return
        _finish_upload(row["id"], uuid, uploader)
    except Exception as e:  # noqa: BLE001
        _fail_upload(row["id"], row["name"], uploader, f"Copying to Expanse failed: {e}")


def resume_interrupted_copies() -> int:
    """At startup, pick up uploads that were being copied to Expanse or
    deleted from it when the service stopped. Returns how many there were.

    Their jobs outlive this process; their threads didn't. An upload whose
    job had already copied the archive still completes; one that hadn't
    fails, and its uploader is asked to upload again. Each waits for its
    uploader's next visit if needed (user_tokens)."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM datasets WHERE copy_status IN ('copying', 'deleting')"
        ).fetchall()
    for row in rows:
        threading.Thread(target=_resume, args=(dict(row),), daemon=True,
                         name=f"resume-dataset-{row['id']}").start()
    return len(rows)


# ── unfinished and failed uploads ──────────────────────────────────────────

_INTERRUPTED_CHECK = ("Checking the upload was interrupted by a service restart. "
                      "Please upload the dataset again.")


def forget_interrupted_uploads() -> None:
    """At startup: a restart wipes this container's disk, so uploads that
    were still arriving are gone (their upload page starts them over, if
    still open), and those being checked can't finish."""
    with get_db() as conn:
        conn.execute("DELETE FROM datasets WHERE copy_status = 'uploading'")
        rows = conn.execute("SELECT * FROM datasets WHERE copy_status = 'checking'").fetchall()
    for row in rows:
        _fail_upload(row["id"], row["name"], row["uploaded_by"], _INTERRUPTED_CHECK)


def delete_unfinished(ds_id: int, name: str) -> None:
    """Remove an upload whose archive never finished arriving."""
    delete_local_files(ds_id, name)
    with get_db() as conn:
        conn.execute("DELETE FROM datasets WHERE id = %s AND copy_status = 'uploading'", (ds_id,))


def remove_failed_uploads() -> None:
    """Remove uploads that failed more than settings.failed_upload_days ago.
    A failed delete has no failed_at, since its copy may still be on
    Expanse: it stays until its uploader deletes it again."""
    with get_db() as conn:
        conn.execute(
            "DELETE FROM datasets WHERE copy_status = 'failed' "
            "AND failed_at < now() - %s * interval '1 day' "
            "AND NOT EXISTS (SELECT 1 FROM jobs WHERE jobs.dataset_id = datasets.id)",
            (settings.failed_upload_days,),
        )


# ── lifetime on Expanse ────────────────────────────────────────────────────

def expires_on(row) -> datetime.date | None:
    """The day SDSC deletes an upload's copy on Expanse (see
    settings.dataset_lifetime_days), or None for datasets this service
    didn't copy there itself."""
    if not row["is_upload"] or tapis_service.dataset_job_uuid(row["remote_path"]) is None:
        return None
    created = datetime.date.fromisoformat(row["created_at"][:10])
    return created + datetime.timedelta(days=settings.dataset_lifetime_days)


def is_expired(row) -> bool:
    expires = expires_on(row)
    return expires is not None and expires <= datetime.date.today()


def usable_datasets(conn) -> list:
    """The datasets offered for search and training: ready, and not expired."""
    rows = conn.execute("SELECT * FROM datasets WHERE is_ready=1 ORDER BY id").fetchall()
    return [row for row in rows if not is_expired(row)]


def training_problem(row) -> str | None:
    """Why a dataset can't be trained on Expanse right now, or None."""
    if not row["is_ready"]:
        return f"Dataset '{row['name']}' is not ready for training."
    if tapis_service.dataset_job_uuid(row["remote_path"]) is None:
        # e.g. an upload copied to Pitzer by an earlier version of the service.
        return (f"Dataset '{row['name']}' isn't on Expanse, so it can't be trained there. "
                f"Please upload it again.")
    if is_expired(row):
        return (f"Dataset '{row['name']}' is no longer on Expanse: SDSC deletes uploads "
                f"{settings.dataset_lifetime_days} days after they were copied there. "
                f"Please upload it again.")
    return None
