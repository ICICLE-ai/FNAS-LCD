"""Dataset service: upload extraction, validation, class detection."""

from __future__ import annotations

import os
import re
import shutil
import stat
import tarfile
import threading
import time
import zipfile
from pathlib import Path
from typing import BinaryIO

from ..config import settings

# Disk space left free when deciding whether an upload fits, so a dataset
# never fills the disk completely.
_FREE_SPACE_MARGIN = 1024 * 1024 * 1024  # 1 GB


class ArchiveRejected(ValueError):
    """An upload that can't be accepted; the message tells the user why."""


def detect_num_classes(data_dir: Path) -> int:
    """Count class subdirectories under data_dir/train/."""
    train_dir = data_dir / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Expected train/ subdirectory in {data_dir}")
    classes = sorted(
        d.name for d in train_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    if not classes:
        raise RuntimeError(f"No class subdirectories found in {train_dir}")
    return len(classes)


def validate_imagefolder(data_dir: Path) -> tuple[bool, str]:
    """Validate that data_dir looks like an ImageFolder dataset.

    Returns (is_valid, error_message).
    """
    if not data_dir.is_dir():
        return False, f"Directory not found: {data_dir}"

    train_dir = data_dir / "train"
    if not train_dir.is_dir():
        return False, f"No train/ subdirectory in {data_dir}"

    classes = [d for d in train_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not classes:
        return False, f"No class directories found in {train_dir}"

    # Check that at least one class has images
    has_images = False
    for cls_dir in classes:
        images = list(cls_dir.glob("*"))
        if any(f.suffix.lower() in (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
               for f in images):
            has_images = True
            break

    if not has_images:
        return False, "No image files found in class directories"

    return True, ""


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


def _is_macos_metadata(name: str) -> bool:
    """macOS bookkeeping entries, which hold no image data: Finder's __MACOSX/
    folder, AppleDouble "._" files and .DS_Store. Finder's Compress adds the
    first; macOS tar adds "._" files next to any file with extended
    attributes. Left in, a "._x.jpg" looks like an image and crashes training."""
    parts = [p for p in name.replace("\\", "/").split("/") if p]
    return bool(parts) and ("__MACOSX" in parts or parts[-1].startswith("._")
                            or parts[-1] == ".DS_Store")


def inspect_archive(archive: BinaryIO, filename: str, max_total: int) -> tuple[int, int]:
    """Check every entry of an uploaded archive without extracting anything.
    Returns the total size of its files once unpacked, and how many macOS
    metadata files will be skipped (see _is_macos_metadata).

    Rejects entries that could write outside the dataset folder (absolute
    paths, ".." components, symbolic or hard links) and special files such as
    devices: an image dataset only needs folders and regular files. Also
    rejects the archive as soon as its contents exceed `max_total` bytes, the
    disk space available for unpacking it.
    """
    total = 0
    skipped = 0
    archive.seek(0)

    def check(name: str, problem: str | None, size: int) -> None:
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

    if filename.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if _is_macos_metadata(info.filename):
                    skipped += not info.is_dir()
                    continue
                # Unix file type bits, when the zip records them.
                kind = stat.S_IFMT(info.external_attr >> 16)
                problem = _unsafe_path(info.filename)
                if not problem and kind == stat.S_IFLNK:
                    problem = "a link"
                elif not problem and kind not in (0, stat.S_IFREG, stat.S_IFDIR):
                    problem = "a special file"
                check(info.filename, problem, 0 if info.is_dir() else info.file_size)
    else:
        mode = "r:gz" if filename.endswith((".tar.gz", ".tgz")) else "r:"
        with tarfile.open(fileobj=archive, mode=mode) as tf:
            for member in tf:
                if _is_macos_metadata(member.name):
                    skipped += not member.isdir()
                    continue
                problem = _unsafe_path(member.name)
                if not problem and (member.issym() or member.islnk()):
                    problem = "a link"
                elif not problem and not (member.isreg() or member.isdir()):
                    problem = "a special file"
                check(member.name, problem, member.size if member.isreg() else 0)
    return total, skipped


def _extraction_filter(member: tarfile.TarInfo, dest_path: str) -> tarfile.TarInfo | None:
    """Skip macOS metadata; otherwise Python's "data" filter, which refuses
    anything that would land outside the destination -- a second line of
    defense behind inspect_archive."""
    if _is_macos_metadata(member.name):
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
                i for i in zf.infolist() if not _is_macos_metadata(i.filename)])
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


def delete_local_files(ds_id: int, name: str) -> None:
    """Remove whatever an upload left on this container's disk: its folder
    and the archive re-packed from it for copying (see stage_upload)."""
    folder = local_upload_dir(ds_id, name)
    shutil.rmtree(folder, ignore_errors=True)
    folder.with_name(folder.name + ".tar").unlink(missing_ok=True)


def add_uploaded_dataset(name: str, archive: BinaryIO, filename: str, uploaded_by: str) -> dict:
    """Full pipeline: extract, validate, register.

    `archive` is the uploaded file, open for reading (see extract_archive_to).
    `uploaded_by` is the uploader's Tapis username; only they may delete it.

    Returns dict with dataset info.
    """
    import psycopg

    safe_name = _sanitize_name(name)

    # Build a unique directory name
    from ..database import get_db
    try:
        with get_db() as conn:
            # Reserve an ID by inserting placeholder
            cursor = conn.execute(
                "INSERT INTO datasets (name, path, num_classes, is_upload, is_ready, description, uploaded_by) "
                "VALUES (%s, '', 0, 1, 0, 'Uploading...', %s) RETURNING id",
                (safe_name, uploaded_by),
            )
            ds_id = cursor.fetchone()["id"]
    except psycopg.errors.UniqueViolation:
        raise ValueError(f'A dataset named "{safe_name}" already exists. Please choose another name.')

    upload_dir = local_upload_dir(ds_id, safe_name)

    try:
        available = shutil.disk_usage(settings.uploads_dir).free - _FREE_SPACE_MARGIN
        _, skipped = inspect_archive(archive, filename, max_total=max(available, 0))
        extract_archive_to(upload_dir, archive, filename)

        valid, msg = validate_imagefolder(upload_dir)
        if not valid:
            # Cleanup
            shutil.rmtree(upload_dir, ignore_errors=True)
            with get_db() as conn:
                conn.execute("DELETE FROM datasets WHERE id = %s", (ds_id,))
            raise ValueError(f"Invalid dataset: {msg}")

        num_classes = detect_num_classes(upload_dir)
        description = f"Uploaded: {filename}"
        if skipped:
            description += f"; skipped {skipped} macOS metadata file{'s' if skipped > 1 else ''}"

        # With remote training on, the dataset becomes ready only once it has
        # been copied to the execution system (stage_upload); until then it
        # isn't offered for search or training.
        staged = settings.tapis_enabled
        with get_db() as conn:
            conn.execute(
                "UPDATE datasets SET name=%s, path=%s, num_classes=%s, is_ready=%s, description=%s, "
                "copy_status=%s WHERE id=%s",
                (safe_name, str(upload_dir), num_classes, 0 if staged else 1, description,
                 "copying" if staged else None, ds_id),
            )
        if staged:
            threading.Thread(target=stage_upload, args=(ds_id,), daemon=True,
                             name=f"stage-dataset-{ds_id}").start()

        return {
            "id": ds_id,
            "name": safe_name,
            "path": str(upload_dir),
            "num_classes": num_classes,
            "is_ready": not staged,
        }

    except Exception:
        # Cleanup on any error
        shutil.rmtree(upload_dir, ignore_errors=True)
        with get_db() as conn:
            conn.execute("DELETE FROM datasets WHERE id = %s", (ds_id,))
        raise


# ── copying uploads to the execution system ────────────────────────────────
#
# Training runs on the HPC system, which can't see this container's disk. So
# once an upload is validated here, stage_upload copies it there in the
# background: it re-packs the validated folder into a plain tar, uploads that
# with Tapis, and runs the small unpack app. Then the dataset is ready for
# training, exactly like a pre-configured one, and nothing of it is kept on
# this container's disk (which a pod restart wipes anyway).

_INTERRUPTED = ("Copying to the training system was interrupted by a service restart. "
                "Please upload the dataset again.")


def remote_upload_dir(ds_id: int, name: str) -> str:
    """Where an upload lives on the execution system."""
    from . import tapis_service
    return f"{tapis_service.uploads_base()}/{ds_id}_{name}"


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


def _fail_copy(ds_id: int, name: str, reason: str) -> None:
    from ..database import get_db
    from . import tapis_service

    with get_db() as conn:
        conn.execute(
            "UPDATE datasets SET copy_status='failed', copy_error=%s WHERE id=%s",
            (reason, ds_id),
        )
    delete_local_files(ds_id, name)
    try:
        tapis_service.delete_upload_dir(remote_upload_dir(ds_id, name))
    except Exception as e:  # noqa: BLE001 - deleting the dataset retries this
        print(f"WARNING: could not remove the remote folder of failed upload {ds_id}: {e}",
              flush=True)


def stage_upload(ds_id: int) -> None:
    """Copy one validated upload to the execution system and unpack it
    there, then mark it ready for training -- or failed, with the reason."""
    from ..database import get_db
    from . import tapis_service

    with get_db() as conn:
        row = conn.execute("SELECT * FROM datasets WHERE id=%s", (ds_id,)).fetchone()
    name = row["name"]
    local_dir = local_upload_dir(ds_id, name)
    tar_path = local_dir.with_name(local_dir.name + ".tar")

    try:
        remote_dir = remote_upload_dir(ds_id, name)
        _repack(local_dir, tar_path)
        tapis_service.make_dir(remote_dir)
        tapis_service.upload_file(tar_path, f"{remote_dir}/{tapis_service.UNPACK_ARCHIVE_NAME}")
        tar_path.unlink()

        uuid = tapis_service.submit_unpack(
            dataset_dir=remote_dir, name=f"fnas-lcd-unpack-dataset-{ds_id}",
            uploaded_by=row["uploaded_by"])
        while (state := tapis_service.get_status(uuid)) not in tapis_service.TERMINAL_STATES:
            time.sleep(settings.tapis_poll_seconds)
        if state != "FINISHED":
            detail = tapis_service.get_last_message(uuid) or state
            log = tapis_service.download_output(uuid, tapis_service.LOG_FILE)
            if log:
                detail += "\n" + log.decode(errors="replace")[-1000:]
            raise RuntimeError(f"the unpack job ({uuid}) ended {state}: {detail}")
    except Exception as e:  # noqa: BLE001
        _fail_copy(ds_id, name, f"Copying to the training system failed: {e}")
        return

    try:
        tapis_service.delete_upload_archive(remote_dir)
    except Exception as e:  # noqa: BLE001 - the dataset is usable regardless
        print(f"WARNING: could not remove the archive from the remote folder of upload {ds_id}: {e}",
              flush=True)

    with get_db() as conn:
        conn.execute(
            "UPDATE datasets SET remote_path=%s, path=%s, copy_status='ready', copy_error=NULL, "
            "is_ready=1 WHERE id=%s",
            (remote_dir, remote_dir, ds_id),
        )


def fail_interrupted_copies() -> int:
    """At startup, mark failed any upload still marked 'copying': the thread
    copying it died with the previous process (and on a pod, the disk
    holding its files too). Returns how many there were."""
    from ..database import get_db

    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, name FROM datasets WHERE copy_status='copying'"
        ).fetchall()
    for row in rows:
        threading.Thread(target=_fail_copy, args=(row["id"], row["name"], _INTERRUPTED),
                         daemon=True).start()
    return len(rows)
