"""Dataset service: upload extraction, validation, class detection."""

from __future__ import annotations

import os
import re
import shutil
import stat
import tarfile
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


def add_uploaded_dataset(name: str, archive: BinaryIO, filename: str) -> dict:
    """Full pipeline: extract, validate, register.

    `archive` is the uploaded file, open for reading (see extract_archive_to).

    Returns dict with dataset info.
    """
    safe_name = _sanitize_name(name)

    # Build a unique directory name
    from ..database import get_db
    with get_db() as conn:
        # Reserve an ID by inserting placeholder
        cursor = conn.execute(
            "INSERT INTO datasets (name, path, num_classes, is_upload, is_ready, description) "
            "VALUES (%s, '', 0, 1, 0, 'Uploading...') RETURNING id",
            (safe_name,),
        )
        ds_id = cursor.fetchone()["id"]

    upload_dir = settings.uploads_dir / f"{ds_id}_{safe_name}"

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

        with get_db() as conn:
            conn.execute(
                "UPDATE datasets SET name=%s, path=%s, num_classes=%s, is_ready=%s, description=%s WHERE id=%s",
                (safe_name, str(upload_dir), num_classes, 1, description, ds_id),
            )

        return {
            "id": ds_id,
            "name": safe_name,
            "path": str(upload_dir),
            "num_classes": num_classes,
            "is_ready": True,
        }

    except Exception:
        # Cleanup on any error
        shutil.rmtree(upload_dir, ignore_errors=True)
        with get_db() as conn:
            conn.execute("DELETE FROM datasets WHERE id = %s", (ds_id,))
        raise
