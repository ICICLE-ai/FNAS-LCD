"""The server side of the upload page's resumable uploads: the core of the
tus protocol (https://tus.io/protocols/resumable-upload), as tus-js-client
uses it. The uploads themselves are kept by services/resumable_uploads.py.

Only errors that may pass are answered 409 or 423, which tus-js-client
retries; every other refusal is final (400, 403, 404, 413, 415).
"""

from __future__ import annotations

import base64

import anyio
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from starlette.requests import ClientDisconnect

from ..config import settings
from ..deps import get_session
from ..services import dataset_service, resumable_uploads

router = APIRouter(prefix="/api/uploads", include_in_schema=False)
_TUS = {"Tus-Resumable": "1.0.0"}


def _uploader(session: dict) -> str:
    # Uploads are attributed, like jobs: only the uploader may delete one.
    if not session.get("tapis_username"):
        raise HTTPException(403, "Log in through the ICICLE portal before uploading a dataset.")
    return session["tapis_username"]


def _metadata(header: str) -> dict[str, str]:
    """Upload-Metadata: comma-separated "key base64(value)" pairs."""
    pairs = [item.strip().split(" ", 1) for item in header.split(",") if item.strip()]
    try:
        return {p[0]: base64.b64decode(p[1]).decode() if len(p) > 1 else "" for p in pairs}
    except ValueError:
        raise HTTPException(400, "Invalid Upload-Metadata") from None


@router.post("")
def start_upload(request: Request, session: dict = Depends(get_session)):
    """Start an upload. The response's Location is where its pieces go."""
    owner = _uploader(session)
    meta = _metadata(request.headers.get("Upload-Metadata", ""))
    name, filename = meta.get("name", "").strip(), meta.get("filename", "")
    length = request.headers.get("Upload-Length", "")
    if not length.isdigit():
        raise HTTPException(400, "Upload-Length is missing")
    length = int(length)
    if not name:
        raise HTTPException(400, "Please enter a name for the dataset.")
    if not filename.endswith(dataset_service.ARCHIVE_TYPES):
        raise HTTPException(400, f"Unsupported format. Allowed: "
                                 f"{', '.join(dataset_service.ARCHIVE_TYPES)}")
    if length == 0:
        raise HTTPException(400, "This file is empty.")
    if length > settings.max_upload_bytes:
        raise HTTPException(413, f"File too large: the maximum upload size is "
                                 f"{dataset_service.format_size(settings.max_upload_bytes)}.")
    try:
        upload = resumable_uploads.start(name, filename, length, owner)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    return Response(status_code=201,
                    headers={**_TUS, "Location": f"{router.prefix}/{upload.ds_id}"})


@router.head("/{ds_id}")
def upload_offset(ds_id: int, session: dict = Depends(get_session)):
    """How much of an upload has arrived, so the page can resume it."""
    upload = resumable_uploads.find(ds_id, _uploader(session))
    if upload is None:
        raise HTTPException(404, "No such upload")
    return Response(headers={**_TUS, "Upload-Offset": str(upload.offset()),
                             "Upload-Length": str(upload.length), "Cache-Control": "no-store"})


@router.patch("/{ds_id}")
async def upload_piece(ds_id: int, request: Request, session: dict = Depends(get_session)):
    """Append the next piece of an upload, which must start where the upload stands."""
    owner = _uploader(session)
    if request.headers.get("Content-Type") != "application/offset+octet-stream":
        raise HTTPException(415, "Expected Content-Type: application/offset+octet-stream")
    try:
        upload = resumable_uploads.claim(ds_id, owner)
    except resumable_uploads.Busy:
        raise HTTPException(423, "Another piece of this upload is still arriving") from None
    except LookupError:
        raise HTTPException(404, "This upload no longer exists. Choose the file again to "
                                 "start over.") from None
    try:
        offset = upload.offset()
        if request.headers.get("Upload-Offset") != str(offset):
            raise HTTPException(409, f"The upload stands at byte {offset}")
        room = upload.length - offset
        async with await anyio.open_file(upload.path, "ab") as f:
            async for chunk in request.stream():
                if len(chunk) > room:
                    raise HTTPException(400, "More data than the upload's length")
                await f.write(chunk)
                room -= len(chunk)
    except ClientDisconnect:
        pass  # what arrived is kept; the page resumes from there
    finally:
        await anyio.to_thread.run_sync(resumable_uploads.release, upload)
    return Response(status_code=204, headers={**_TUS, "Upload-Offset": str(upload.offset())})
