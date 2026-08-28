"""Tapis account-connection routes (OAuth authorization_code flow).

Lets a browser session connect its own Tapis identity, so training jobs
submitted from that session run under the user's own account instead of the
shared service credential. Proven against the live `icicleai` tenant before
being wired in here: self-service client registration, a real browser login
redirect, and `POST /v3/oauth2/tokens` (not the singular `/token` the
discovery doc advertises -- that one 500s) all work as expected.
"""

from __future__ import annotations

from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import RedirectResponse

from ..config import settings
from ..database import get_db
from ..deps import apply_session_cookie, get_session

router = APIRouter(prefix="/auth/tapis")


def _redirect_uri(request: Request) -> str:
    # Must be byte-identical between the /login redirect and the /callback
    # exchange, and must match what's registered on the Tapis client.
    return str(request.base_url).rstrip("/") + "/auth/tapis/callback"


@router.get("/login")
def login(request: Request, response: Response, session: dict = Depends(get_session)):
    if not settings.tapis_client_id:
        raise HTTPException(500, "TAPIS_CLIENT_ID is not configured")

    params = {
        "client_id": settings.tapis_client_id,
        "response_type": "code",
        "redirect_uri": _redirect_uri(request),
    }
    redirect = RedirectResponse(
        f"{settings.tapis_base_url}/v3/oauth2/authorize?{urlencode(params)}"
    )
    return apply_session_cookie(response, redirect)


@router.get("/callback")
def callback(request: Request, code: str, response: Response,
             session: dict = Depends(get_session)):
    from ..services import tapis_service

    try:
        tapis_username = tapis_service.connect_user(code, _redirect_uri(request))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Tapis authorization failed: {e}")

    with get_db() as conn:
        conn.execute(
            "UPDATE sessions SET tapis_username=%s WHERE id=%s",
            (tapis_username, session["id"]),
        )

    return apply_session_cookie(response, RedirectResponse("/account"))


@router.post("/logout")
def logout(response: Response, session: dict = Depends(get_session)):
    # Clears which Tapis identity this session acts as; does not revoke the
    # stored refresh token itself, so reconnecting doesn't require Tapis to
    # grant authorization again.
    with get_db() as conn:
        conn.execute(
            "UPDATE sessions SET tapis_username=NULL WHERE id=%s",
            (session["id"],),
        )
    return apply_session_cookie(response, RedirectResponse("/account", status_code=303))
