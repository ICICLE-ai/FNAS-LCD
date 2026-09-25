"""Shared FastAPI dependencies.

Two ways a caller can be identified: the browser-session lookup used by the
Tapis OAuth flow, and a pass-through Tapis token the caller already holds
(e.g. injected by a hosting platform as an `X-Tapis-Token` cookie). A token,
when present, always wins and is validated with Tapis itself.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import threading
import time
from typing import Optional

from fastapi import Request, Response

from .config import settings
from .database import get_db

SESSION_COOKIE = "fnas_session"
TAPIS_TOKEN_COOKIE = "X-Tapis-Token"


class AuthRequired(Exception):
    """No usable Tapis identity where one is required (missing/invalid token)."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


# token hash -> (username, cache-until epoch seconds). In memory only, never
# persisted; keyed by a hash so raw tokens aren't held as dict keys.
_token_cache: dict[str, tuple[str, float]] = {}
_token_cache_lock = threading.Lock()
_CACHE_SECONDS = 60


def _token_expiry(token: str) -> Optional[float]:
    """The token's `exp` claim, read without verifying anything -- used only
    to avoid caching past expiry, never to trust the token."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return float(json.loads(base64.urlsafe_b64decode(payload))["exp"])
    except Exception:  # noqa: BLE001
        return None


def validate_tapis_token(token: str) -> str:
    """Return the Tapis username a token belongs to, or raise AuthRequired.

    Asks Tapis (`/v3/oauth2/userinfo`), so an expired or revoked token is
    rejected, not just a malformed one. Results are cached for up to a
    minute (never past the token's own expiry) to avoid a network call on
    every request.
    """
    key = hashlib.sha256(token.encode()).hexdigest()
    now = time.time()
    with _token_cache_lock:
        hit = _token_cache.get(key)
        if hit and hit[1] > now:
            return hit[0]

    from .services import tapis_service
    try:
        username = tapis_service.get_userinfo(token)
    except Exception as e:  # noqa: BLE001
        raise AuthRequired(f"Could not validate your Tapis token: {e}")

    until = now + _CACHE_SECONDS
    exp = _token_expiry(token)
    if exp is not None:
        until = min(until, exp)
    with _token_cache_lock:
        if len(_token_cache) > 1024:
            _token_cache.clear()
        _token_cache[key] = (username, until)
    return username


def get_tapis_token(request: Request) -> Optional[str]:
    """Read a Tapis access token the caller already has in hand, if present.

    This is a request-scoped credential, not one this service stores or
    refreshes -- valid only as long as the token itself is. Distinct from
    `get_session`/`fnas_session`, which tracks a browser across requests
    regardless of whether a Tapis credential is attached to it.
    """
    return request.cookies.get(TAPIS_TOKEN_COOKIE) or None


def require_login(request: Request) -> None:
    """Router-level gate: when REQUIRE_TAPIS_TOKEN is on, every request must
    carry a valid Tapis token. A no-op otherwise, so local development (no
    portal in front, no cookie) is unaffected."""
    if not settings.require_tapis_token:
        return
    token = get_tapis_token(request)
    if not token:
        raise AuthRequired("Log in through the ICICLE portal to use this service.")
    validate_tapis_token(token)


def get_session(request: Request, response: Response) -> dict:
    """Return the caller's identity as a session-shaped dict
    (`{"id": ..., "tapis_username": ...}`).

    If the request carries a Tapis token, that is the identity: it is
    validated with Tapis and no session row is created or touched. An
    invalid token raises AuthRequired rather than silently falling back to
    a browser session. Otherwise falls back to the OAuth browser session.

    The session cookie value is an opaque random id with nothing derivable
    from it -- a stolen cookie is only as powerful as a stolen `sessions`
    row, and revoking a session is a DELETE, not key rotation.
    """
    token = get_tapis_token(request)
    if token:
        return {"id": None, "tapis_username": validate_tapis_token(token)}

    session_id = request.cookies.get(SESSION_COOKIE)

    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE id=%s", (session_id,)
        ).fetchone() if session_id else None

        if row is None:
            session_id = secrets.token_urlsafe(32)
            conn.execute(
                "INSERT INTO sessions (id) VALUES (%s)", (session_id,)
            )
            response.set_cookie(
                SESSION_COOKIE, session_id,
                httponly=True, samesite="lax", max_age=60 * 60 * 24 * 365,
            )
            row = {"id": session_id, "tapis_username": None}
        else:
            conn.execute(
                "UPDATE sessions SET last_seen_at=to_char(now(), 'YYYY-MM-DD HH24:MI:SS') "
                "WHERE id=%s", (session_id,),
            )

    return dict(row)


def apply_session_cookie(dep_response: Response, route_response: Response) -> Response:
    """Copy any Set-Cookie set by `get_session` onto a response a route
    constructs and returns directly.

    FastAPI only merges the dependency-tree's shared response headers into
    responses *it* builds (e.g. returning a dict). A route that returns its
    own Response/RedirectResponse/TemplateResponse bypasses that merge
    entirely, so a session cookie set during dependency resolution would
    silently never reach the browser without this.
    """
    for header, value in dep_response.raw_headers:
        if header == b"set-cookie":
            route_response.raw_headers.append((header, value))
    return route_response
