"""Shared FastAPI dependencies.

A caller is identified by the Tapis token the hosting platform passes in as
an `X-Tapis-Token` cookie (the ICICLE pod does, once the user has logged
in), validated with Tapis itself. Without one -- local development -- the
caller is anonymous and can look around but not upload datasets or submit
jobs.
"""

from __future__ import annotations

import hashlib
import threading
import time
from typing import Optional

from fastapi import Request

from .config import settings

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

    from .services import tapis_service, user_tokens
    try:
        username = tapis_service.get_userinfo(token)
    except Exception as e:  # noqa: BLE001
        raise AuthRequired(f"Could not validate your Tapis token: {e}")

    until = now + _CACHE_SECONDS
    exp = user_tokens.token_claims(token).get("exp")
    if exp is not None:
        until = min(until, exp)
    # Tapis work for this user (jobs, uploads) is done with their newest token.
    user_tokens.remember(username, token, exp)
    with _token_cache_lock:
        if len(_token_cache) > 1024:
            _token_cache.clear()
        _token_cache[key] = (username, until)
    return username


def get_tapis_token(request: Request) -> Optional[str]:
    """Read a Tapis access token the caller already has in hand, if present.

    This is a request-scoped credential, not one this service stores on disk
    or refreshes -- valid only as long as the token itself is.
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


def get_session(request: Request) -> dict:
    """Return the caller's identity, `{"tapis_username": ...}`: the user
    their Tapis token belongs to, or None if the request carries no token.
    An invalid token raises AuthRequired rather than counting as anonymous.
    """
    token = get_tapis_token(request)
    return {"tapis_username": validate_tapis_token(token) if token else None}
