"""Each user's newest Tapis access token, kept in memory only.

Every Tapis action is taken with the user's own login: the token the ICICLE
pod hands this service as a cookie on each page the user opens (see
deps.validate_tapis_token, which records it here). Work that outlives a page
request -- watching a training job, copying an upload to Expanse -- uses the
newest token seen for that user.

Nothing is written to disk. When a user's token has expired and they haven't
opened a page since, that work waits (wait_for) and carries on the next time
they open FNAS-LCD. A restart forgets every token, so work resumed after one
waits the same way.
"""

from __future__ import annotations

import base64
import json
import threading
import time
from typing import Optional

# A token this close to expiring isn't handed out: a call made with it could
# fail halfway, e.g. while waiting for Tapis to answer.
_MIN_REMAINING_SECONDS = 120

# username -> (token, expiry in epoch seconds)
_tokens: dict[str, tuple[str, float]] = {}
_changed = threading.Condition()


def token_claims(token: str) -> dict:
    """The claims inside a Tapis access token (a JWT), read without checking
    its signature: Tapis itself checks the token on every call. Empty if the
    token can't be read."""
    try:
        payload = token.split(".")[1]
        return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except Exception:  # noqa: BLE001
        return {}


def remember(username: str, token: str, expires_at: Optional[float]) -> None:
    """Record a token Tapis has just confirmed belongs to `username`."""
    if not username or not token or expires_at is None:
        return
    with _changed:
        held = _tokens.get(username)
        if held is None or expires_at > held[1]:
            _tokens[username] = (token, expires_at)
            _changed.notify_all()


def current(username: Optional[str]) -> Optional[str]:
    """The user's newest token, if it is valid for a while yet."""
    if not username:
        return None
    with _changed:
        held = _tokens.get(username)
    if held and held[1] - time.time() > _MIN_REMAINING_SECONDS:
        return held[0]
    return None


def wait_for(username: Optional[str], stop: Optional[threading.Event] = None) -> Optional[str]:
    """Return a valid token for the user, waiting until they open a page if
    theirs has expired. Returns None if `stop` is set first, or if there is
    no user to wait for."""
    if not username:
        return None
    while True:
        token = current(username)
        if token:
            return token
        if stop is not None and stop.is_set():
            return None
        with _changed:
            # Woken by remember(); the timeout re-checks `stop` now and then.
            _changed.wait(timeout=5)
