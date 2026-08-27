"""Shared FastAPI dependencies.

Currently just the browser-session lookup used by the Tapis OAuth flow.
"""

from __future__ import annotations

import secrets

from fastapi import Request, Response

from .database import get_db

SESSION_COOKIE = "fnas_session"


def get_session(request: Request, response: Response) -> dict:
    """Return the caller's session row, creating one (and setting the cookie)
    if this is a new browser.

    The cookie value is an opaque random id with nothing derivable from it --
    a stolen cookie is only as powerful as a stolen `sessions` row, and
    revoking a session is a DELETE, not key rotation.
    """
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
