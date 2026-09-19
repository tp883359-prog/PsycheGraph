"""Authentication for the agent server.

Two modes, chosen by environment:

* **Local demo (default).** No `LANGGRAPH_DEMO_API_TOKEN` is set, so every
  request is accepted and attributed to a single local identity. This is the
  Phase 1-9 behaviour and is fine on `localhost` only.
* **Shared instance.** When `LANGGRAPH_DEMO_API_TOKEN` is set, every API
  request must carry `Authorization: Bearer <token>`. The web app reads the
  same variable and forwards it on its in-process calls, so a browser user
  never sees the token and cannot reach `/threads`, `/runs`, `/store` or
  `/docs` without it.

What this deliberately is *not*: a user account system, resource-level
authorisation, or an OAuth flow. `docs/SECURITY_DEPLOYMENT.md` lists the routes
that must stay behind a proxy and what Phase 10B still has to solve (for
example per-owner filtering with `auth.on.threads.*` handlers if the raw API is
ever exposed).

The comparison is constant-time, an empty token never authorises, and nothing
about the token is ever logged.
"""

from __future__ import annotations

import hmac
import logging
import os

from langgraph_sdk import Auth
from langgraph_sdk.auth.exceptions import HTTPException

logger = logging.getLogger(__name__)

TOKEN_ENV = "LANGGRAPH_DEMO_API_TOKEN"
"""Environment variable holding the shared bearer token, when one is used."""

DEFAULT_USER = "default_user"
"""Identity used while no token is configured (local demo)."""

TOKEN_USER = "api_token_user"
"""Identity attributed to callers that present the shared token."""

auth = Auth()


def configured_token() -> str:
    """Return the configured bearer token.

    Returns:
        The token from the environment, or an empty string when unset.
    """
    return os.environ.get(TOKEN_ENV, "").strip()


def is_authorized(authorization: str | None) -> bool:
    """Check an `Authorization` header against the configured token.

    Args:
        authorization: Raw header value (`Bearer <token>`, or None/empty).

    Returns:
        True when the request may proceed.
    """
    expected = configured_token()
    if not expected:
        return True
    if not authorization:
        return False
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() != "bearer" or not value:
        return False
    return hmac.compare_digest(value, expected)


@auth.authenticate
async def authenticate(authorization: str) -> str:
    """Authenticate one API request.

    Args:
        authorization: `Authorization` header value supplied by the caller.

    Returns:
        The identity the LangGraph runtime attributes the request to.

    Raises:
        HTTPException: 401 when a token is configured and the header does not
            match it.
    """
    if is_authorized(authorization):
        return TOKEN_USER if configured_token() else DEFAULT_USER
    logger.warning("rejected an API request with a missing or invalid token")
    raise HTTPException(status_code=401, detail="invalid or missing bearer token")
