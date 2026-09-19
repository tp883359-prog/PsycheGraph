"""Thread ownership: one browser session may only open its own threads.

The web UI identifies a session with a random `user_id` cookie. That cookie is
not an authentication credential for the API - the API keeps its own (optional)
token, see `auth.py` - but it *is* the ownership key the web layer validates on
every route:

    GET  /conversations/{thread_id}
    POST /conversations/{thread_id}/send-message
    GET  /conversations/{thread_id}/stream

Rules:

* a thread with an owner is only served to that owner (others get 404, so the
  existence of somebody else's thread is not confirmed);
* a thread without an owner (created through the API or Studio) is claimed by
  the first web session that opens it, and the claim is verified by re-reading
  the metadata so two racing sessions cannot both end up owning it.

This is deliberately small: it closes the cross-session read of a known thread
id without turning the demo into an account system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

logger = logging.getLogger(__name__)

OWNER_KEY = "user_id"
"""Thread metadata key that stores the owning session."""


@dataclass(frozen=True)
class AccessDecision:
    """Result of an ownership check.

    Attributes:
        allowed: Whether the session may use the thread.
        owner: Owner recorded on the thread, when there is one.
        claimed: True when this call claimed an unowned thread.
    """

    allowed: bool
    owner: str | None = None
    claimed: bool = False


def thread_owner(metadata: Any) -> str | None:
    """Read the owner from thread metadata.

    Args:
        metadata: Thread metadata mapping.

    Returns:
        The owner session id, or None when the thread has no owner.
    """
    if not isinstance(metadata, Mapping):
        return None
    raw = metadata.get(OWNER_KEY)
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def evaluate(metadata: Any, session_id: str) -> AccessDecision:
    """Decide whether a session may open a thread.

    Args:
        metadata: Current thread metadata.
        session_id: Session asking for access.

    Returns:
        The decision; `claimed` tells the caller to persist the ownership.
    """
    owner = thread_owner(metadata)
    if owner is None:
        return AccessDecision(allowed=True, owner=None, claimed=True)
    if owner == session_id:
        return AccessDecision(allowed=True, owner=owner, claimed=False)
    logger.warning("thread access denied for a different session")
    return AccessDecision(allowed=False, owner=owner, claimed=False)
