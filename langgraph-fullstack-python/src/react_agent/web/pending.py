"""In-process registry for "one run per thread".

A run is created *inside* the SSE request (create and stream in one call), so
the message the browser posted has to wait for the stream request to pick it up.
This module holds that short-lived message and one flag per thread saying a run
is in flight.

The registry lives in the dev-server process and is deliberately tiny: no
persistence, no cross-process coordination. A restart simply means "no pending
message", which the stream endpoint reports as a closed stream.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class PendingMessage:
    """One user message waiting for its stream request.

    Attributes:
        text: The user's message.
        created_at: Monotonic submission time, for diagnostics.
        session_id: Owning session, used for the per-session concurrency limit.
    """

    text: str
    created_at: float
    session_id: str


class RunRegistry:
    """Track the pending message and the active run of each thread.

    This is process-local state on purpose: the demo runs a single application
    replica (`WEB_REPLICAS=1`). Moving to several replicas means moving this to
    a shared store such as Redis; `docs/SECURITY_DEPLOYMENT.md` says so.
    """

    def __init__(self) -> None:
        """Create an empty registry."""
        self._pending: dict[str, PendingMessage] = {}
        self._active: dict[str, PendingMessage] = {}

    def is_busy(self, thread_id: str) -> bool:
        """Report whether the thread already has a pending or running analysis.

        Args:
            thread_id: Thread to inspect.

        Returns:
            True when another run owns the thread.
        """
        return thread_id in self._pending or thread_id in self._active

    def session_load(self, session_id: str) -> int:
        """Count the threads one session has pending or running.

        Args:
            session_id: Session to inspect.

        Returns:
            Number of in-flight analyses of that session.
        """
        return sum(
            1
            for message in (*self._pending.values(), *self._active.values())
            if message.session_id == session_id
        )

    def submit(self, thread_id: str, message: str, session_id: str) -> bool:
        """Queue a message for the stream endpoint.

        Args:
            thread_id: Thread the message belongs to.
            message: User message text.
            session_id: Owning session.

        Returns:
            False when the thread is busy (the caller then shows a notice).
        """
        if self.is_busy(thread_id):
            return False
        self._pending[thread_id] = PendingMessage(
            text=message, created_at=time.time(), session_id=session_id
        )
        return True

    def take(self, thread_id: str) -> PendingMessage | None:
        """Take the queued message and mark the thread as running.

        Args:
            thread_id: Thread to take from.

        Returns:
            The pending message, or None when there is nothing to run.
        """
        pending = self._pending.pop(thread_id, None)
        if pending is None:
            return None
        self._active[thread_id] = pending
        return pending

    def release(self, thread_id: str) -> None:
        """Mark a thread as no longer running.

        Args:
            thread_id: Thread to release.
        """
        self._active.pop(thread_id, None)

    def clear(self) -> None:
        """Drop every entry (used by tests)."""
        self._pending.clear()
        self._active.clear()


REGISTRY = RunRegistry()
"""Process-wide registry used by the routes."""
