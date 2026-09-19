"""Server-side limits for the public demo: input size, rate, concurrency.

A single request costs 6-8 DeepSeek calls, so an open demo needs a cost guard
that does not depend on the browser. Everything here is enforced in the route
handler, before a run is queued, and the limits are configurable:

    WEB_MAX_INPUT_CHARS            default 4000
    WEB_RATE_LIMIT_PER_MINUTE      default 6
    WEB_RATE_LIMIT_PER_HOUR        default 40
    WEB_MAX_ACTIVE_RUNS_PER_SESSION default 1

Sessions are the signed-in-is-nothing `user_id` cookie: a random identifier the
server also validates against thread ownership (see `access.py`). Limits are
per session, not per IP, because there is no proxy layer in local development;
`docs/SECURITY_DEPLOYMENT.md` explains what a deployment should add on top.
"""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Mapping

DEFAULT_MAX_INPUT_CHARS = 4000
DEFAULT_PER_MINUTE = 6
DEFAULT_PER_HOUR = 40
DEFAULT_MAX_ACTIVE_RUNS = 1

INPUT_TOO_LONG = "输入太长了：本演示最多接受 {limit} 个字符，请缩短后重试。"
RATE_LIMITED = "本演示限制了请求频率（每分钟最多 {per_minute} 次，每小时 {per_hour} 次），请稍后再试。"
CONCURRENT_LIMIT = "同时只能有一条分析在运行，请等它结束后再发送。"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def load_limits() -> Mapping[str, int]:
    """Read the limits from the environment.

    Returns:
        Mapping with `max_input_chars`, `per_minute`, `per_hour` and
        `max_active_runs`.
    """
    return {
        "max_input_chars": _env_int("WEB_MAX_INPUT_CHARS", DEFAULT_MAX_INPUT_CHARS),
        "per_minute": _env_int("WEB_RATE_LIMIT_PER_MINUTE", DEFAULT_PER_MINUTE),
        "per_hour": _env_int("WEB_RATE_LIMIT_PER_HOUR", DEFAULT_PER_HOUR),
        "max_active_runs": _env_int(
            "WEB_MAX_ACTIVE_RUNS_PER_SESSION", DEFAULT_MAX_ACTIVE_RUNS
        ),
    }


@dataclass(frozen=True)
class LimitDecision:
    """Outcome of a limit check.

    Attributes:
        allowed: Whether the request may proceed.
        reason: `input_too_long`, `rate_limited` or `too_many_runs`.
        message: Safe, reader-facing explanation.
        status_code: HTTP status to answer with (429 for rate limits).
    """

    allowed: bool
    reason: str | None = None
    message: str = ""
    status_code: int = 200


def check_input_length(text: str, limit: int | None = None) -> LimitDecision:
    """Validate the length of one user message.

    Character count is used (not bytes), so a Chinese message is measured the
    way a reader would count it.

    Args:
        text: User message.
        limit: Maximum characters; the environment default when omitted.

    Returns:
        The decision.
    """
    resolved = limit if limit is not None else load_limits()["max_input_chars"]
    if len(text) > resolved:
        return LimitDecision(
            allowed=False,
            reason="input_too_long",
            message=INPUT_TOO_LONG.format(limit=resolved),
            status_code=413,
        )
    return LimitDecision(allowed=True)


class RateLimiter:
    """Sliding-window request limiter, per session.

    Windows are kept in memory: the demo runs a single application replica (see
    `docs/SECURITY_DEPLOYMENT.md`), and a deployment with several replicas must
    move this state to a shared store.
    """

    def __init__(
        self,
        *,
        per_minute: int | None = None,
        per_hour: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create a limiter.

        Args:
            per_minute: Requests allowed per rolling minute.
            per_hour: Requests allowed per rolling hour.
            clock: Monotonic clock (injectable for tests).
        """
        limits = load_limits()
        self.per_minute = per_minute if per_minute is not None else limits["per_minute"]
        self.per_hour = per_hour if per_hour is not None else limits["per_hour"]
        self._clock = clock
        self._events: dict[str, Deque[float]] = {}

    def _recent(self, session_id: str) -> Deque[float]:
        """Return the session's request timestamps within the hour window.

        One deque per session is pruned with the *hour* cutoff; the minute
        window is derived from it by counting. Pruning with the shorter window
        would silently throw away the history the hourly limit needs.

        Args:
            session_id: Session identifier.

        Returns:
            The surviving timestamps, oldest first.
        """
        events = self._events.setdefault(session_id, deque())
        cutoff = self._clock() - 3600.0
        while events and events[0] < cutoff:
            events.popleft()
        return events

    def check(self, session_id: str) -> LimitDecision:
        """Check whether a session may start another run.

        Args:
            session_id: Session identifier (the ownership cookie value).

        Returns:
            The decision; an accepted request is recorded immediately so that
            two concurrent submissions cannot both pass.
        """
        now = self._clock()
        events = self._recent(session_id)
        per_minute = sum(1 for stamp in events if stamp > now - 60.0)
        if per_minute >= self.per_minute or len(events) >= self.per_hour:
            return LimitDecision(
                allowed=False,
                reason="rate_limited",
                message=RATE_LIMITED.format(
                    per_minute=self.per_minute, per_hour=self.per_hour
                ),
                status_code=429,
            )
        events.append(now)
        return LimitDecision(allowed=True)

    def reset(self, session_id: str | None = None) -> None:
        """Forget recorded requests.

        Args:
            session_id: Session to reset; every session when omitted.
        """
        if session_id is None:
            self._events.clear()
            return
        self._events.pop(session_id, None)


LIMITER = RateLimiter()
"""Process-wide limiter used by the routes."""


def check_concurrency(
    session_id: str, active_runs: int, limit: int | None = None
) -> LimitDecision:
    """Check how many runs a session already has in flight.

    Args:
        session_id: Session identifier.
        active_runs: How many runs this session currently owns.
        limit: Maximum concurrent runs; the environment default when omitted.

    Returns:
        The decision.
    """
    resolved = limit if limit is not None else load_limits()["max_active_runs"]
    if active_runs >= resolved:
        return LimitDecision(
            allowed=False,
            reason="too_many_runs",
            message=CONCURRENT_LIMIT,
            status_code=429,
        )
    return LimitDecision(allowed=True)
