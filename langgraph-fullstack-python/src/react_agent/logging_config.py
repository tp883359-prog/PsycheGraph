"""Logging configuration for the agent and the web layer.

Production default is INFO; `LOG_LEVEL` (or an explicit argument) overrides it.
Only the `react_agent` logger hierarchy is configured: the LangGraph server
keeps its own formatting, and setting a level here means our records appear at
the right verbosity without fighting the server's handlers.

What may be logged: run/thread identifiers (truncated where they are long),
node names, statuses, durations, counts, exception class names.
What must never be logged: API keys, prompts, full user text, full evidence
chunks, database URIs - all of that is redacted by simply never passing it.
"""

from __future__ import annotations

import logging
import os

LOGGER_NAME = "react_agent"
DEFAULT_LEVEL = "INFO"
VALID_LEVELS: tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

_configured = False

NOISY_THIRD_PARTY: tuple[str, ...] = (
    "httpx",
    "httpcore",
    "huggingface_hub",
    "urllib3",
    "filelock",
)
"""Libraries whose INFO chatter would drown the agent's own records."""


def resolve_level(level: str | None = None) -> int:
    """Resolve a log level name to a logging constant.

    Args:
        level: Explicit level name; the `LOG_LEVEL` environment variable is
            used when omitted, then `INFO`.

    Returns:
        The logging level constant. Unknown names fall back to INFO.
    """
    raw = (level or os.environ.get("LOG_LEVEL") or DEFAULT_LEVEL).strip().upper()
    if raw not in VALID_LEVELS:
        return logging.INFO
    return int(getattr(logging, raw))


def configure_logging(level: str | None = None) -> int:
    """Apply the log level to the project's logger hierarchy.

    Args:
        level: Optional level name overriding `LOG_LEVEL`.

    Returns:
        The resolved numeric level.
    """
    global _configured  # noqa: PLW0603 - one process-wide configuration
    resolved = resolve_level(level)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(resolved)
    if not _configured:
        # Keep records visible even when nothing configured the root logger
        # (scripts, prewarm, tests run outside the LangGraph server).
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=resolved,
                format="%(asctime)s %(levelname)s %(name)s %(message)s",
            )
        for name in NOISY_THIRD_PARTY:
            logging.getLogger(name).setLevel(max(resolved, logging.WARNING))
        _configured = True
    return resolved


def short_id(value: str | None, length: int = 8) -> str:
    """Shorten an identifier for logs.

    Args:
        value: Identifier such as a run or thread id.
        length: Characters to keep.

    Returns:
        The shortened identifier, or `-` when missing.
    """
    if not value:
        return "-"
    return value[:length]
