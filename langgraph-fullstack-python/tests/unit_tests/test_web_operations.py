"""Phase 10A operations tests: health, limits, ownership, auth, logging.

These cover the production-hardening modules added on top of the Phase 9 UI:
`web/health.py`, `web/limits.py`, `web/access.py`, `react_agent/auth.py` and
`react_agent/logging_config.py`. No model, no network, no server.
"""

from __future__ import annotations

import types
from typing import Any

import pytest

import react_agent.auth as auth_module
import react_agent.logging_config as logging_module
import react_agent.web.access as access_module
import react_agent.web.health as health_module
import react_agent.web.limits as limits_module


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's own environment variables out of the tests."""
    for name in (
        "LOG_LEVEL",
        auth_module.TOKEN_ENV,
        "WEB_MAX_INPUT_CHARS",
        "WEB_RATE_LIMIT_PER_MINUTE",
        "WEB_RATE_LIMIT_PER_HOUR",
        "WEB_MAX_ACTIVE_RUNS_PER_SESSION",
    ):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# Health report
# ---------------------------------------------------------------------------


def test_health_report_has_no_secrets_and_a_stable_shape() -> None:
    report = health_module.build_report(
        graph_loader=lambda: True,
        index_checker=lambda: True,
        embedding_checker=lambda: 1,
    )
    payload = report.as_dict()

    assert set(payload) == {
        "status",
        "graph_loaded",
        "vectorstore_available",
        "embedding_loaded",
    }
    assert payload["status"] == "ok"
    assert payload["graph_loaded"] is True
    assert payload["vectorstore_available"] is True
    assert payload["embedding_loaded"] is True
    # Nothing that could leak configuration or credentials.
    text = str(payload).lower()
    for forbidden in ("key", "token", "path", "secret", "/users/", "c:\\"):
        assert forbidden not in text


def test_health_report_degrades_when_the_index_or_model_is_missing() -> None:
    report = health_module.build_report(
        graph_loader=lambda: True,
        index_checker=lambda: False,
        embedding_checker=lambda: False,
    )

    assert report.status == "degraded"
    assert report.as_dict()["vectorstore_available"] is False
    assert report.as_dict()["embedding_loaded"] is False


def test_health_report_is_unavailable_without_the_graph() -> None:
    report = health_module.build_report(
        graph_loader=lambda: False,
        index_checker=lambda: True,
        embedding_checker=lambda: False,
    )

    assert report.status == "unavailable"


def test_health_report_survives_failing_checkers() -> None:
    def boom() -> bool:
        raise RuntimeError("import exploded")

    report = health_module.build_report(
        graph_loader=boom,
        index_checker=boom,
        embedding_checker=boom,
    )

    assert report.status == "unavailable"
    assert report.graph_loaded is False
    assert report.vectorstore_available is False


def test_embedding_loaded_follows_the_lru_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLoader:
        """Shaped like the `lru_cache` wrapper around `get_embeddings`."""

        def __init__(self, size: int) -> None:
            self.size = size

        def cache_info(self) -> Any:
            return types.SimpleNamespace(currsize=self.size)

    monkeypatch.setattr(health_module, "get_embeddings", FakeLoader(0))
    assert health_module.embedding_is_loaded() is False

    monkeypatch.setattr(health_module, "get_embeddings", FakeLoader(1))
    assert health_module.embedding_is_loaded() is True
    assert health_module.EMBEDDING_CACHE_HIT == 1


# ---------------------------------------------------------------------------
# Input limits
# ---------------------------------------------------------------------------


def test_input_limit_counts_unicode_characters() -> None:
    limit = 10
    exactly = "梦" * limit  # one character per dream
    over = "梦" * (limit + 1)

    assert limits_module.check_input_length(exactly, limit=limit).allowed is True
    decision = limits_module.check_input_length(over, limit=limit)
    assert decision.allowed is False
    assert decision.status_code == 413
    assert "10" in decision.message


def test_input_limit_uses_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEB_MAX_INPUT_CHARS", "5")

    assert limits_module.check_input_length("12345").allowed is True
    assert limits_module.check_input_length("123456").allowed is False
    assert limits_module.load_limits()["max_input_chars"] == 5


def test_input_limit_ignores_a_broken_environment_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEB_MAX_INPUT_CHARS", "not-a-number")

    assert (
        limits_module.check_input_length(
            "x" * limits_module.DEFAULT_MAX_INPUT_CHARS
        ).allowed
        is True
    )


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class FakeClock:
    """A clock the test controls, in seconds."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_rate_limiter_enforces_the_minute_window() -> None:
    clock = FakeClock()
    limiter = limits_module.RateLimiter(per_minute=3, per_hour=100, clock=clock)

    for _ in range(3):
        assert limiter.check("session-a").allowed is True
    blocked = limiter.check("session-a")
    assert blocked.allowed is False
    assert blocked.status_code == 429

    clock.advance(61)
    assert limiter.check("session-a").allowed is True


def test_rate_limiter_enforces_the_hour_window() -> None:
    clock = FakeClock()
    limiter = limits_module.RateLimiter(per_minute=100, per_hour=2, clock=clock)

    assert limiter.check("s").allowed is True
    clock.advance(120)  # outside the minute window, inside the hour
    assert limiter.check("s").allowed is True
    clock.advance(120)
    blocked = limiter.check("s")
    assert blocked.allowed is False
    assert blocked.status_code == 429

    clock.advance(3601)  # the hour window has now forgotten both requests
    assert limiter.check("s").allowed is True


def test_rate_limiter_is_per_session() -> None:
    clock = FakeClock()
    limiter = limits_module.RateLimiter(per_minute=1, per_hour=10, clock=clock)

    assert limiter.check("session-a").allowed is True
    assert limiter.check("session-a").allowed is False
    assert limiter.check("session-b").allowed is True


def test_rejected_requests_do_not_extend_the_window() -> None:
    clock = FakeClock()
    limiter = limits_module.RateLimiter(per_minute=1, per_hour=10, clock=clock)

    assert limiter.check("s").allowed is True
    for _ in range(5):
        assert limiter.check("s").allowed is False
    clock.advance(61)
    # The three extra rejections were not recorded, so the window has room.
    assert limiter.check("s").allowed is True


def test_rate_limiter_reset_clears_state() -> None:
    limiter = limits_module.RateLimiter(per_minute=1, per_hour=1)
    assert limiter.check("s").allowed is True
    limiter.reset()
    assert limiter.check("s").allowed is True


def test_concurrency_limit() -> None:
    assert limits_module.check_concurrency("s", 0, limit=1).allowed is True
    busy = limits_module.check_concurrency("s", 1, limit=1)
    assert busy.allowed is False
    assert busy.status_code == 429


# ---------------------------------------------------------------------------
# Thread ownership
# ---------------------------------------------------------------------------


def test_unowned_thread_is_claimed_by_the_first_session() -> None:
    decision = access_module.evaluate(None, "session-a")

    assert decision.allowed is True
    assert decision.claimed is True
    assert decision.owner is None


def test_owner_can_return_and_a_foreign_session_is_denied() -> None:
    metadata = {access_module.OWNER_KEY: "session-a"}

    own = access_module.evaluate(metadata, "session-a")
    assert own.allowed is True
    assert own.claimed is False

    intruder = access_module.evaluate(metadata, "session-b")
    assert intruder.allowed is False
    assert intruder.owner == "session-a"


def test_access_tolerates_garbage_metadata() -> None:
    assert access_module.evaluate("not-a-mapping", "session-a").allowed is True
    assert access_module.evaluate({"user_id": None}, "session-a").claimed is True


# ---------------------------------------------------------------------------
# API authentication
# ---------------------------------------------------------------------------


def test_local_mode_accepts_everything_and_uses_the_local_identity() -> None:
    assert auth_module.configured_token() == ""
    assert auth_module.is_authorized(None) is True
    assert auth_module.is_authorized("Bearer whatever") is True


def test_blank_token_in_the_environment_stays_in_local_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(auth_module.TOKEN_ENV, "   ")

    assert auth_module.configured_token() == ""
    assert auth_module.is_authorized(None) is True


@pytest.mark.parametrize(
    ("token", "header", "expected"),
    [
        ("s3cret", "Bearer s3cret", True),
        ("s3cret", "bearer s3cret", True),
        ("s3cret", "Bearer wrong", False),
        ("s3cret", "s3cret", False),
        ("s3cret", "Basic s3cret", False),
        ("s3cret", "Bearer ", False),
        ("s3cret", "", False),
    ],
)
def test_token_mode_requires_a_matching_bearer_header(
    monkeypatch: pytest.MonkeyPatch, token: str, header: str, expected: bool
) -> None:
    monkeypatch.setenv(auth_module.TOKEN_ENV, token)

    assert auth_module.is_authorized(header) is expected


def test_authenticate_returns_the_identity_and_rejects_bad_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    assert asyncio.run(auth_module.authenticate("")) == auth_module.DEFAULT_USER

    monkeypatch.setenv(auth_module.TOKEN_ENV, "s3cret")
    assert (
        asyncio.run(auth_module.authenticate("Bearer s3cret")) == auth_module.TOKEN_USER
    )
    with pytest.raises(Exception) as excinfo:
        asyncio.run(auth_module.authenticate("Bearer wrong"))
    assert "401" in str(excinfo.value) or "invalid" in str(excinfo.value)


def test_rejected_tokens_are_never_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import asyncio

    monkeypatch.setenv(auth_module.TOKEN_ENV, "s3cret")

    with caplog.at_level("WARNING", logger="react_agent.auth"):
        with pytest.raises(Exception):
            asyncio.run(auth_module.authenticate("Bearer wrong-token-value"))

    assert "rejected an API request" in caplog.text
    assert "wrong-token-value" not in caplog.text
    assert "s3cret" not in caplog.text


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------


def test_resolve_level_falls_back_to_info(monkeypatch: pytest.MonkeyPatch) -> None:
    assert logging_module.resolve_level("debug") == logging_module.logging.DEBUG
    assert logging_module.resolve_level("WARNING") == logging_module.logging.WARNING
    assert logging_module.resolve_level("nonsense") == logging_module.logging.INFO
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    assert logging_module.resolve_level() == logging_module.logging.ERROR
    monkeypatch.setenv("LOG_LEVEL", "")
    assert logging_module.resolve_level() == logging_module.logging.INFO


def test_configure_logging_sets_the_project_logger() -> None:
    level = logging_module.configure_logging("DEBUG")

    assert level == logging_module.logging.DEBUG
    assert logging_module.logging.getLogger(logging_module.LOGGER_NAME).level == level
    # Third-party chatter stays at WARNING or above whatever the level is.
    noisy = logging_module.logging.getLogger("httpx")
    assert noisy.level >= logging_module.logging.WARNING


def test_short_id_keeps_logs_readable() -> None:
    uuid_like = "6b0f0f7e-1f5b-4a20-9c26-2f9a4a1f9a11"

    assert logging_module.short_id(uuid_like) == "6b0f0f7e"
    assert logging_module.short_id(uuid_like, 4) == "6b0f"
    assert logging_module.short_id("abc") == "abc"


def test_log_records_do_not_contain_the_full_thread_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging as std_logging

    logger = std_logging.getLogger("react_agent.web.routes")
    thread_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

    with caplog.at_level(std_logging.INFO, logger="react_agent.web.routes"):
        logger.info(
            "run queued thread=%s chars=%d", logging_module.short_id(thread_id), 12
        )

    assert "run queued thread=aaaaaaaa chars=12" in caplog.text
    assert thread_id not in caplog.text
