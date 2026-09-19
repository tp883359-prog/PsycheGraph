"""Route-level guardrails: health, static assets, limits, ownership.

Phase 10A added behaviour that only exists at the HTTP boundary, so it is
tested there: the readiness endpoint, the vendored asset allowlist, the input
and rate limits with their real status codes, and the rule that one browser
session cannot open another session's conversation.
"""

from __future__ import annotations

from typing import Any

import pytest
from starlette.testclient import TestClient

import react_agent.web.limits as limits_module
import react_agent.web.routes as routes_module
from react_agent.web.app import app
from react_agent.web.health import HealthReport
from react_agent.web.pending import REGISTRY
from tests.unit_tests.web_helpers import FakeClient, normal_sequence

THREAD_ID = "thread-guardrails"
SEND_URL = f"/conversations/{THREAD_ID}/send-message"
PAGE_URL = f"/conversations/{THREAD_ID}"


@pytest.fixture(autouse=True)
def clean_state(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Keep the registry, the limiter and the environment clean."""
    for name in (
        "WEB_MAX_INPUT_CHARS",
        "WEB_RATE_LIMIT_PER_MINUTE",
        "WEB_RATE_LIMIT_PER_HOUR",
        "WEB_MAX_ACTIVE_RUNS_PER_SESSION",
        "WEB_COOKIE_SECURE",
        "LANGGRAPH_DEMO_API_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    REGISTRY.clear()
    limits_module.LIMITER.reset()
    yield
    REGISTRY.clear()
    limits_module.LIMITER.reset()
    routes_module.set_client_instance(None)


@pytest.fixture()
def client() -> Any:
    """Return a browser-like test client (it keeps cookies)."""
    with TestClient(app) as test_client:
        yield test_client


def install(sequence: Any = None) -> FakeClient:
    """Inject a fake LangGraph client.

    Args:
        sequence: Event sequence to replay.

    Returns:
        The installed double.
    """
    fake = FakeClient(normal_sequence() if sequence is None else sequence)
    routes_module.set_client_instance(fake)
    return fake


# ------------------------------------------------------------------- health


def test_health_endpoint_returns_only_safe_fields(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        routes_module,
        "build_report",
        lambda: HealthReport(
            status="ok",
            graph_loaded=True,
            vectorstore_available=True,
            embedding_loaded=False,
        ),
    )

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "status": "ok",
        "graph_loaded": True,
        "vectorstore_available": True,
        "embedding_loaded": False,
    }
    text = response.text.lower()
    for forbidden in ("key", "token", "secret", "c:\\", "/users/"):
        assert forbidden not in text


def test_health_endpoint_reports_degraded_retrieval(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        routes_module,
        "build_report",
        lambda: HealthReport(
            status="degraded",
            graph_loaded=True,
            vectorstore_available=False,
            embedding_loaded=False,
        ),
    )

    payload = client.get("/health").json()

    assert payload["status"] == "degraded"
    assert payload["vectorstore_available"] is False


# ----------------------------------------------------------- static assets


@pytest.mark.parametrize("asset", ["htmx.min.js", "htmx-ext-sse.js"])
def test_vendored_assets_are_served_locally(client: Any, asset: str) -> None:
    response = client.get(f"/static/{asset}")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/javascript")
    assert len(response.content) > 1000


@pytest.mark.parametrize(
    "path",
    [
        "/static/auth.py",
        "/static/..%2Fauth.py",
        "/static/%2e%2e%2fauth.py",
        "/static/unknown.js",
        "/static/",
    ],
)
def test_static_route_never_serves_anything_else(client: Any, path: str) -> None:
    response = client.get(path)

    assert response.status_code == 404
    assert "auth.py" not in response.text


def test_pages_reference_the_vendored_assets_not_a_cdn(client: Any) -> None:
    install()
    body = client.get(PAGE_URL).text

    assert "/static/htmx.min.js" in body
    assert "/static/htmx-ext-sse.js" in body
    assert "unpkg.com" not in body
    assert "cdn.jsdelivr.net" not in body
    assert "fonts.googleapis.com" not in body


# --------------------------------------------------------------- input size


def test_oversized_message_is_rejected_with_413(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install()
    monkeypatch.setenv("WEB_MAX_INPUT_CHARS", "10")

    response = client.post(SEND_URL, data={"msg": "梦" * 11})

    assert response.status_code == 413
    assert "超长了" in response.text
    assert REGISTRY.take(THREAD_ID) is None
    assert fake.runs.calls == []


def test_message_exactly_at_the_limit_is_accepted(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    install()
    monkeypatch.setenv("WEB_MAX_INPUT_CHARS", "10")

    response = client.post(SEND_URL, data={"msg": "梦" * 10})

    assert response.status_code == 200
    assert "梦" * 10 in response.text


def test_empty_message_is_not_queued(client: Any) -> None:
    install()

    response = client.post(SEND_URL, data={"msg": "   "})

    assert response.status_code == 200
    assert "请输入内容" in response.text
    assert REGISTRY.take(THREAD_ID) is None


# -------------------------------------------------------------- rate limits


def test_rate_limit_answers_429_and_queues_nothing(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    install()
    # The limiter reads its windows from the environment when it is built, so a
    # test that changes the numbers replaces the process-wide instance.
    monkeypatch.setattr(
        limits_module,
        "LIMITER",
        limits_module.RateLimiter(per_minute=2, per_hour=99),
    )
    monkeypatch.setenv("WEB_MAX_ACTIVE_RUNS_PER_SESSION", "99")

    assert client.post(SEND_URL, data={"msg": "第一条"}).status_code == 200
    assert client.post(SEND_URL, data={"msg": "第二条"}).status_code == 200
    third = client.post(SEND_URL, data={"msg": "第三条"})

    assert third.status_code == 429
    assert "节奏太快" in third.text
    # Only the two accepted requests were queued; the third one never entered
    # the registry (the thread keeps the earliest message because the run has
    # not started in this test).
    pending = REGISTRY.take(THREAD_ID)
    assert pending is not None
    assert pending.text == "第一条"
    assert REGISTRY.take(THREAD_ID) is None


def test_concurrency_limit_is_enforced_server_side(client: Any) -> None:
    install()

    first = client.post(SEND_URL, data={"msg": "先来一条"})
    second = client.post(SEND_URL, data={"msg": "再来一条"})

    assert first.status_code == 200
    assert second.status_code == 429
    assert "一次只跑一条分析" in second.text
    # The queued message is still the first one: nothing was overwritten.
    pending = REGISTRY.take(THREAD_ID)
    assert pending is not None
    assert pending.text == "先来一条"


# --------------------------------------------------------------- ownership


def test_another_browser_cannot_open_a_claimed_conversation(client: Any) -> None:
    fake = install()
    assert client.post(SEND_URL, data={"msg": "我的私密问题"}).status_code == 200
    owner = fake.threads.metadata["user_id"]
    assert owner

    with TestClient(app) as other_browser:
        page = other_browser.get(PAGE_URL)
        send = other_browser.post(SEND_URL, data={"msg": "偷看"})
        leaked_cookie = other_browser.cookies.get("user_id")

    assert page.status_code == 404
    assert "找不到" in page.text
    assert "我的私密问题" not in page.text
    assert send.status_code == 404
    assert leaked_cookie is None


def test_owner_keeps_access_after_leaving_the_page(client: Any) -> None:
    install()
    assert client.get(PAGE_URL).status_code == 200
    assert client.get(PAGE_URL).status_code == 200
    assert client.get(PAGE_URL).status_code == 200


def test_claiming_is_written_to_the_thread_metadata(client: Any) -> None:
    fake = install()

    assert client.post(SEND_URL, data={"msg": "claim me"}).status_code == 200

    owner_updates = [u for u in fake.threads.updates if "user_id" in u]
    assert owner_updates, "the owner must be persisted on the thread"
    assert owner_updates[0]["user_id"] == client.cookies.get("user_id")


def test_direct_visit_gets_a_session_cookie(client: Any) -> None:
    install()

    response = client.get(PAGE_URL)

    assert response.status_code == 200
    cookie = response.headers.get("set-cookie", "")
    assert "user_id=" in cookie
    assert "httponly" in cookie.lower()
    assert "samesite=lax" in cookie.lower()
    # Plain HTTP (local development through TestClient): auto mode must not
    # force Secure, otherwise the browser would drop the cookie.
    assert "secure" not in cookie.lower()


def test_session_cookie_is_secure_behind_an_https_proxy(client: Any) -> None:
    install()

    response = client.get(PAGE_URL, headers={"X-Forwarded-Proto": "https"})

    assert response.status_code == 200
    assert "secure" in response.headers.get("set-cookie", "").lower()


def test_cookie_secure_can_be_forced_and_disabled(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    install()

    monkeypatch.setenv(routes_module.COOKIE_SECURE_ENV, "true")
    forced = client.get(PAGE_URL, headers={"X-Forwarded-Proto": "http"})
    assert "secure" in forced.headers.get("set-cookie", "").lower()

    monkeypatch.setenv(routes_module.COOKIE_SECURE_ENV, "false")
    relaxed = client.get(PAGE_URL, headers={"X-Forwarded-Proto": "https"})
    assert "secure" not in relaxed.headers.get("set-cookie", "").lower()


def test_pages_never_expose_the_api_token(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The browser gets HTML, never the shared bearer token (§11 of Phase 10B)."""
    monkeypatch.setenv("LANGGRAPH_DEMO_API_TOKEN", "phase10b-canary-token")
    install()

    page = client.get(PAGE_URL)
    posted = client.post(SEND_URL, data={"msg": "我梦见水。"})

    assert page.status_code == 200 and posted.status_code == 200
    for body in (page.text, posted.text):
        assert "phase10b-canary-token" not in body
        assert "Bearer" not in body


def test_landing_redirect_issues_the_same_cookie_attributes(client: Any) -> None:
    """The redirect is where the cookie is born; it must not be weaker."""
    plain = client.get("/", follow_redirects=False)
    cookie = plain.headers.get("set-cookie", "")

    assert plain.status_code == 302
    assert "user_id=" in cookie
    assert "httponly" in cookie.lower()
    assert "samesite=lax" in cookie.lower()
    assert "max-age=" in cookie.lower()
    assert "path=/" in cookie.lower()
    assert "secure" not in cookie.lower()


def test_landing_redirect_cookie_is_secure_behind_https_proxy(client: Any) -> None:
    response = client.get(
        "/new-thread", follow_redirects=False, headers={"X-Forwarded-Proto": "https"}
    )

    assert response.status_code == 302
    assert "secure" in response.headers.get("set-cookie", "").lower()


def test_unowned_thread_is_claimable_by_the_first_visitor(client: Any) -> None:
    fake = install()

    response = client.get(PAGE_URL)

    assert response.status_code == 200
    assert fake.threads.metadata.get("user_id")
