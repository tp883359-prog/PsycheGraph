"""HTTP routes of the web UI.

The `langgraph dev` server loads the FastHTML app through
`src/react_agent/app.py`, which re-exports `react_agent.web.app.app`. Routes
live here so the app module stays a thin assembly file.

Endpoints:

    GET  /                          -> redirect to a fresh conversation
    GET  /new-thread                -> redirect to a fresh conversation
    GET  /conversations/{thread_id} -> the three-column chat page
    POST /conversations/{thread_id}/send-message
                                    -> user bubble + placeholder + SSE hub
    GET  /conversations/{thread_id}/stream
                                    -> the SSE stream of one run
    GET  /evaluation                -> System / Evaluation page
    GET  /health                    -> readiness JSON (no secrets)

`send-message` and `stream` are separate on purpose: EventSource can only issue
a GET, so the message is queued in-process by `send-message` and picked up by
the stream request, which creates *and* streams the run in one call. That
ordering removes the classic race where a run finishes before the browser joins
its stream.

Every conversation route validates thread ownership (`access.py`) and every
run passes through the server-side limits (`limits.py`): input length, per
session rate and a per session concurrency cap of one. The limits do not rely
on the browser.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Mapping, Sequence

from fasthtml.common import Div, P
from starlette.requests import Request
from starlette.responses import RedirectResponse, StreamingResponse

from react_agent.logging_config import configure_logging, short_id
from react_agent.web import access, fragments, limits
from react_agent.web.components import (
    assistant_placeholder,
    chat_bubble,
    composer,
    conversation_columns,
    evidence_body,
    page,
    sse_hub,
)
from react_agent.web.evaluation import evaluation_page, load_evaluation_summary
from react_agent.web.health import build_report
from react_agent.web.pending import REGISTRY
from react_agent.web.streaming import run_workflow_events
from react_agent.web.workflow import workflow_panel

configure_logging()
logger = logging.getLogger(__name__)

SSE_HEADERS: Mapping[str, str] = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}
"""Headers that keep proxies from buffering the event stream."""

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
"""Directory holding the vendored htmx files (no `Path.cwd()` involved)."""

ALLOWED_STATIC_ASSETS: frozenset[str] = frozenset({"htmx.min.js", "htmx-ext-sse.js"})
"""Explicit allowlist: the route never serves arbitrary files."""

BUSY_MESSAGE = "上一条分析还在进行中，请等它结束后再发送。"
"""Shown when a thread already has a pending or running analysis."""

EMPTY_MESSAGE = "请输入内容后再发送。"
"""Shown when the composer submits an empty message."""

TITLE_CHARS = 24
"""How much of the first user message becomes the thread title."""

_CLIENT: Any = None


def get_client_instance() -> Any:
    """Return the process-wide LangGraph SDK client.

    Inside a LangGraph server `get_client()` connects in-process, so the custom
    HTTP app talks to the runtime without a network hop. When the deployment
    sets `LANGGRAPH_DEMO_API_TOKEN`, the same token is forwarded on those
    in-process calls so the browser never has to know it.

    Returns:
        The SDK client.
    """
    global _CLIENT  # noqa: PLW0603 - one client per server process
    if _CLIENT is None:
        from langgraph_sdk import get_client

        from react_agent.auth import TOKEN_ENV, configured_token

        token = configured_token()
        headers = {"Authorization": f"Bearer {token}"} if token else None
        try:
            _CLIENT = get_client(headers=headers)
        except Exception as exc:  # noqa: BLE001 - fall back to the plain client
            logger.warning(
                "in-process client without %s failed (%s); retrying plainly",
                TOKEN_ENV,
                type(exc).__name__,
            )
            _CLIENT = get_client(headers=None)
    return _CLIENT


def set_client_instance(client: Any) -> None:
    """Override the client (used by tests).

    Args:
        client: Replacement client, or None to fall back to the SDK.
    """
    global _CLIENT  # noqa: PLW0603
    _CLIENT = client


def user_id_of(request: Request) -> str:
    """Return the cookie-based user id of a request.

    Args:
        request: Incoming request.

    Returns:
        The existing id, or a fresh uuid (the caller sets the cookie).
    """
    return str(request.cookies.get("user_id") or uuid.uuid4())


COOKIE_SECURE_ENV = "WEB_COOKIE_SECURE"
"""`auto` (default) marks the cookie `Secure` when the request arrived over
HTTPS (directly or via `X-Forwarded-Proto` from the reverse proxy). `true`/
`false` force the flag; forcing it on plain-HTTP local development would make
the browser drop the cookie."""


def cookie_is_secure(request: Request) -> bool:
    """Decide whether the session cookie may carry the `Secure` flag.

    Args:
        request: Incoming request.

    Returns:
        True when the cookie should be HTTPS-only.
    """
    forced = os.environ.get(COOKIE_SECURE_ENV, "auto").strip().lower()
    if forced in {"1", "true", "yes", "on"}:
        return True
    if forced in {"0", "false", "no", "off"}:
        return False
    forwarded = request.headers.get("x-forwarded-proto", "")
    scheme = forwarded.split(",")[0].strip() or request.url.scheme
    return scheme == "https"


def session_cookie_value(request: Request, user_id: str) -> str | None:
    """Return the `Set-Cookie` value for a request that has no session yet.

    Args:
        request: Incoming request.
        user_id: Value resolved by `user_id_of`.

    Returns:
        The header value, or None when the browser already sent a cookie.
    """
    if request.cookies.get("user_id"):
        return None
    value = f"user_id={user_id}; Path=/; HttpOnly; SameSite=lax; Max-Age=31536000"
    if cookie_is_secure(request):
        value += "; Secure"
    return value


def session_cookie(request: Request, user_id: str) -> list[Any]:
    """Build the `Set-Cookie` header that pins a session to this browser.

    Every page that can be opened directly (a bookmark, a shared link) issues
    the cookie when the request did not carry one. Without this, a visitor who
    opens `/conversations/<id>` before ever seeing `/` would claim the thread
    with an id the browser never stored, and the next page load would look like
    somebody else's session.

    Args:
        request: Incoming request.
        user_id: Value resolved by `user_id_of`.

    Returns:
        A one-element header list for FastHTML responses, or an empty list.
        The value is an opaque uuid: it is not a login.
    """
    value = session_cookie_value(request, user_id)
    if value is None:
        return []
    from fasthtml.common import HttpHeader

    return [HttpHeader("set-cookie", value)]


def new_conversation_redirect(request: Request) -> Any:
    """Build the redirect that starts a fresh conversation.

    This is where a browser receives its session cookie for the first time, so
    the attributes must match the ones the conversation routes use - including
    `Secure` behind an HTTPS proxy. Setting them here and nowhere else avoids
    the class of bug where one code path issues a weaker cookie.

    Args:
        request: Incoming request.

    Returns:
        The 302 response carrying the session cookie.
    """
    thread_id = str(uuid.uuid4())
    user_id = user_id_of(request)
    response = RedirectResponse(f"/conversations/{thread_id}", status_code=302)
    response.set_cookie(
        key="user_id",
        value=user_id,
        path="/",
        max_age=31536000,
        httponly=True,
        samesite="lax",
        secure=cookie_is_secure(request),
    )
    return response


async def fetch_thread_state(client: Any, thread_id: str) -> Mapping[str, Any]:
    """Read a thread's latest state values.

    Args:
        client: SDK client.
        thread_id: Thread to read.

    Returns:
        The state values, or an empty mapping when unavailable.
    """
    try:
        state = await client.threads.get_state(thread_id)
    except Exception as exc:  # noqa: BLE001 - a new thread simply has no state
        logger.debug("no state for thread %s: %s", thread_id, type(exc).__name__)
        return {}
    if isinstance(state, Mapping):
        values = state.get("values")
        if isinstance(values, Mapping):
            return values
    return {}


def messages_of(values: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return the chat messages of a state mapping.

    Args:
        values: State values.

    Returns:
        Message mappings in order.
    """
    raw = values.get("messages")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return [item for item in raw if isinstance(item, Mapping)]
    return []


async def list_threads(client: Any, user_id: str) -> list[Mapping[str, Any]]:
    """List the conversations of one user.

    Args:
        client: SDK client.
        user_id: Cookie-based user id.

    Returns:
        Thread records, newest first as returned by the server.
    """
    try:
        threads = await client.threads.search(metadata={"user_id": user_id}, limit=50)
    except Exception as exc:  # noqa: BLE001 - the sidebar must never break a page
        logger.warning("could not list threads: %s", type(exc).__name__)
        return []
    if isinstance(threads, Sequence):
        return [thread for thread in threads if isinstance(thread, Mapping)]
    return []


async def maybe_set_title(client: Any, thread_id: str, message: str) -> None:
    """Store a short thread title from the first user message.

    No model call is involved: the title is the beginning of the user's own
    text, which keeps the sidebar free of extra cost.

    Args:
        client: SDK client.
        thread_id: Thread to title.
        message: First user message.
    """
    try:
        thread = await client.threads.get(thread_id)
    except Exception as exc:  # noqa: BLE001 - a missing title is cosmetic
        logger.debug("could not read thread %s: %s", thread_id, type(exc).__name__)
        return
    metadata = thread.get("metadata") if isinstance(thread, Mapping) else None
    if isinstance(metadata, Mapping) and metadata.get("title"):
        return
    title = " ".join(message.split())[:TITLE_CHARS] or "新对话"
    try:
        await client.threads.update(thread_id, metadata={"title": title})
    except Exception as exc:  # noqa: BLE001
        logger.debug("could not title thread %s: %s", thread_id, type(exc).__name__)


async def ensure_thread(
    client: Any,
    *,
    thread_id: str,
    session_id: str,
) -> tuple[bool, bool]:
    """Make sure a thread exists and belongs to this session.

    A thread created by this web app always carries `user_id` in its metadata.
    A thread created elsewhere (API, Studio) has no owner and is claimed by the
    first session that opens it; the claim is verified by re-reading the
    metadata, so two racing sessions cannot both end up owning it.

    The claim is written with `threads.update`, never with the `metadata`
    argument of `threads.create`: the runtime applies that argument even when
    the thread already exists (with `if_exists="do_nothing"`), which would let
    every new visitor take ownership of an existing conversation.

    Args:
        client: SDK client.
        thread_id: Thread id from the URL.
        session_id: Session cookie value.

    Returns:
        `(allowed, claimed)`. `allowed` is False when the thread belongs to
        another session; the caller then answers 404.
    """
    try:
        await client.threads.create(thread_id=thread_id, if_exists="do_nothing")
    except Exception as exc:  # noqa: BLE001 - an existing bad thread is handled below
        logger.debug("could not ensure thread: %s", type(exc).__name__)

    metadata: Any = None
    try:
        thread = await client.threads.get(thread_id)
        if isinstance(thread, Mapping):
            metadata = thread.get("metadata")
    except Exception as exc:  # noqa: BLE001 - no metadata means "no owner recorded"
        logger.debug("could not read thread metadata: %s", type(exc).__name__)

    decision = access.evaluate(metadata, session_id)
    if not decision.allowed or not decision.claimed:
        return decision.allowed, decision.claimed

    try:
        await client.threads.update(thread_id, metadata={access.OWNER_KEY: session_id})
        refreshed = await client.threads.get(thread_id)
        refreshed_metadata = (
            refreshed.get("metadata") if isinstance(refreshed, Mapping) else None
        )
        verified = access.evaluate(refreshed_metadata, session_id)
    except Exception as exc:  # noqa: BLE001 - claim is best effort
        logger.debug("could not claim thread: %s", type(exc).__name__)
        return True, False
    return verified.allowed, True


def notice_bubble(
    message: str, *, status_code: int = 200, cookie: str | None = None
) -> Any:
    """Render a limit notice with a real HTTP status.

    Args:
        message: Safe reader-facing text.
        status_code: 200, 413 or 429.
        cookie: Optional `Set-Cookie` value. Rate and size limits are evaluated
            after the session was identified, so a rejected request still has to
            hand the browser the cookie it will need next time.

    Returns:
        An HTML response carrying the bubble; the client swaps it in for 429
        and 413 as well (see the `htmx:beforeSwap` handler).
    """
    from fasthtml.common import to_xml
    from starlette.responses import HTMLResponse

    fragment = chat_bubble(
        Div(P(message, cls="input-hint"), cls="limit-notice"), role="bot"
    )
    headers = {"set-cookie": cookie} if cookie else None
    return HTMLResponse(str(to_xml(fragment)), status_code=status_code, headers=headers)


def not_found_page() -> Any:
    """Answer as if the thread did not exist.

    Used when a session asks for a thread owned by somebody else: the response
    does not confirm that the thread exists.

    Returns:
        A 404 HTML response.
    """
    from starlette.responses import HTMLResponse

    body = (
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<title>找不到会话</title></head><body style='font-family: system-ui;"
        "max-width: 40rem; margin: 4rem auto; line-height: 1.7'>"
        "<h1 style='font-size: 1.2rem'>找不到这个会话</h1>"
        "<p>它可能已删除，或者不属于当前浏览器会话。</p>"
        "<p><a href='/'>返回首页</a></p></body></html>"
    )
    return HTMLResponse(body, status_code=404)


async def _empty_stream() -> AsyncGenerator[str, None]:
    """Yield a single close frame (used for unauthorized stream requests).

    Yields:
        The `close` frame.
    """
    yield fragments.empty_stream_frame()


def register_routes(app: Any) -> None:
    """Attach every web route to a FastHTML app.

    Args:
        app: FastHTML application created in `react_agent.web.app`.
    """
    # FastHTML ships no stubs, so the decorators are cast once here instead of
    # sprinkling `type: ignore` over every handler.
    route: Callable[..., Callable[..., Any]] = app.get
    route_post: Callable[..., Callable[..., Any]] = app.post

    @route("/")
    async def root(request: Request) -> Any:
        """Redirect to a fresh conversation."""
        return new_conversation_redirect(request)

    @route("/new-thread")
    async def new_thread(request: Request) -> Any:
        """Redirect to a fresh conversation."""
        return new_conversation_redirect(request)

    @route("/health")
    async def health_route() -> Any:
        """Report readiness without exposing configuration."""
        from starlette.responses import JSONResponse

        return JSONResponse(build_report().as_dict())

    @route("/static/{filename}")
    async def static_asset(filename: str) -> Any:
        """Serve the vendored front-end assets (no CDN at runtime)."""
        from starlette.responses import FileResponse, Response

        name = Path(filename).name
        if name not in ALLOWED_STATIC_ASSETS:
            return Response(status_code=404)
        path = STATIC_DIR / name
        if not path.is_file():
            logger.warning("vendored asset missing: %s", name)
            return Response(status_code=404)
        return FileResponse(
            path,
            media_type="text/javascript",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @route("/evaluation")
    async def evaluation_route() -> Any:
        """Render the System / Evaluation page from the exported artifact."""
        summary = load_evaluation_summary()
        return page(
            title="PsycheGraph · System / Evaluation",
            active="evaluation",
            body=evaluation_page(summary),
        )

    @route("/conversations/{thread_id}")
    async def conversation(thread_id: str, request: Request) -> Any:
        """Render the chat page of one conversation."""
        client = get_client_instance()
        user_id = user_id_of(request)
        allowed, _claimed = await ensure_thread(
            client, thread_id=thread_id, session_id=user_id
        )
        if not allowed:
            return not_found_page()
        values = await fetch_thread_state(client, thread_id)
        messages = messages_of(values)
        citations = fragments.citations_for_state(values)
        sources = fragments.sources_for_state(values)
        threads = await list_threads(client, user_id)
        body = conversation_columns(
            threads=threads,
            current_thread_id=thread_id,
            messages=messages,
            citations=citations,
            sources=sources,
        )
        page_parts = page(
            title="PsycheGraph",
            active="chat",
            body=body,
            thread_id=thread_id,
        )
        logger.info("page rendered thread=%s", short_id(thread_id))
        return (*page_parts, *session_cookie(request, user_id))

    @route_post("/conversations/{thread_id}/send-message")
    async def send_message(thread_id: str, request: Request) -> Any:
        """Queue a message, enforcing ownership and the demo limits."""
        form = await request.form()
        raw = form.get("msg")
        message = str(raw).strip() if raw is not None else ""
        if not message:
            return chat_bubble(P(EMPTY_MESSAGE, cls="input-hint"), role="bot")
        client = get_client_instance()
        user_id = user_id_of(request)

        # Validate before touching the thread: an oversized message must not
        # create or claim anything, and the visitor keeps the identity that the
        # page (or this rejection) hands back.
        length = limits.check_input_length(message)
        if not length.allowed:
            return notice_bubble(
                length.message,
                status_code=length.status_code,
                cookie=session_cookie_value(request, user_id),
            )
        allowed, _claimed = await ensure_thread(
            client, thread_id=thread_id, session_id=user_id
        )
        if not allowed:
            return not_found_page()

        cookie = session_cookie_value(request, user_id)
        concurrency = limits.check_concurrency(user_id, REGISTRY.session_load(user_id))
        if not concurrency.allowed:
            return notice_bubble(
                concurrency.message,
                status_code=concurrency.status_code,
                cookie=cookie,
            )
        rate = limits.LIMITER.check(user_id)
        if not rate.allowed:
            return notice_bubble(
                rate.message, status_code=rate.status_code, cookie=cookie
            )

        if not REGISTRY.submit(thread_id, message, user_id):
            return notice_bubble(BUSY_MESSAGE)
        await maybe_set_title(client, thread_id, message)
        logger.info("run queued thread=%s chars=%d", short_id(thread_id), len(message))
        return (
            chat_bubble(message, role="user"),
            assistant_placeholder(),
            Div(sse_hub(thread_id), hx_swap_oob="innerHTML:#stream-controls"),
            workflow_panel(oob=True),
            evidence_body(None, oob=True),
            composer(thread_id, disabled=True, oob=True),
            *session_cookie(request, user_id),
        )

    @route("/conversations/{thread_id}/stream")
    async def stream(thread_id: str, request: Request) -> Any:
        """Stream one run of the thread as sanitized SSE events."""
        allowed, _claimed = await ensure_thread(
            get_client_instance(),
            thread_id=thread_id,
            session_id=user_id_of(request),
        )
        if not allowed:
            return StreamingResponse(
                _empty_stream(),
                media_type="text/event-stream",
                headers=dict(SSE_HEADERS),
            )
        return StreamingResponse(
            _stream_events(thread_id),
            media_type="text/event-stream",
            headers=dict(SSE_HEADERS),
        )


async def _stream_events(thread_id: str) -> AsyncGenerator[str, None]:
    """Run the graph for a queued message and yield SSE frames.

    Args:
        thread_id: Thread to run.

    Yields:
        SSE frames; the last one is always the `close` event, which also makes
        the browser close its EventSource.
    """
    pending = REGISTRY.take(thread_id)
    if pending is None:
        yield fragments.empty_stream_frame()
        return
    client = get_client_instance()
    answered = False
    started = time.monotonic()
    logger.info("run streaming thread=%s", short_id(thread_id))
    try:
        async for event in run_workflow_events(
            client, thread_id=thread_id, message=pending.text
        ):
            if event.event_type in {"answer", "error"}:
                answered = True
            frame = fragments.frame_for(
                event, thread_id=thread_id, answer_seen=answered
            )
            if frame:
                yield frame
    finally:
        REGISTRY.release(thread_id)
        logger.info(
            "run finished thread=%s answered=%s elapsed=%.1fs",
            short_id(thread_id),
            answered,
            time.monotonic() - started,
        )
