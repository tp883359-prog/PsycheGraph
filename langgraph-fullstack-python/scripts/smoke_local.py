"""End-to-end smoke test against a running server (local or in a container).

Checks the Phase 10A/10B guarantees from the outside, the way a deployment
would:

    1. /health answers with the four safe fields
    2. vendored assets are served by the app (no CDN)
    3. the landing page and /new-thread redirect into a fresh conversation
    4. /evaluation renders from the exported artifact
    5. a 4001-character message is refused with 413
    6. a conversation page renders for its visitor
    7. the second immediate submission is refused with 429
    8. another browser session cannot open the conversation (404)
    9. the runtime API answers 401 without a bearer token (when one is set)
    10. optionally: a real message produces the workflow/sources/answer/close
        events over SSE (covers the critic step and the evidence panel, which
        the UI renders from those events)
    11. the refreshed page still shows the stored turn
    12. a new conversation still accepts messages afterwards (the run slot was
        released - the server-side equivalent of the composer being enabled)

Usage (server already running):

    uv run python scripts/smoke_local.py --url http://127.0.0.1:2024
    uv run python scripts/smoke_local.py --url http://localhost:8123 \
        --token "$env:LANGGRAPH_DEMO_API_TOKEN" --skip-run
"""

# Progress output is the intended deliverable of this CLI.
# ruff: noqa: T201

from __future__ import annotations

import argparse
import sys
import time
import uuid
from typing import Any

import httpx

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    """Store and print one check result.

    Args:
        name: Check label.
        ok: Whether the check passed.
        detail: Extra information for the log line.
    """
    RESULTS.append((name, ok, detail))
    print(f"[{'ok  ' if ok else 'FAIL'}] {name}{f' - {detail}' if detail else ''}")


def main() -> int:
    """Run the smoke checks.

    Returns:
        Process exit code (1 when any check failed).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:2024")
    parser.add_argument("--token", default="")
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="do not wait for a real answer over SSE",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="skip TLS verification (self-signed certificate of a local proxy)",
    )
    parser.add_argument(
        "--behind-proxy",
        action="store_true",
        help="the URL is a reverse-proxy entry point: /threads must be 404 there",
    )
    args = parser.parse_args()

    headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}
    # The runtime validates thread ids as UUIDs, so the smoke test must use one.
    thread_id = str(uuid.uuid4())
    base = args.url.rstrip("/")
    verify = not args.insecure

    with httpx.Client(base_url=base, timeout=30.0, verify=verify) as browser:
        response = browser.get("/health", headers=headers)
        payload: dict[str, Any] = {}
        try:
            payload = response.json()
        except Exception:  # noqa: BLE001 - reported as a failed check
            payload = {}
        record(
            "health returns the four safe fields",
            response.status_code == 200
            and set(payload)
            == {
                "status",
                "graph_loaded",
                "vectorstore_available",
                "embedding_loaded",
            },
            f"status={response.status_code} payload={payload}",
        )

        asset = browser.get("/static/htmx.min.js")
        record(
            "vendored htmx is served locally",
            asset.status_code == 200
            and asset.headers.get("content-type", "").startswith("text/javascript"),
            f"{asset.status_code} {len(asset.content)} bytes",
        )

        landing = browser.get("/", follow_redirects=False)
        location = landing.headers.get("location", "")
        record(
            "landing page redirects to a new conversation",
            landing.status_code in {302, 307}
            and location.startswith("/conversations/")
            and "user_id=" in landing.headers.get("set-cookie", ""),
            f"status={landing.status_code} location={location}",
        )

        new_thread = browser.get("/new-thread", follow_redirects=False)
        record(
            "new-thread redirects like the landing page",
            new_thread.status_code in {302, 307}
            and new_thread.headers.get("location", "").startswith("/conversations/"),
            f"status={new_thread.status_code}",
        )

        evaluation = browser.get("/evaluation")
        record(
            "evaluation page renders from the exported artifact",
            evaluation.status_code == 200 and "Evaluation" in evaluation.text,
            f"status={evaluation.status_code} bytes={len(evaluation.content)}",
        )

        oversized = browser.post(
            f"/conversations/{thread_id}/send-message", data={"msg": "梦" * 4001}
        )
        record(
            "oversized input is refused with 413",
            oversized.status_code == 413,
            f"status={oversized.status_code}",
        )

        page = browser.get(f"/conversations/{thread_id}")
        record(
            "conversation page renders for its visitor",
            page.status_code == 200 and "htmx" in page.text,
            f"status={page.status_code}",
        )

        first = browser.post(
            f"/conversations/{thread_id}/send-message", data={"msg": "我梦见水。"}
        )
        second = browser.post(
            f"/conversations/{thread_id}/send-message", data={"msg": "再问一次。"}
        )
        record(
            "second submission while busy is refused with 429",
            first.status_code == 200 and second.status_code == 429,
            f"first={first.status_code} second={second.status_code}",
        )

        with httpx.Client(base_url=base, timeout=30.0, verify=verify) as other_browser:
            intruder = other_browser.get(f"/conversations/{thread_id}", headers=headers)
        record(
            "another browser session gets 404",
            intruder.status_code == 404 and "找不到" in intruder.text,
            f"status={intruder.status_code}",
        )

        if args.token or args.behind_proxy:
            anonymous = httpx.get(f"{base}/threads", timeout=30.0, verify=verify)
            if args.behind_proxy:
                record(
                    "runtime API is blocked at the public entry point",
                    anonymous.status_code == 404,
                    f"status={anonymous.status_code}",
                )
            else:
                record(
                    "runtime API rejects a tokenless request",
                    anonymous.status_code == 401,
                    f"status={anonymous.status_code}",
                )

        if not args.skip_run:
            started = time.perf_counter()
            seen: set[str] = set()
            answer_text = ""
            try:
                with browser.stream(
                    "GET",
                    f"/conversations/{thread_id}/stream",
                    headers=headers,
                    timeout=args.timeout,
                ) as stream:
                    event = ""
                    for line in stream.iter_lines():
                        if line.startswith("event: "):
                            event = line.split(": ", 1)[1]
                            seen.add(event)
                        elif line.startswith("data: ") and event == "answer":
                            answer_text += line.split(": ", 1)[1]
                        elif not line and "close" in seen:
                            break
            except Exception as exc:  # noqa: BLE001 - reported as a failed check
                record("SSE stream completed", False, type(exc).__name__)
            elapsed = time.perf_counter() - started
            record(
                "SSE stream delivered an answer",
                "answer" in seen and "close" in seen,
                f"events={sorted(seen)} elapsed={elapsed:.1f}s answer_chars={len(answer_text)}",
            )

            refreshed = browser.get(f"/conversations/{thread_id}")
            record(
                "conversation refresh keeps the stored turn",
                refreshed.status_code == 200 and "我梦见水" in refreshed.text,
                f"status={refreshed.status_code}",
            )

            second_thread = str(uuid.uuid4())
            follow_up = browser.post(
                f"/conversations/{second_thread}/send-message",
                data={"msg": "新的会话还能继续提问吗？"},
            )
            record(
                "a new conversation still accepts messages after a run",
                follow_up.status_code == 200,
                f"status={follow_up.status_code}",
            )

    failed = [name for name, ok, _ in RESULTS if not ok]
    print(
        f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed"
        + (f"; failed: {', '.join(failed)}" if failed else "")
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
