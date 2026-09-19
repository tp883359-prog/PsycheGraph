"""Route and SSE integration tests.

The whole HTTP surface is exercised through Starlette's test client against the
real FastHTML app, with only the LangGraph client replaced by a replayed event
sequence. That covers the parts a unit test cannot: which SSE event names are
produced, in what order, what the swapped HTML contains, and that a run failure
still re-enables the composer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

import react_agent.web.evaluation as evaluation_module
import react_agent.web.routes as routes_module
from react_agent.web.app import app
from react_agent.web.pending import REGISTRY
from tests.unit_tests.web_helpers import (
    CRITIQUE_TEXT,
    DRAFT_TEXT,
    FAKE_ANSWER,
    WINDOWS_PATH,
    FakeClient,
    evidence_by_school,
    fallback_sequence,
    normal_sequence,
    revision_sequence,
)

THREAD_ID = "thread-1"
"""Thread used by every route test."""

SEND_URL = f"/conversations/{THREAD_ID}/send-message"
STREAM_URL = f"/conversations/{THREAD_ID}/stream"


@pytest.fixture(autouse=True)
def clean_registry() -> Any:
    """Keep the in-process run registry empty between tests."""
    REGISTRY.clear()
    yield
    REGISTRY.clear()


@pytest.fixture()
def client() -> Any:
    """Return a test client for the FastHTML app."""
    with TestClient(app) as test_client:
        yield test_client
    routes_module.set_client_instance(None)


def install(fake: FakeClient) -> FakeClient:
    """Inject a fake LangGraph client into the routes.

    Args:
        fake: Client double.

    Returns:
        The same client, for chaining.
    """
    routes_module.set_client_instance(fake)
    return fake


def sse_frames(body: str) -> list[tuple[str, str]]:
    """Parse an event-stream body into (event, data) pairs.

    Args:
        body: Raw SSE body.

    Returns:
        Pairs in order.
    """
    frames: list[tuple[str, str]] = []
    for block in body.split("\n\n"):
        lines = [line for line in block.splitlines() if line]
        if not lines:
            continue
        event = next(
            (line.split(": ", 1)[1] for line in lines if line.startswith("event: ")),
            "message",
        )
        data = "\n".join(
            line.split(": ", 1)[1] for line in lines if line.startswith("data: ")
        )
        frames.append((event, data))
    return frames


# ------------------------------------------------------------------- pages


def test_root_redirects_to_a_new_conversation(client: Any) -> None:
    """The root URL always starts a fresh conversation."""
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"].startswith("/conversations/")
    assert "user_id" in response.cookies


def test_new_thread_redirects(client: Any) -> None:
    """The sidebar's new-chat link works."""
    response = client.get("/new-thread", follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"].startswith("/conversations/")


def test_conversation_page_renders_three_columns(client: Any) -> None:
    """The page shows conversations, chat, workflow and evidence."""
    install(
        FakeClient(threads=[{"thread_id": THREAD_ID, "metadata": {"title": "水的梦"}}])
    )
    response = client.get(f"/conversations/{THREAD_ID}")
    assert response.status_code == 200
    body = response.text
    assert "PsycheGraph" in body
    assert "Multi-agent psychoanalytic theory explorer" in body
    assert "不用于临床诊断" in body
    assert 'id="chatlist"' in body
    assert 'id="workflow-panel"' in body
    assert 'id="evidence-panel"' in body
    assert 'id="input-area"' in body
    assert 'id="stream-controls"' in body
    assert 'hx-ext="sse"' in body
    assert 'sse-close="close"' not in body  # the hub only arrives with a run
    assert 'aria-label="输入消息"' in body
    assert "水的梦" in body


def test_conversation_page_shows_stored_run(client: Any) -> None:
    """A reload keeps the newest answer, its citations and the evidence."""
    install(
        FakeClient(
            state={
                "messages": [
                    {"type": "human", "content": "我梦见水。"},
                    {
                        "type": "ai",
                        "content": f"{FAKE_ANSWER}\n\n理论依据（本地知识库）\n[E1] 来源",
                    },
                ],
                "final_result": {"used_evidence_ids": ["freud_dreams_000001"]},
                "finalization_status": "passed",
                "evidence_by_school": evidence_by_school(),
                "evidence_meta": {"available": True, "counts": {}},
            }
        )
    )
    body = client.get(f"/conversations/{THREAD_ID}").text
    assert "我梦见水。" in body
    assert 'data-evidence="freud_dreams_000001"' in body
    assert "梦的工作把愿望改写成意象" in body
    assert "项目自编测试材料" in body


# -------------------------------------------------------------- send-message


def test_send_message_returns_bubble_placeholder_and_hub(client: Any) -> None:
    """One POST swaps in the user bubble, the waiting bubble and the hub."""
    install(FakeClient(normal_sequence()))
    response = client.post(SEND_URL, data={"msg": "我梦见水。"})
    assert response.status_code == 200
    body = response.text
    assert "我梦见水。" in body
    assert 'id="assistant-placeholder"' in body
    assert f'sse-connect="/conversations/{THREAD_ID}/stream"' in body
    assert 'sse-swap="workflow"' in body
    assert 'sse-swap="answer"' in body
    assert 'sse-swap="sources"' in body
    assert 'sse-swap="error"' in body
    assert 'sse-swap="close"' in body
    assert 'hx-target="#chatlist"' in body
    assert "hx-swap-oob" in body
    assert "disabled" in body  # the composer is disabled while the run streams
    assert 'id="composer-text"' in body


def test_send_message_swaps_instead_of_appending_duplicates(client: Any) -> None:
    """A run must not leave a second composer or a second hub in the DOM."""
    install(FakeClient(normal_sequence()))
    body = client.post(SEND_URL, data={"msg": "我梦见水。"}).text
    assert body.count('id="composer-text"') == 1
    assert body.count('id="input-area"') == 1
    assert body.count("sse-connect=") == 1
    assert body.count('id="sse-hub"') == 1
    assert 'hx-swap-oob="outerHTML:#input-area"' in body
    assert 'hx-swap-oob="innerHTML:#stream-controls"' in body
    assert 'hx-swap-oob="outerHTML:#workflow-panel"' in body
    assert 'hx-swap-oob="innerHTML:#evidence-body"' in body
    assert 'hx-swap-oob="delete:#assistant-placeholder"' not in body


def test_send_message_rejects_empty_input(client: Any) -> None:
    """An empty submit is answered with a hint, not a run."""
    install(FakeClient(normal_sequence()))
    body = client.post(SEND_URL, data={"msg": "   "}).text
    assert "请输入内容后再发送。" in body


def test_second_message_while_busy_is_refused(client: Any) -> None:
    """No duplicate concurrent runs in one session.

    Phase 10A moved this guard server-side: the per-session concurrency limit
    answers 429 before a second run can be queued, and the browser swaps the
    notice in because `htmx:beforeSwap` accepts 413/429.
    """
    install(FakeClient(normal_sequence()))
    client.post(SEND_URL, data={"msg": "第一条"})
    response = client.post(SEND_URL, data={"msg": "第二条"})
    assert response.status_code == 429
    assert "同时只能有一条分析在运行" in response.text
    assert 'id="assistant-placeholder"' not in response.text


def test_thread_gets_a_title_from_the_first_message(client: Any) -> None:
    """The sidebar title costs no model call."""
    fake = install(FakeClient(normal_sequence()))
    client.post(SEND_URL, data={"msg": "从三个精神分析视角分析《哈姆雷特》"})
    assert fake.threads.updates
    # The first metadata write claims the thread for this browser session; the
    # title follows it.
    titles = [u["title"] for u in fake.threads.updates if "title" in u]
    assert titles, "the title must be stored on the thread"
    assert titles[0].startswith("从三个精神分析视角分析")
    assert len(titles[0]) <= 24


# ------------------------------------------------------------------- streams


def stream_body(client: Any) -> str:
    """Fetch the SSE body of one run.

    Args:
        client: Test client.

    Returns:
        The whole event stream as text.
    """
    response = client.get(STREAM_URL)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    return response.text


def test_stream_without_pending_message_closes_immediately(client: Any) -> None:
    """A stray stream request is answered with a close event."""
    install(FakeClient(normal_sequence()))
    frames = sse_frames(stream_body(client))
    assert [event for event, _ in frames] == ["close"]


def test_full_run_stream_contract(client: Any) -> None:
    """The normal path: workflow progresses, then sources, then one answer."""
    install(FakeClient(normal_sequence()))
    client.post(SEND_URL, data={"msg": "我梦见水。"})
    frames = sse_frames(stream_body(client))
    events = [event for event, _ in frames]
    assert events[0] == "workflow"
    assert events[-1] == "close"
    assert events.count("answer") == 1
    assert events.count("sources") == 1
    assert events.index("sources") < events.index("answer")
    assert len(REGISTRY._active) == 0  # noqa: SLF001 - the run released its thread


def test_stream_reports_every_node_status(client: Any) -> None:
    """All nodes reach a completed state in the swapped rows."""
    install(FakeClient(normal_sequence()))
    client.post(SEND_URL, data={"msg": "我梦见水。"})
    body = stream_body(client)
    for node in (
        "supervisor",
        "evidence",
        "freudian",
        "object_relations",
        "lacanian",
        "synthesizer",
        "deterministic_validator",
        "critic",
        "finalize",
    ):
        assert f"outerHTML:#wf-{node}" in body
    assert 'data-status="completed"' in body
    assert "第二次审核" not in body


def test_stream_never_leaks_internals(client: Any) -> None:
    """Drafts, critiques and local paths stay on the server."""
    install(FakeClient(revision_sequence(cited=["freud_dreams_000001"])))
    client.post(SEND_URL, data={"msg": "分析《哈姆雷特》。"})
    body = stream_body(client)
    for forbidden in (
        DRAFT_TEXT,
        CRITIQUE_TEXT,
        WINDOWS_PATH,
        "source_path",
        "retrieval_score",
        "final_result",
        "supervisor_plan",
    ):
        assert forbidden not in body


def test_revision_path_is_streamed(client: Any) -> None:
    """A revised run shows the revision row and the second review."""
    install(FakeClient(revision_sequence(cited=["freud_dreams_000001"])))
    client.post(SEND_URL, data={"msg": "分析《哈姆雷特》。"})
    body = stream_body(client)
    assert "outerHTML:#wf-revise_synthesis" in body
    assert "正在修订综合回答" in body
    assert "第二次审核" in body
    assert "修订一次后通过" in body


def test_safe_fallback_stream(client: Any) -> None:
    """A fallback run explains itself in plain words."""
    install(FakeClient(fallback_sequence()))
    client.post(SEND_URL, data={"msg": "我是不是有边缘型人格障碍？"})
    body = stream_body(client)
    assert "质量检查未能通过，系统采用了更保守的回答。" in body
    assert "outerHTML:#wf-safe_finalize" in body
    assert "未通过，采用保守回答" in body


def test_failed_run_emits_error_and_reenables_input(client: Any) -> None:
    """A failure ends with an error card, a close event and a usable composer."""
    install(
        FakeClient(normal_sequence()[:8], error=RuntimeError("boom: sk-secret-value"))
    )
    client.post(SEND_URL, data={"msg": "我梦见水。"})
    frames = sse_frames(stream_body(client))
    events = [event for event, _ in frames]
    assert events.count("error") == 1
    assert events[-1] == "close"
    body = stream_body(client) if False else "".join(data for _event, data in frames)
    assert "本次分析未完成，请重试。" in body
    assert "sk-secret-value" not in body
    assert "RuntimeError" not in body
    assert "删除等待气泡" not in body
    assert 'name="msg"' in body  # the composer is restored


def test_answer_frame_carries_clickable_citations(client: Any) -> None:
    """Citation labels in the answer link to real evidence cards."""
    install(FakeClient(normal_sequence(cited=["freud_dreams_000001"])))
    client.post(SEND_URL, data={"msg": "我梦见水。"})
    body = stream_body(client)
    answer_frame = next(data for event, data in sse_frames(body) if event == "answer")
    assert 'data-evidence="freud_dreams_000001"' in answer_frame
    assert "delete:#assistant-placeholder" in answer_frame
    sources_frame = next(data for event, data in sse_frames(body) if event == "sources")
    assert 'data-evidence-card="freud_dreams_000001"' in sources_frame


def test_second_run_streams_after_the_first(client: Any) -> None:
    """Two sequential runs in one thread both stream normally."""
    install(FakeClient(normal_sequence()))
    client.post(SEND_URL, data={"msg": "第一条"})
    stream_body(client)
    client.post(SEND_URL, data={"msg": "第二条"})
    frames = sse_frames(stream_body(client))
    assert [event for event, _ in frames].count("answer") == 1


# ---------------------------------------------------------------- evaluation


def test_evaluation_page_without_artifact(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The page explains how to generate the summary when it is missing."""
    monkeypatch.setattr(
        evaluation_module, "DEFAULT_SUMMARY_PATH", Path("does-not-exist.json")
    )
    body = client.get("/evaluation").text
    assert "System / Evaluation" in body
    assert "export_evaluation_summary.py" in body
    assert "项目自编测试语料" in body
    assert "原著数据库" in body


def test_evaluation_page_with_artifact(
    client: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The page renders the exported numbers and the disclaimer."""
    payload = {
        "schema_version": 1,
        "generated_at": "2026-09-16T00:00:00Z",
        "source": "data/evals/full/report_rescored.json",
        "dataset_cases": 60,
        "rescored": True,
        "variants": [
            {
                "key": "single_agent",
                "label": "Single Agent",
                "runs": 60,
                "metrics": {"avg_llm_calls": 1.0, "avg_total_tokens": 3244.0},
            },
            {
                "key": "full_system",
                "label": "Full System (Critic)",
                "runs": 60,
                "metrics": {"avg_llm_calls": 6.23, "avg_total_tokens": 33692.0},
            },
        ],
        "notes": ["test corpus"],
    }
    path = tmp_path / "evaluation_summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(evaluation_module, "DEFAULT_SUMMARY_PATH", path)
    body = client.get("/evaluation").text
    assert "Single Agent" in body
    assert "Full System" in body
    assert "3,244" in body
    assert "33,692" in body
    assert "N/A 表示该指标对该变体不适用" in body
    assert "不代表心理诊断有效性" in body
