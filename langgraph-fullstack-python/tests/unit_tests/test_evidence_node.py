"""Evidence node tests: local retrieval only, never a model call."""

import asyncio
import json
from typing import Any

import pytest
from langchain_core.messages import HumanMessage

import react_agent.llm as llm_module
from react_agent.agents.evidence import evidence_node
from react_agent.state import SPECIALIST_PERSPECTIVES
from tests.unit_tests.conftest import RagIndex, make_plan


def _state(**extra: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "messages": [HumanMessage(content="我梦见水，水是平静的湖水。")],
        "supervisor_plan": make_plan(),
    }
    state.update(extra)
    return state


def test_evidence_node_stores_json_evidence_for_every_school(
    rag_index: RagIndex,
) -> None:
    update = asyncio.run(evidence_node(_state()))

    assert set(update) == {"evidence_by_school", "evidence_meta"}
    evidence = update["evidence_by_school"]
    assert isinstance(evidence, dict)
    assert set(evidence) == set(SPECIALIST_PERSPECTIVES)
    for school, entries in evidence.items():
        assert entries, f"no evidence retrieved for {school}"
        for entry in entries:
            assert isinstance(entry, dict)
            assert entry["school"] in {school, "general"}
            assert entry["evidence_id"]
            assert entry["text"]
    # State must stay JSON-serialisable: no Pydantic objects on the wire.
    json.dumps(update)


def test_evidence_node_makes_no_model_call(
    monkeypatch: pytest.MonkeyPatch, rag_index: RagIndex
) -> None:
    def _forbid(*_: Any, **__: Any) -> Any:
        raise AssertionError("the evidence node must not call DeepSeek")

    monkeypatch.setattr(llm_module, "get_chat_model", _forbid)
    monkeypatch.setattr(llm_module, "build_structured_runnable", _forbid)

    update = asyncio.run(evidence_node(_state()))

    assert update["evidence_meta"]["available"] is True


def test_evidence_node_requires_the_supervisor_plan(fake_model: Any) -> None:
    with pytest.raises(ValueError, match="supervisor_plan"):
        asyncio.run(evidence_node({"messages": [HumanMessage(content="我梦见水。")]}))


def test_evidence_node_without_an_index_returns_empty_lists(fake_model: Any) -> None:
    update = asyncio.run(evidence_node(_state()))

    assert update["evidence_by_school"] == {
        school: [] for school in SPECIALIST_PERSPECTIVES
    }
    assert update["evidence_meta"]["available"] is False
    assert update["evidence_meta"]["counts"] == {
        school: 0 for school in SPECIALIST_PERSPECTIVES
    }


def test_evidence_meta_records_timing_and_counts(rag_index: RagIndex) -> None:
    update = asyncio.run(evidence_node(_state()))
    meta = update["evidence_meta"]

    assert meta["available"] is True
    assert meta["reason"] is None
    assert meta["elapsed_seconds"] >= 0.0
    assert meta["retrieval_seconds"] >= 0.0
    assert meta["counts"] == {
        school: len(entries) for school, entries in update["evidence_by_school"].items()
    }


def test_evidence_node_reads_the_latest_user_message(rag_index: RagIndex) -> None:
    """The query is built from the plan plus the newest human message."""
    state = _state()
    state["messages"] = [
        HumanMessage(content="第一轮：我梦见水。"),
        HumanMessage(content="第二轮：水是平静的湖水，我很安心。"),
    ]

    update = asyncio.run(evidence_node(state))

    assert update["evidence_by_school"]["freudian"]


def test_evidence_node_offloads_retrieval_to_a_worker_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The blocking retrieval must not run on the event loop thread.

    This is what removed the `--allow-blocking` requirement: the dev server's
    blocking-call detector only looks at the event loop thread.
    """
    import threading

    import react_agent.agents.evidence as evidence_module
    from react_agent.rag.retriever import RetrievalOutcome

    event_loop_thread = threading.get_ident()
    seen: dict[str, int] = {}

    def fake_retrieve(plan: Any, messages: Any, **kwargs: Any) -> RetrievalOutcome:
        seen["thread"] = threading.get_ident()
        return RetrievalOutcome(
            evidence_by_school={"freudian": [], "object_relations": [], "lacanian": []},
            available=True,
            elapsed_seconds=0.001,
        )

    monkeypatch.setattr(evidence_module, "retrieve_evidence", fake_retrieve)
    update = asyncio.run(evidence_node(_state()))

    assert seen["thread"] != event_loop_thread
    assert update["evidence_meta"]["available"] is True


def test_evidence_node_keeps_the_event_loop_responsive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow retrieval must not stop other coroutines from running."""
    import threading
    import time

    import react_agent.agents.evidence as evidence_module
    from react_agent.rag.retriever import RetrievalOutcome

    main_thread = threading.get_ident()

    def slow_retrieve(plan: Any, messages: Any, **kwargs: Any) -> RetrievalOutcome:
        assert threading.get_ident() != main_thread, "retrieval ran on the event loop"
        time.sleep(0.3)
        return RetrievalOutcome(
            evidence_by_school={"freudian": [], "object_relations": [], "lacanian": []},
            available=False,
            elapsed_seconds=0.3,
            reason="test",
        )

    async def run() -> int:
        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            for _ in range(10):
                ticks += 1
                await asyncio.sleep(0.03)

        await asyncio.gather(evidence_node(_state()), ticker())
        return ticks

    monkeypatch.setattr(evidence_module, "retrieve_evidence", slow_retrieve)
    ticks = asyncio.run(run())

    assert ticks == 10, "the event loop was blocked while the retrieval ran"
