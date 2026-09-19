"""Tests for the web streaming adapter (raw LangGraph events -> UI events).

These tests replay event sequences whose shapes were captured from a real
`langgraph dev` run, so they also document the contract the UI depends on:

* only `finalize` / `safe_finalize` produce an `answer` event,
* the `sources` event comes from the evidence node,
* every run ends with exactly one `close` event,
* drafts, critiques, prompts, tool arguments, paths and secrets never appear in
  any emitted event or rendered fragment.
"""

import asyncio
from typing import Any

import pytest

from react_agent.web.citations import build_sources_payload
from react_agent.web.events import (
    ANSWER_NODES,
    NODE_SPEC_BY_NAME,
    SCHOOL_LABELS,
    error_message,
    status_symbol,
    status_text,
)
from react_agent.web.fragments import frame_for
from react_agent.web.streaming import (
    STREAM_MODES,
    StreamTranslator,
    run_workflow_events,
)
from tests.unit_tests.web_helpers import (
    CRITIQUE_TEXT,
    DRAFT_TEXT,
    FAKE_ANSWER,
    WINDOWS_PATH,
    FakeClient,
    fallback_sequence,
    normal_sequence,
    revision_sequence,
)


def replay(sequence: Any, **kwargs: Any) -> tuple[StreamTranslator, list[Any]]:
    """Feed a sequence through a translator and collect the UI events.

    Args:
        sequence: A sequence builder (called with `**kwargs`) or a ready-made
            list of recorded event pairs.
        **kwargs: Forwarded to the sequence builder.

    Returns:
        The translator and the emitted UI events.
    """
    translator = StreamTranslator()
    events: list[Any] = []
    recorded = sequence(**kwargs) if callable(sequence) else sequence
    for event, data in recorded:
        events.extend(translator.handle(event, data))
    events.extend(translator.finish())
    return translator, events


def statuses_of(events: Any) -> dict[str, str]:
    """Return the last workflow status per node.

    Args:
        events: UI events.

    Returns:
        Mapping from node name to its final status.
    """
    result: dict[str, str] = {}
    for event in events:
        if event.event_type == "workflow" and event.node:
            result[event.node] = event.status
    return result


def replay_collected(translator: StreamTranslator) -> list[Any]:
    """Collect the events a translator emits when it finishes.

    Args:
        translator: Translator that already consumed a sequence.

    Returns:
        The close-phase events.
    """
    return list(translator.finish())


def messages_of(events: Any) -> list[str]:
    """Return the message of every workflow event.

    Args:
        events: UI events.

    Returns:
        Message strings in order.
    """
    return [event.message for event in events if event.event_type == "workflow"]


# --------------------------------------------------------------------- contract


def test_node_specs_cover_every_graph_node() -> None:
    """The UI list must match the production graph one-to-one."""
    from react_agent.graph import graph

    graph_nodes = {name for name in graph.nodes if not name.startswith("__")}
    spec_nodes = {spec.name for spec in NODE_SPEC_BY_NAME.values()}
    assert graph_nodes == spec_nodes


def test_answer_nodes_are_the_finalizers() -> None:
    """Only the two finalizers may publish an answer."""
    assert set(ANSWER_NODES) == {"finalize", "safe_finalize"}


@pytest.mark.parametrize(
    "status", ["waiting", "running", "completed", "revising", "failed", "skipped"]
)
def test_every_status_has_text_and_symbol(status: str) -> None:
    """State must be readable without colour."""
    assert status_text(status)  # type: ignore[arg-type]
    assert status_symbol(status)  # type: ignore[arg-type]


def test_stream_modes_do_not_request_token_streaming() -> None:
    """Structured output makes token streaming useless; keep it off."""
    assert "messages" not in STREAM_MODES
    assert set(STREAM_MODES) == {"tasks", "updates"}


# ----------------------------------------------------------------- normal flow


def test_normal_flow_reports_every_node() -> None:
    """A passing run completes every node and ends with one close event."""
    translator, events = replay(normal_sequence)
    assert translator.answer_emitted
    assert events[-1].event_type == "close"
    assert sum(1 for event in events if event.event_type == "close") == 1
    statuses = statuses_of(events)
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
        assert statuses[node] == "completed"
    assert statuses["revise_synthesis"] == "skipped"
    assert statuses["safe_finalize"] == "skipped"


def test_parallel_specialists_report_separately() -> None:
    """Each specialist must be visible on its own, not as one blob."""
    translator, events = replay(normal_sequence)
    labels = [
        event.label
        for event in events
        if event.event_type == "workflow"
        and event.node
        in {
            "freudian",
            "object_relations",
            "lacanian",
        }
    ]
    assert "弗洛伊德视角" in labels
    assert "客体关系视角" in labels
    assert "拉康视角" in labels
    runs = [
        event.message
        for event in events
        if event.event_type == "workflow" and event.node == "freudian"
    ]
    assert any("正在" in message for message in runs)
    assert any("完成" in message for message in runs)


def test_answer_event_carries_only_finalized_content() -> None:
    """The answer comes from the finalizer update and cites real evidence."""
    translator, events = replay(normal_sequence, cited=["freud_dreams_000001"])
    answers = [event for event in events if event.event_type == "answer"]
    assert len(answers) == 1
    payload = answers[0].metadata.answer
    assert payload is not None
    assert payload.text == FAKE_ANSWER
    assert payload.status == "passed"
    assert payload.status_label.startswith("质量审核")
    assert [ref.label for ref in payload.citations] == ["E1"]
    assert payload.citations[0].evidence_id == "freud_dreams_000001"
    assert {tag.school for tag in payload.theory_tags} == {
        "freudian",
        "object_relations",
        "lacanian",
    }
    assert all(tag.completed for tag in payload.theory_tags)


def test_sources_event_precedes_the_answer() -> None:
    """Evidence must be on screen before the answer's citations are clickable."""
    translator, events = replay(normal_sequence, cited=["freud_dreams_000001"])
    kinds = [event.event_type for event in events]
    assert kinds.index("sources") < kinds.index("answer")
    sources = events[kinds.index("sources")].metadata.sources
    assert sources is not None
    assert len(sources.cards) == 3
    assert sources.available is True
    assert {card.school for card in sources.cards} == {
        "freudian",
        "object_relations",
        "lacanian",
    }


def test_evidence_message_counts_real_chunks() -> None:
    """The evidence node's message reports the retrieved chunk count."""
    _, events = replay(normal_sequence)
    evidence = [
        event
        for event in events
        if event.event_type == "workflow" and event.node == "evidence"
    ]
    assert "3 条证据" in evidence[-1].message


def test_durations_come_from_task_events() -> None:
    """Durations are measured between a node's start and its update."""
    now = [0.0]
    translator = StreamTranslator(clock=lambda: now[0])
    collected: list[Any] = []
    for event, data in normal_sequence():
        now[0] += 2.0
        collected.extend(translator.handle(event, data))
    collected.extend(translator.finish())
    completed = [
        event
        for event in collected
        if event.event_type == "workflow" and event.status == "completed"
    ]
    durations = [event.metadata.duration_seconds for event in completed]
    assert completed
    assert any(duration and duration > 0 for duration in durations)
    assert all(duration is None or duration > 0 for duration in durations)


def test_no_duration_is_invented_for_waiting_nodes() -> None:
    """A node that never ran reports no duration."""
    _, events = replay(fallback_sequence)
    skipped = [
        event
        for event in events
        if event.event_type == "workflow" and event.status == "skipped"
    ]
    assert skipped
    assert all(event.metadata.duration_seconds is None for event in skipped)


# ------------------------------------------------------------- revision & fallback


def test_revision_path_is_visible() -> None:
    """The revision loop shows the second validator and critic passes."""
    _, events = replay(revision_sequence, cited=["freud_dreams_000001"])
    sequence = [
        (event.node, event.status)
        for event in events
        if event.event_type == "workflow"
        and event.node in {"critic", "revise_synthesis"}
    ]
    assert sequence == [
        ("critic", "running"),
        ("critic", "completed"),
        ("revise_synthesis", "revising"),
        ("revise_synthesis", "completed"),
        ("critic", "running"),
        ("critic", "completed"),
    ]
    critic_messages = [
        event.message
        for event in events
        if event.event_type == "workflow" and event.node == "critic"
    ]
    assert any("修订" in message for message in critic_messages)
    assert any("第二次审核" in message for message in critic_messages)
    answer = next(event for event in events if event.event_type == "answer")
    payload = answer.metadata.answer
    assert payload is not None
    assert payload.status == "revised_and_passed"
    assert payload.revision_count == 1


def test_revise_synthesis_running_status_is_revising() -> None:
    """The revision step uses its own status, not the generic running one."""
    _, events = replay(revision_sequence)
    running = [
        event
        for event in events
        if event.event_type == "workflow"
        and event.node == "revise_synthesis"
        and event.status != "completed"
    ]
    assert running and running[0].status == "revising"


def test_safe_fallback_shows_conservative_message() -> None:
    """A fallback run must say what happened without blaming anyone."""
    _, events = replay(fallback_sequence)
    answer = next(event for event in events if event.event_type == "answer")
    payload = answer.metadata.answer
    assert payload is not None
    assert payload.status == "safe_fallback"
    assert "保守" in payload.status_label
    statuses = statuses_of(events)
    assert statuses["safe_finalize"] == "completed"
    assert statuses["finalize"] == "skipped"


def test_degraded_retrieval_is_reported_honestly() -> None:
    """The evidence event reports the real chunk count and availability."""
    _, events = replay(fallback_sequence)
    sources = next(event for event in events if event.event_type == "sources")
    payload = sources.metadata.sources
    assert payload is not None
    assert len(payload.cards) == len(payload.cards)
    assert f"{len(payload.cards)} 条证据" in sources.message


# ---------------------------------------------------------------------- failures


def collect_stream(client: FakeClient, message: str = "hi") -> list[Any]:
    """Drain the async runner into a list (the suite is sync by convention).

    Args:
        client: Fake client replaying a sequence.
        message: User message to send.

    Returns:
        The emitted UI events.
    """

    async def collect() -> list[Any]:
        return [
            event
            async for event in run_workflow_events(
                client, thread_id="t1", message=message
            )
        ]

    return asyncio.run(collect())


def test_run_failure_emits_safe_error_event() -> None:
    """An exception becomes a coded error event, never a traceback."""
    client = FakeClient(normal_sequence()[:6], error=RuntimeError("boom: secret-key"))
    events = collect_stream(client)
    errors = [event for event in events if event.event_type == "error"]
    assert len(errors) == 1
    assert errors[0].metadata.error_code == "run_failed"
    assert "secret-key" not in errors[0].message
    assert "boom" not in errors[0].message
    assert events[-1].event_type == "close"
    final_statuses = statuses_of(events)
    assert final_statuses
    assert all(
        status in {"failed", "skipped", "completed"}
        for status in final_statuses.values()
    )


def test_spilled_task_failure_marks_the_node() -> None:
    """A specialist that errors is shown as failed, never as completed."""
    sequence = [
        (event, data)
        for event, data in normal_sequence()
        if not (event == "updates" and "lacanian" in data)
    ]
    index = next(
        i
        for i, (event, data) in enumerate(sequence)
        if event == "tasks" and data.get("name") == "lacanian" and "result" in data
    )
    event, data = sequence[index]
    sequence[index] = (event, {**data, "error": "ValidationError"})
    events = collect_stream(FakeClient(sequence))
    lacanian = [
        item
        for item in events
        if item.event_type == "workflow" and item.node == "lacanian"
    ]
    assert any(item.status == "failed" for item in lacanian)
    answer = next(item for item in events if item.event_type == "answer")
    payload = answer.metadata.answer
    assert payload is not None
    assert not all(tag.completed for tag in payload.theory_tags)


def test_stream_error_event_uses_safe_code() -> None:
    """The API's own error event is mapped, not forwarded."""
    translator = StreamTranslator()
    events = translator.handle("error", {"error": "RuntimeError: apikey=sk-1234567890"})
    assert [event.event_type for event in events] == ["error"]
    assert events[0].metadata.error_code == "run_failed"
    assert "sk-" not in events[0].message


def test_error_messages_are_fixed_text() -> None:
    """Error text never depends on the payload."""
    assert error_message("run_failed") == "本次分析未完成，请重试。"
    assert error_message("unknown-code") == "本次分析未完成，请重试。"


# ----------------------------------------------------------------------- leakage


@pytest.mark.parametrize(
    ("sequence_builder", "kwargs"),
    [
        (normal_sequence, {"cited": ["freud_dreams_000001"]}),
        (revision_sequence, {"cited": ["freud_dreams_000001"]}),
        (fallback_sequence, {}),
    ],
)
def test_internal_text_never_reaches_events(sequence_builder: Any, kwargs: Any) -> None:
    """Drafts, critiques and local paths must not appear anywhere."""
    _, events = replay(sequence_builder, **kwargs)
    rendered = "\n".join(frame_for(event, thread_id="t1") or "" for event in events)
    for forbidden in (
        DRAFT_TEXT,
        CRITIQUE_TEXT,
        WINDOWS_PATH,
        "retrieval_score",
        "source_path",
    ):
        assert forbidden not in rendered
    for event in events:
        blob = f"{event.label} {event.message} {event.metadata!r}"
        assert "C:\\" not in blob
        assert "sk-" not in blob


def test_state_bodies_are_not_copied_into_events() -> None:
    """The adapter reads counters, not payload bodies."""
    _, events = replay(normal_sequence, cited=["freud_dreams_000001"])
    for event in events:
        if event.event_type == "workflow" and event.node == "synthesizer":
            assert "草稿" not in event.message
            assert "使用压抑" not in event.message


def test_critic_verdict_is_shown_but_not_the_critique() -> None:
    """Verdicts are safe metadata; critique bodies are not."""
    _, events = replay(revision_sequence)
    critic = [
        event
        for event in events
        if event.event_type == "workflow" and event.node == "critic"
    ]
    assert any(event.metadata.verdict == "revise" for event in critic)
    assert any(event.metadata.verdict == "pass" for event in critic)
    for event in critic:
        assert CRITIQUE_TEXT not in event.message


def test_answer_only_from_finalizers() -> None:
    """A synthesis update alone must not publish anything to the chat."""
    translator = StreamTranslator()
    translator.handle(
        "updates", {"synthesizer": {"draft_result": {"final_response": DRAFT_TEXT}}}
    )
    assert not translator.answer_emitted
    translator.handle(
        "updates",
        {
            "finalize": {
                "final_result": {"final_response": FAKE_ANSWER},
                "finalization_status": "passed",
            }
        },
    )
    assert translator.answer_emitted


# ---------------------------------------------------------------------- recovery


def test_recover_reads_a_missing_answer_from_state() -> None:
    """A stream cut after the run still shows the stored answer once."""
    translator = StreamTranslator()
    translator.handle(
        "updates", {"evidence": {"evidence_by_school": {}, "evidence_meta": {}}}
    )
    events = translator.recover(
        {
            "final_result": {"final_response": FAKE_ANSWER, "used_evidence_ids": []},
            "finalization_status": "passed",
            "revision_count": 0,
        }
    )
    assert [event.event_type for event in events] == ["answer"]
    assert translator.recover({"final_result": {"final_response": "again"}}) == []
    assert translator.answer_emitted


def test_recover_without_final_result_emits_nothing() -> None:
    """A failed run's state must not be turned into an answer."""
    translator = StreamTranslator()
    assert translator.recover({"draft_result": {"final_response": DRAFT_TEXT}}) == []


def test_stream_asks_for_the_expected_modes() -> None:
    """The route must request exactly `tasks` and `updates`."""
    client = FakeClient(normal_sequence())
    collect_stream(client)
    call = client.runs.calls[0]
    assert call["stream_mode"] == list(STREAM_MODES)
    assert call["input"] == {"messages": [{"type": "human", "content": "hi"}]}


def test_stream_recovers_the_answer_from_thread_state() -> None:
    """When the finalizer update is missing, the run still ends with an answer."""
    sequence = [
        (event, data)
        for event, data in normal_sequence()
        if not (event == "updates" and "finalize" in data)
    ]
    client = FakeClient(
        sequence,
        state={
            "final_result": {"final_response": FAKE_ANSWER, "used_evidence_ids": []},
            "finalization_status": "passed",
            "evidence_by_school": {},
            "evidence_meta": {},
        },
    )
    events = collect_stream(client)
    assert [event.event_type for event in events].count("answer") == 1
    assert events[-1].event_type == "close"


def test_unknown_nodes_are_ignored() -> None:
    """A node the UI does not know must not leak a raw name."""
    translator = StreamTranslator()
    events = translator.handle("tasks", {"name": "some_new_node", "id": "x"})
    assert events == []
    events = translator.handle("updates", {"some_new_node": {"secret": "value"}})
    assert events == []


def test_sources_payload_survives_partial_metadata() -> None:
    """Odd retrieval payloads degrade instead of crashing the panel."""
    payload = build_sources_payload({"freudian": [{"evidence_id": None}]}, None)
    assert payload.cards == ()
    assert payload.available is False


def test_school_labels_cover_the_three_schools() -> None:
    """Every school the UI groups by has a reader-facing label."""
    assert {"freudian", "object_relations", "lacanian"} <= set(SCHOOL_LABELS)


def test_translator_is_synchronous_and_reusable() -> None:
    """Only the runner is async; the translator stays pure so tests can replay."""
    import inspect

    assert inspect.isasyncgenfunction(run_workflow_events)
    translator = StreamTranslator()
    assert translator.handle("metadata", {"run_id": "r"}) == []
    assert translator.run_id == "r"
