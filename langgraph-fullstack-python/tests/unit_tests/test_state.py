"""Tests for the PsycheGraph state contract."""

import operator
from typing import Annotated, get_args, get_origin, get_type_hints

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph.message import add_messages

from react_agent.graph import graph
from react_agent.state import SPECIALIST_PERSPECTIVES, PsycheGraphState


def _annotation(field: str) -> object:
    return get_type_hints(PsycheGraphState, include_extras=True)[field]


def _raw_annotation(field: str) -> object:
    """Return the annotation as written, without langgraph's unwrapping."""
    return PsycheGraphState.__annotations__[field]


def _reducer(field: str) -> object:
    """Return the reducer attached to a state field.

    `get_type_hints` rewrites the annotation, so the raw annotation is used
    here: `messages` is `Annotated[list[BaseMessage], add_messages]` and
    `specialist_results` is `NotRequired[Annotated[dict[...], operator.ior]]`.
    """
    annotation = _raw_annotation(field)
    args = get_args(annotation)
    if args and get_origin(args[0]) is Annotated:
        annotation = args[0]
    return get_args(annotation)[1]


def test_state_declares_the_review_pipeline_fields() -> None:
    assert set(get_type_hints(PsycheGraphState)) == {
        "messages",
        "supervisor_plan",
        "evidence_by_school",
        "evidence_meta",
        "specialist_results",
        "draft_result",
        "deterministic_issues",
        "critique",
        "revision_count",
        "final_result",
        "finalization_status",
    }


def test_working_fields_are_optional_before_execution() -> None:
    assert PsycheGraphState.__required_keys__ == frozenset({"messages"})
    assert PsycheGraphState.__optional_keys__ == frozenset(
        {
            "supervisor_plan",
            "evidence_by_school",
            "evidence_meta",
            "specialist_results",
            "draft_result",
            "deterministic_issues",
            "critique",
            "revision_count",
            "final_result",
            "finalization_status",
        }
    )


def test_review_fields_stay_json_serialisable() -> None:
    """Drafts, issues and critiques live in the state as plain dicts."""
    hints = get_type_hints(PsycheGraphState, include_extras=True)
    for field in ("draft_result", "critique", "final_result"):
        assert "CritiqueResult" not in str(hints[field])
        assert "SynthesisResult" not in str(hints[field])
        assert "dict" in str(hints[field])
    assert "DeterministicIssue" not in str(hints["deterministic_issues"])
    assert "dict" in str(hints["deterministic_issues"])


def test_evidence_fields_stay_json_serialisable() -> None:
    """Evidence lives in the state as plain dicts, never as Pydantic models."""
    hints = get_type_hints(PsycheGraphState, include_extras=True)
    assert "dict" in str(hints["evidence_by_school"])
    assert "EvidenceItem" not in str(hints["evidence_by_school"])
    assert "EvidenceItem" not in str(hints["evidence_meta"])


def test_messages_field_uses_add_messages_reducer() -> None:
    assert _reducer("messages") is add_messages


def test_specialist_results_field_uses_merging_reducer() -> None:
    assert _reducer("specialist_results") is operator.ior


def test_messages_reducer_appends_instead_of_overwriting() -> None:
    reducer = _reducer("messages")
    history: list[BaseMessage] = [HumanMessage(content="我梦见水。")]
    update: list[BaseMessage] = [AIMessage(content="可以多说一点梦里的场景吗？")]
    result = reducer(history, update)  # type: ignore[operator]
    assert [message.content for message in result] == [
        "我梦见水。",
        "可以多说一点梦里的场景吗？",
    ]


def test_specialist_reducer_merges_parallel_writes() -> None:
    reducer = _reducer("specialist_results")
    merged = reducer({"freudian": {"summary": "a"}}, {"lacanian": {"summary": "c"}})  # type: ignore[operator]
    merged = reducer(merged, {"object_relations": {"summary": "b"}})  # type: ignore[operator]
    assert set(merged) == {"freudian", "object_relations", "lacanian"}


def test_specialist_reducer_replaces_the_same_school() -> None:
    reducer = _reducer("specialist_results")
    merged = reducer(
        {"freudian": {"summary": "旧结果"}}, {"freudian": {"summary": "新结果"}}
    )  # type: ignore[operator]
    assert merged == {"freudian": {"summary": "新结果"}}


def test_three_specialist_schools_are_declared() -> None:
    assert SPECIALIST_PERSPECTIVES == ("freudian", "object_relations", "lacanian")


def test_graph_has_the_review_pipeline() -> None:
    topology = graph.get_graph()
    assert set(topology.nodes) == {
        "__start__",
        "supervisor",
        "evidence",
        "freudian",
        "object_relations",
        "lacanian",
        "synthesizer",
        "deterministic_validator",
        "critic",
        "revise_synthesis",
        "finalize",
        "safe_finalize",
        "__end__",
    }
    assert {(edge.source, edge.target) for edge in topology.edges} == {
        ("__start__", "supervisor"),
        ("supervisor", "evidence"),
        ("evidence", "freudian"),
        ("evidence", "object_relations"),
        ("evidence", "lacanian"),
        ("freudian", "synthesizer"),
        ("object_relations", "synthesizer"),
        ("lacanian", "synthesizer"),
        ("synthesizer", "deterministic_validator"),
        ("deterministic_validator", "critic"),
        ("critic", "finalize"),
        ("critic", "revise_synthesis"),
        ("critic", "safe_finalize"),
        ("revise_synthesis", "deterministic_validator"),
        ("finalize", "__end__"),
        ("safe_finalize", "__end__"),
    }


def test_only_the_finalizers_reach_the_end_after_the_critic() -> None:
    topology = graph.get_graph()
    edges = {(edge.source, edge.target) for edge in topology.edges}
    assert {target for source, target in edges if source == "critic"} == {
        "finalize",
        "revise_synthesis",
        "safe_finalize",
    }
    assert {source for source, target in edges if target == "__end__"} == {
        "finalize",
        "safe_finalize",
    }
    # The revision loop goes back through the validator, never straight to the critic.
    assert ("revise_synthesis", "deterministic_validator") in edges


def test_evidence_sits_between_supervisor_and_specialists() -> None:
    topology = graph.get_graph()
    edges = {(edge.source, edge.target) for edge in topology.edges}
    specialists = {"freudian", "object_relations", "lacanian"}
    assert {target for source, target in edges if source == "supervisor"} == {
        "evidence"
    }
    assert {target for source, target in edges if source == "evidence"} == specialists
    assert {
        source for source, target in edges if target == "synthesizer"
    } == specialists
    # Every specialist is downstream of retrieval, so it can never cite
    # evidence the run did not retrieve.
    assert all(
        any(source == "evidence" for source, target in edges if target == school)
        for school in specialists
    )
