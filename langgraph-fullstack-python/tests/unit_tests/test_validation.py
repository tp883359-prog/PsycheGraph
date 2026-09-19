"""Deterministic validator tests: code decides what code can decide.

Nothing in these tests calls a model: the validator is a pure function over the
state, and every issue it reports is a fact about the data.
"""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from react_agent.schemas import DeterministicIssue
from react_agent.state import SPECIALIST_PERSPECTIVES
from react_agent.validation import collect_deterministic_issues
from tests.unit_tests.conftest import (
    make_evidence_by_school,
    make_plan,
    make_school_analysis,
    make_synthesis,
)


def _state(**overrides: Any) -> dict[str, Any]:
    """Build a consistent state that passes every check until overridden."""
    state: dict[str, Any] = {
        "messages": [HumanMessage(content="我梦见水。")],
        "supervisor_plan": make_plan(),
        "evidence_by_school": make_evidence_by_school(),
        "specialist_results": {
            school: make_school_analysis(school) for school in SPECIALIST_PERSPECTIVES
        },
        "draft_result": make_synthesis(),
        "revision_count": 0,
    }
    state.update(overrides)
    return state


def _categories(state: dict[str, Any]) -> list[str]:
    return sorted(issue.category for issue in collect_deterministic_issues(state))


def test_a_consistent_state_produces_no_issues() -> None:
    assert collect_deterministic_issues(_state()) == []


def test_issues_are_typed_models() -> None:
    issues = collect_deterministic_issues(_state(draft_result=None))

    assert issues
    assert all(isinstance(issue, DeterministicIssue) for issue in issues)
    assert issues[0].severity == "error"


def test_missing_draft_is_reported() -> None:
    assert _categories(_state(draft_result=None)) == ["state_consistency"]


def test_draft_citing_unretrieved_evidence_is_an_error() -> None:
    issues = collect_deterministic_issues(
        _state(draft_result=make_synthesis(used_evidence_ids=["ghost_000001"]))
    )

    assert [issue.category for issue in issues] == ["invalid_evidence_reference"]
    assert issues[0].related_agent == "synthesizer"
    assert issues[0].related_evidence_ids == ["ghost_000001"]
    assert issues[0].severity == "error"


def test_specialist_citing_another_schools_evidence_is_an_error() -> None:
    results = {
        school: make_school_analysis(school) for school in SPECIALIST_PERSPECTIVES
    }
    results["freudian"] = make_school_analysis(
        "freudian",
        interpretations=[
            {
                "perspective": "freudian",
                "claim": "引用不属于本学派的证据。",
                "textual_basis": ["我梦见水"],
                "uncertainty": "无。",
                "evidence_ids": ["lacanian_note_000000"],
            }
        ],
    )

    issues = collect_deterministic_issues(_state(specialist_results=results))

    assert [issue.category for issue in issues] == ["invalid_evidence_reference"]
    assert issues[0].related_agent == "freudian_node"


def test_perspective_mismatch_is_reported_for_result_and_interpretation() -> None:
    results = {
        school: make_school_analysis(school) for school in SPECIALIST_PERSPECTIVES
    }
    results["lacanian"] = make_school_analysis("lacanian")
    results["lacanian"]["perspective"] = "freudian"

    labelled_wrong = make_school_analysis(
        "object_relations",
        interpretations=[
            {
                "perspective": "lacanian",
                "claim": "学派标签写错了。",
                "textual_basis": ["我梦见水"],
                "uncertainty": "无。",
            }
        ],
    )
    results["object_relations"] = labelled_wrong

    categories = _categories(_state(specialist_results=results))

    assert categories == ["perspective_mismatch", "perspective_mismatch"]


def test_missing_specialist_result_is_reported() -> None:
    results = {"freudian": make_school_analysis("freudian")}

    issues = collect_deterministic_issues(_state(specialist_results=results))

    assert [issue.category for issue in issues] == [
        "state_consistency",
        "state_consistency",
    ]


def test_clinical_request_requires_every_refusal_flag() -> None:
    categories = _categories(
        _state(supervisor_plan=make_plan(clinical_diagnosis_requested=True))
    )

    # Three specialists plus the draft must each carry the refusal flag.
    assert categories == ["clinical_safety_mismatch"] * 4


def test_clinical_request_is_satisfied_when_all_flags_are_set() -> None:
    results = {
        school: make_school_analysis(school, clinical_diagnosis_refused=True)
        for school in SPECIALIST_PERSPECTIVES
    }
    state = _state(
        supervisor_plan=make_plan(clinical_diagnosis_requested=True),
        specialist_results=results,
        draft_result=make_synthesis(clinical_diagnosis_refused=True),
    )

    assert collect_deterministic_issues(state) == []


def test_empty_draft_fields_are_reported() -> None:
    issues = collect_deterministic_issues(
        _state(draft_result=make_synthesis(final_response="", limitations=[]))
    )

    assert [issue.category for issue in issues] == ["state_consistency"]
    assert "final_response" in issues[0].message
    assert "limitations" in issues[0].message


def test_dangling_citation_label_is_an_error() -> None:
    issues = collect_deterministic_issues(
        _state(
            draft_result=make_synthesis(
                final_response="这条读解引用 [E1] 与 [E2] 作为依据。",
                used_evidence_ids=[],
            )
        )
    )

    assert [issue.category for issue in issues] == ["citation_rendering_error"]
    assert "[E1]" in issues[0].message and "[E2]" in issues[0].message


def test_citation_label_within_the_used_list_is_accepted() -> None:
    state = _state(
        draft_result=make_synthesis(
            final_response="这条读解引用 [E1] 作为依据。",
            used_evidence_ids=["freudian_note_000000"],
        )
    )

    assert collect_deterministic_issues(state) == []


def test_a_leaked_ai_message_is_an_error() -> None:
    state = _state(
        messages=[
            HumanMessage(content="我梦见水。"),
            AIMessage(content="未经审核的草稿"),
        ]
    )

    issues = collect_deterministic_issues(state)

    assert [issue.category for issue in issues] == ["state_consistency"]
    assert "chat history" in issues[0].message


def test_the_previous_turns_answer_is_not_a_leak() -> None:
    """A multi-turn conversation legitimately ends earlier turns with an AI reply."""
    state = _state(
        messages=[
            HumanMessage(content="我梦见水。"),
            AIMessage(content="上一轮的已发布回答。"),
            HumanMessage(content="水是平静的湖水。"),
        ]
    )

    assert collect_deterministic_issues(state) == []


def test_a_broken_revision_counter_is_reported() -> None:
    assert _categories(_state(revision_count=-1)) == ["state_consistency"]
    assert _categories(_state(revision_count="one")) == ["state_consistency"]


def test_state_without_evidence_is_still_consistent() -> None:
    state = _state(evidence_by_school={})

    assert collect_deterministic_issues(state) == []
