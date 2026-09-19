"""Critic node tests: one narrow reviewer, never a fourth analyst."""

import asyncio
from typing import Any, cast

import pytest
from langchain_core.messages import HumanMessage

from react_agent.agents.context import InvalidEvidenceReferenceError
from react_agent.agents.critic import (
    CRITIC_SYSTEM_PROMPT,
    critic_node,
    deterministic_errors,
    enforce_deterministic_errors,
)
from react_agent.schemas import CritiqueResult
from react_agent.state import SPECIALIST_PERSPECTIVES
from tests.unit_tests.conftest import (
    FakeChatModel,
    make_critic_issue,
    make_critique,
    make_evidence_by_school,
    make_plan,
    make_school_analysis,
    make_synthesis,
)


def _state(**overrides: Any) -> dict[str, Any]:
    """Build a state ready for review."""
    state: dict[str, Any] = {
        "messages": [HumanMessage(content="我梦见水。")],
        "supervisor_plan": make_plan(),
        "evidence_by_school": make_evidence_by_school(),
        "specialist_results": {
            school: make_school_analysis(school) for school in SPECIALIST_PERSPECTIVES
        },
        "draft_result": make_synthesis(),
        "deterministic_issues": [],
        "revision_count": 0,
    }
    state.update(overrides)
    return state


def test_critic_node_stores_only_the_critique(fake_model: FakeChatModel) -> None:
    update = asyncio.run(critic_node(_state()))

    assert set(update) == {"critique"}
    critique = cast("dict[str, Any]", update["critique"])
    assert critique["verdict"] == "pass"
    assert critique["clinical_safety_ok"] is True
    assert fake_model.call_names() == ["CritiqueResult"]


def test_critic_prompt_carries_the_review_material(fake_model: FakeChatModel) -> None:
    asyncio.run(
        critic_node(
            _state(
                draft_result=make_synthesis(final_response="DRAFT-IN-REVIEW"),
                deterministic_issues=[
                    {
                        "category": "citation_rendering_error",
                        "message": "ISSUE-FROM-CODE",
                        "related_agent": "synthesizer",
                        "related_evidence_ids": [],
                        "severity": "error",
                    }
                ],
            )
        )
    )

    prompt = fake_model.calls_for(CritiqueResult)[0]
    assert prompt[0].content == CRITIC_SYSTEM_PROMPT
    joined = "\n".join(str(message.content) for message in prompt)
    assert "DRAFT-IN-REVIEW" in joined
    assert "ISSUE-FROM-CODE" in joined
    assert "你不是第四个分析师" in joined
    assert "不输出任何分数" in joined
    assert joined.rstrip().endswith("我梦见水。")


def test_deterministic_error_forces_a_revise_verdict(
    fake_model: FakeChatModel,
) -> None:
    """A model that passes a broken draft must not be trusted."""
    fake_model.overrides = {CritiqueResult: make_critique(verdict="pass")}
    update = asyncio.run(
        critic_node(
            _state(
                draft_result=make_synthesis(used_evidence_ids=["ghost_000001"]),
                deterministic_issues=[
                    {
                        "category": "invalid_evidence_reference",
                        "message": "draft cites ghost_000001",
                        "related_agent": "synthesizer",
                        "related_evidence_ids": ["ghost_000001"],
                        "severity": "error",
                    }
                ],
            )
        )
    )

    critique = cast("dict[str, Any]", update["critique"])
    assert critique["verdict"] == "revise"
    assert critique["evidence_grounding_ok"] is False
    assert any(issue["category"] == "evidence_support" for issue in critique["issues"])
    assert critique["revision_instructions"]


def test_warnings_do_not_force_a_revision(fake_model: FakeChatModel) -> None:
    fake_model.overrides = {CritiqueResult: make_critique(verdict="pass")}
    update = asyncio.run(
        critic_node(
            _state(
                deterministic_issues=[
                    {
                        "category": "state_consistency",
                        "message": "a note, not a blocker",
                        "related_agent": None,
                        "related_evidence_ids": [],
                        "severity": "warning",
                    }
                ]
            )
        )
    )

    assert cast("dict[str, Any]", update["critique"])["verdict"] == "pass"


def test_enforcement_keeps_the_reported_issue(
    fake_model: FakeChatModel,
) -> None:
    """An error the model did report is not duplicated."""
    error = {
        "category": "clinical_safety_mismatch",
        "message": "draft diagnoses the user",
        "related_agent": "synthesizer",
        "related_evidence_ids": [],
        "severity": "error",
    }
    fake_model.overrides = {
        CritiqueResult: make_critique(
            verdict="revise",
            issues=[
                make_critic_issue(
                    category="clinical_safety",
                    description="draft diagnoses the user",
                )
            ],
            clinical_safety_ok=False,
        )
    }
    update = asyncio.run(critic_node(_state(deterministic_issues=[error])))

    critique = cast("dict[str, Any]", update["critique"])
    assert len(critique["issues"]) == 1
    assert critique["clinical_safety_ok"] is False


def test_hallucinated_critic_evidence_id_is_rejected(
    fake_model: FakeChatModel,
) -> None:
    fake_model.overrides = {
        CritiqueResult: make_critique(
            verdict="revise",
            issues=[make_critic_issue(related_evidence_ids=["invented_000001"])],
        )
    }

    with pytest.raises(InvalidEvidenceReferenceError, match="invented_000001"):
        asyncio.run(critic_node(_state()))


def test_critic_evidence_ids_are_resolved_to_real_chunks(
    fake_model: FakeChatModel,
) -> None:
    fake_model.overrides = {
        CritiqueResult: make_critique(
            verdict="revise",
            issues=[make_critic_issue(related_evidence_ids=["Lacanian_Note_000000."])],
            evidence_grounding_ok=False,
        )
    }
    update = asyncio.run(critic_node(_state()))

    critique = cast("dict[str, Any]", update["critique"])
    assert critique["issues"][0]["related_evidence_ids"] == ["lacanian_note_000000"]


def test_an_ambiguous_critic_evidence_id_is_rejected(
    fake_model: FakeChatModel,
) -> None:
    """ "note_000000" could mean three different chunks, so it is not guessed."""
    fake_model.overrides = {
        CritiqueResult: make_critique(
            verdict="revise",
            issues=[make_critic_issue(related_evidence_ids=["note_000000"])],
        )
    }

    with pytest.raises(InvalidEvidenceReferenceError, match="note_000000"):
        asyncio.run(critic_node(_state()))


def test_deterministic_errors_helper_reads_only_errors() -> None:
    state = _state(
        deterministic_issues=[
            {"category": "state_consistency", "message": "a", "severity": "warning"},
            {
                "category": "state_consistency",
                "message": "b",
                "severity": "error",
                "related_evidence_ids": [],
                "related_agent": None,
            },
        ]
    )

    errors = deterministic_errors(state)

    assert [error["message"] for error in errors] == ["b"]


def test_enforcement_is_a_no_op_without_errors() -> None:
    payload = make_critique()
    assert enforce_deterministic_errors(_state(), payload) == payload


def test_critic_never_writes_to_messages(fake_model: FakeChatModel) -> None:
    update = asyncio.run(critic_node(_state()))

    assert "messages" not in update
