"""Schema tests for the Phase 7 review contracts."""

import pytest
from pydantic import ValidationError

from react_agent.schemas import CriticIssue, CritiqueResult, DeterministicIssue
from tests.unit_tests.conftest import make_critic_issue, make_critique

CRITIC_CATEGORIES = {
    "observation_fidelity",
    "evidence_support",
    "theoretical_consistency",
    "overinterpretation",
    "clinical_safety",
    "synthesis_quality",
}


def test_critic_issue_accepts_every_declared_category() -> None:
    for category in sorted(CRITIC_CATEGORIES):
        issue = CriticIssue.model_validate(make_critic_issue(category=category))
        assert issue.category == category


def test_critic_issue_rejects_an_unknown_category() -> None:
    with pytest.raises(ValidationError):
        CriticIssue.model_validate(make_critic_issue(category="style"))


def test_critic_issue_requires_a_revision_instruction() -> None:
    payload = make_critic_issue()
    payload.pop("revision_instruction")

    with pytest.raises(ValidationError):
        CriticIssue.model_validate(payload)


def test_critique_result_accepts_pass_and_revise() -> None:
    assert CritiqueResult.model_validate(make_critique()).verdict == "pass"
    assert (
        CritiqueResult.model_validate(
            make_critique(
                verdict="revise",
                issues=[make_critic_issue()],
                revision_instructions=["降级该结论。"],
            )
        ).verdict
        == "revise"
    )


def test_critique_result_rejects_an_unknown_verdict() -> None:
    with pytest.raises(ValidationError):
        CritiqueResult.model_validate(make_critique(verdict="maybe"))


def test_critique_result_requires_all_three_flags() -> None:
    for field in (
        "clinical_safety_ok",
        "evidence_grounding_ok",
        "observation_fidelity_ok",
    ):
        payload = make_critique()
        payload.pop(field)
        with pytest.raises(ValidationError):
            CritiqueResult.model_validate(payload)


def test_critique_result_exposes_no_uncalibrated_scores() -> None:
    fields = set(CritiqueResult.model_fields)

    assert fields == {
        "verdict",
        "issues",
        "summary",
        "revision_instructions",
        "clinical_safety_ok",
        "evidence_grounding_ok",
        "observation_fidelity_ok",
    }
    assert not {"confidence", "score", "probability"} & fields


def test_critique_defaults_are_empty_not_missing() -> None:
    result = CritiqueResult.model_validate(
        {
            "verdict": "pass",
            "summary": "没有问题。",
            "clinical_safety_ok": True,
            "evidence_grounding_ok": True,
            "observation_fidelity_ok": True,
        }
    )

    assert result.issues == []
    assert result.revision_instructions == []


def test_escaped_sequences_are_normalized_in_critic_text() -> None:
    result = CriticIssue.model_validate(
        make_critic_issue(
            description="第一行\\n第二行", revision_instruction="说“够了”"
        )
    )

    assert result.description == "第一行\n第二行"
    assert result.revision_instruction == "说“够了”"


def test_deterministic_issue_defaults_to_an_error() -> None:
    issue = DeterministicIssue(
        category="state_consistency", message="draft_result is missing"
    )

    assert issue.severity == "error"
    assert issue.related_agent is None
    assert issue.related_evidence_ids == []


def test_deterministic_issue_rejects_an_unknown_category() -> None:
    with pytest.raises(ValidationError):
        DeterministicIssue.model_validate(
            {"category": "bad_mood", "message": "x", "severity": "error"}
        )


def test_deterministic_issue_cleans_evidence_ids() -> None:
    issue = DeterministicIssue(
        category="invalid_evidence_reference",
        message="unknown id",
        related_evidence_ids=[" a_000001 ", "", "a_000001"],
    )

    assert issue.related_evidence_ids == ["a_000001"]
