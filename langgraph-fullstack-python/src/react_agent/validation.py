"""Deterministic validation of one PsycheGraph run.

Everything in this module is pure code: no model is called, no retrieval
happens, nothing is written to the state. These checks answer the questions that
can be answered by reading data, so the LLM Critic is left with the questions
that genuinely need judgement.

The division of labour is deliberate:

    code    - does this id exist? do the flags contradict the plan? did a draft
              leak into the chat? is a citation label dangling?
    critic  - does the cited passage actually support this claim? is the theory
              attributed to the right school? is a possibility stated as a fact?

Model output must never be the only thing standing between a user and an
unsupported claim, but a model also must not be asked to re-check what a lookup
can decide.
"""

import re
from typing import Any

from react_agent.schemas import DeterministicIssue
from react_agent.state import SPECIALIST_PERSPECTIVES

CITATION_LABEL_RE = re.compile(r"\[E(\d+)\]")
"""Matches the short labels the finalizer renders, for example `[E1]`."""

REQUIRED_DRAFT_FIELDS: tuple[str, ...] = (
    "common_ground",
    "differences",
    "integrated_interpretation",
    "limitations",
    "follow_up_questions",
    "final_response",
)
"""Fields a draft must actually carry before it can be reviewed."""


def _evidence_id_set(state: Any) -> set[str]:
    """Return every evidence id retrieved in this run.

    Args:
        state: Current graph state.

    Returns:
        The union of the three schools' evidence ids.
    """
    ids: set[str] = set()
    for entries in (state.get("evidence_by_school") or {}).values():
        for entry in entries or []:
            if isinstance(entry, dict) and entry.get("evidence_id"):
                ids.add(str(entry["evidence_id"]))
    return ids


def _school_evidence_id_set(state: Any, school: str) -> set[str]:
    """Return the evidence ids one specialist was allowed to cite.

    Args:
        state: Current graph state.
        school: School name.

    Returns:
        The ids offered to that school this run.
    """
    entries = (state.get("evidence_by_school") or {}).get(school) or []
    return {
        str(entry["evidence_id"])
        for entry in entries
        if isinstance(entry, dict) and entry.get("evidence_id")
    }


def _issue(
    category: str,
    message: str,
    *,
    related_agent: str | None = None,
    related_evidence_ids: list[str] | None = None,
    severity: str = "error",
) -> DeterministicIssue:
    """Build one issue with the shared defaults.

    Args:
        category: One of the `DeterministicIssue` categories.
        message: Human-readable explanation.
        related_agent: Node whose output is affected.
        related_evidence_ids: Ids involved, when any.
        severity: `error` or `warning`.

    Returns:
        The validated issue.
    """
    return DeterministicIssue(
        category=category,  # type: ignore[arg-type]
        message=message,
        related_agent=related_agent,
        related_evidence_ids=related_evidence_ids or [],
        severity=severity,  # type: ignore[arg-type]
    )


def collect_deterministic_issues(state: Any) -> list[DeterministicIssue]:
    """Run every code-level check for the current run.

    Args:
        state: Current graph state, after the Synthesizer (or the revision
            synthesizer) wrote `draft_result`.

    Returns:
        All issues found, in a stable order. An empty list means the data is
        internally consistent; it says nothing about the quality of the
        reasoning, which is the Critic's job.
    """
    issues: list[DeterministicIssue] = []
    plan = state.get("supervisor_plan") or {}
    diagnosis_requested = bool(plan.get("clinical_diagnosis_requested", False))
    allowed_ids = _evidence_id_set(state)
    draft = state.get("draft_result")

    if not isinstance(draft, dict):
        issues.append(
            _issue(
                "state_consistency",
                "no draft_result is available for review",
                related_agent="synthesizer",
            )
        )
    else:
        issues.extend(_check_draft(draft, allowed_ids, diagnosis_requested))
    issues.extend(_check_specialists(state, diagnosis_requested))
    issues.extend(_check_messages(state))
    issues.extend(_check_revision_count(state))
    return issues


def _check_draft(
    draft: dict[str, Any],
    allowed_ids: set[str],
    diagnosis_requested: bool,
) -> list[DeterministicIssue]:
    """Check the synthesized draft.

    Args:
        draft: `draft_result` as JSON.
        allowed_ids: Every evidence id retrieved this run.
        diagnosis_requested: Whether the user asked for a clinical judgement.

    Returns:
        Issues about the draft itself.
    """
    issues: list[DeterministicIssue] = []
    used = [str(item) for item in draft.get("used_evidence_ids") or []]
    unknown = sorted({item for item in used if item not in allowed_ids})
    if unknown:
        issues.append(
            _issue(
                "invalid_evidence_reference",
                "draft_result cites evidence ids that were not retrieved this "
                f"run: {', '.join(unknown[:5])}",
                related_agent="synthesizer",
                related_evidence_ids=unknown,
            )
        )

    missing = [field for field in REQUIRED_DRAFT_FIELDS if not draft.get(field)]
    if missing:
        issues.append(
            _issue(
                "state_consistency",
                f"draft_result is missing required content: {', '.join(missing)}",
                related_agent="synthesizer",
            )
        )

    labels = CITATION_LABEL_RE.findall(str(draft.get("final_response") or ""))
    dangling = sorted({int(label) for label in labels if int(label) > len(used)})
    if dangling:
        rendered = ", ".join(f"[E{label}]" for label in dangling)
        issues.append(
            _issue(
                "citation_rendering_error",
                f"final_response contains {rendered}, but only {len(used)} "
                "evidence item(s) are cited by the draft; the label cannot be "
                "resolved to a retrieved passage",
                related_agent="synthesizer",
            )
        )

    if diagnosis_requested and not draft.get("clinical_diagnosis_refused"):
        issues.append(
            _issue(
                "clinical_safety_mismatch",
                "the user asked for a clinical judgement, but the draft does not "
                "set clinical_diagnosis_refused",
                related_agent="synthesizer",
            )
        )
    return issues


def _check_specialists(
    state: Any, diagnosis_requested: bool
) -> list[DeterministicIssue]:
    """Check the three school results against their own contracts.

    Args:
        state: Current graph state.
        diagnosis_requested: Whether the user asked for a clinical judgement.

    Returns:
        Issues about the specialist layer.
    """
    issues: list[DeterministicIssue] = []
    results = state.get("specialist_results") or {}
    for school in SPECIALIST_PERSPECTIVES:
        result = results.get(school)
        if not isinstance(result, dict):
            issues.append(
                _issue(
                    "state_consistency",
                    f"no result was stored for the {school} specialist",
                    related_agent=f"{school}_node",
                )
            )
            continue
        if result.get("perspective") != school:
            issues.append(
                _issue(
                    "perspective_mismatch",
                    f"{school} result is labelled perspective="
                    f"{result.get('perspective')!r}",
                    related_agent=f"{school}_node",
                )
            )
        allowed = _school_evidence_id_set(state, school)
        for interpretation in result.get("interpretations") or []:
            if not isinstance(interpretation, dict):
                continue
            if interpretation.get("perspective") != school:
                issues.append(
                    _issue(
                        "perspective_mismatch",
                        f"{school} interpretation is labelled perspective="
                        f"{interpretation.get('perspective')!r}",
                        related_agent=f"{school}_node",
                    )
                )
            cited = [str(item) for item in interpretation.get("evidence_ids") or []]
            unknown = sorted({item for item in cited if item not in allowed})
            if unknown:
                issues.append(
                    _issue(
                        "invalid_evidence_reference",
                        f"{school} interpretation cites evidence it was never "
                        f"given: {', '.join(unknown[:5])}",
                        related_agent=f"{school}_node",
                        related_evidence_ids=unknown,
                    )
                )
        if diagnosis_requested and not result.get("clinical_diagnosis_refused"):
            issues.append(
                _issue(
                    "clinical_safety_mismatch",
                    f"the user asked for a clinical judgement, but the {school} "
                    "result does not set clinical_diagnosis_refused",
                    related_agent=f"{school}_node",
                )
            )
    return issues


def _check_messages(state: Any) -> list[DeterministicIssue]:
    """Check that no internal draft leaked into the current turn's chat.

    Earlier turns legitimately end with an AI answer, so only what follows the
    last user message is inspected: until the finalizer runs, a turn must
    consist of the user's input and nothing else.

    Args:
        state: Current graph state.

    Returns:
        One issue when an AI message already exists before finalization.
    """
    tail: list[Any] = []
    for message in reversed(list(state.get("messages") or [])):
        if getattr(message, "type", "") == "human":
            break
        tail.append(message)
    leaked = [message for message in tail if getattr(message, "type", "") == "ai"]
    if not leaked:
        return []
    return [
        _issue(
            "state_consistency",
            "an AI message exists before finalization; internal drafts must "
            "never enter the chat history",
            related_agent="graph",
        )
    ]


def _check_revision_count(state: Any) -> list[DeterministicIssue]:
    """Check the revision counter is a sane non-negative integer.

    Args:
        state: Current graph state.

    Returns:
        One issue when the counter is not usable for routing.
    """
    count = state.get("revision_count", 0)
    if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
        return []
    return [
        _issue(
            "state_consistency",
            f"revision_count is not a non-negative integer: {count!r}",
            related_agent="graph",
        )
    ]
