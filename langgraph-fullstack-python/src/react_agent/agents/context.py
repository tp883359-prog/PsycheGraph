"""Prompt construction shared by the PsycheGraph agents.

Every agent receives the same conversation, but each may additionally receive
the Supervisor plan or the other schools' results. These helpers keep that
assembly in one place so the agent modules stay focused on their role.
"""

from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage

from react_agent.rag.citations import describe_source
from react_agent.rag.retriever import evidence_items_from_state
from react_agent.schemas import EvidenceItem
from react_agent.state import SPECIALIST_PERSPECTIVES, PsycheGraphState

SCHOOL_LABELS: dict[str, str] = {
    "freudian": "freudian（弗洛伊德视角）",
    "object_relations": "object_relations（客体关系视角）",
    "lacanian": "lacanian（拉康视角）",
}
"""Display label for each school, used when other agents read its result."""

EVIDENCE_RULES = (
    "文献证据使用规则：\n"
    "- evidence_ids 只能从下面列出的 evidence_id 中选择；一个都不许编造。\n"
    "- 没有相应文献支持的解释，evidence_ids 留空，并在 uncertainty 中说明该读解缺少本地文献支持。\n"
    "- textual_basis 只写用户提供的文本；文献内容是理论依据，不是用户的经历或事实。\n"
    "- 文献里讨论某种病理性机制，不代表用户具有该机制；不得据此推断用户的心理状态。\n"
    "- 不要把文献原文大段抄进回答；用自己的话解释，引用由系统统一渲染。"
)
"""Contract that every specialist must follow when citing theory evidence."""

NO_EVIDENCE_NOTICE = (
    "本次没有可用的本地文献证据（知识库为空或未建立索引）。"
    "因此 evidence_ids 必须为空列表，并在 uncertainty 或 limitations 中说明"
    "当前解释没有本地文献支持。不得编造任何 evidence_id 或文献出处。"
)
"""Shown to specialists when retrieval produced nothing for their school."""


class PerspectiveMismatchError(ValueError):
    """Raised when a specialist answers as a different school than assigned."""


class InvalidEvidenceReferenceError(ValueError):
    """Raised when a model cites an evidence id that was never offered to it.

    Silently keeping a hallucinated citation would let invented literature reach
    the user, so the node fails loudly instead.
    """


def cited_evidence_ids(analysis: Any) -> list[str]:
    """Return every evidence id cited across a school analysis.

    Args:
        analysis: Validated `SchoolAnalysis` output.

    Returns:
        Cited ids in the order the interpretations list them.
    """
    ids: list[str] = []
    for interpretation in getattr(analysis, "interpretations", None) or []:
        ids.extend(getattr(interpretation, "evidence_ids", None) or [])
    return ids


def collect_invalid_evidence_ids(cited: list[str], allowed: set[str]) -> list[str]:
    """Return the cited ids that were not offered to this agent.

    Args:
        cited: Ids the model returned.
        allowed: Ids supplied in the prompt.

    Returns:
        The unknown ids, in the order they were cited.
    """
    return [evidence_id for evidence_id in cited if evidence_id not in allowed]


def _normalise_evidence_id(value: str) -> str:
    """Lower-case an id and drop everything that is not alphanumeric.

    Args:
        value: Raw id (as written by the model or by the loader).

    Returns:
        A comparison key.
    """
    return "".join(ch for ch in value.lower() if ch.isalnum())


def resolve_evidence_id(cited: str, allowed: set[str]) -> str | None:
    """Map a cited id onto an offered id, or return None.

    Models routinely reproduce an id with a small formatting error (a dropped
    prefix, different case, added punctuation) while clearly pointing at a
    passage they were given. Because the resolution only ever picks an *offered*
    id, accepting such a citation cannot introduce invented literature; anything
    that does not resolve uniquely is still rejected.

    Args:
        cited: Id string returned by the model.
        allowed: Ids actually supplied in the prompt.

    Returns:
        The matching allowed id, or None when the citation is ambiguous or
        genuinely unknown.
    """
    raw = str(cited).strip()
    if not raw:
        return None
    if raw in allowed:
        return raw
    key = _normalise_evidence_id(raw)
    if not key:
        return None
    candidates = [
        candidate for candidate in allowed if candidate.lower() == raw.lower()
    ]
    if not candidates:
        candidates = [
            candidate
            for candidate in allowed
            if _normalise_evidence_id(candidate) == key
            or candidate.lower().endswith(raw.lower())
        ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def rewrite_cited_evidence_ids(
    result: dict[str, Any],
    allowed: set[str],
    agent_name: str,
) -> None:
    """Replace cited ids with the offered ids they refer to, in place.

    Args:
        result: JSON specialist result; each interpretation's `evidence_ids` is
            rewritten with resolved ids and de-duplicated.
        allowed: Ids supplied to this agent.
        agent_name: Name used in the error message.

    Raises:
        InvalidEvidenceReferenceError: If a cited id cannot be resolved to an
            offered id, so no invented citation can reach the user.
    """
    unresolved: list[str] = []
    for interpretation in result.get("interpretations") or []:
        if not isinstance(interpretation, dict):
            continue
        resolved: list[str] = []
        for cited in interpretation.get("evidence_ids") or []:
            match = resolve_evidence_id(str(cited), allowed)
            if match is None:
                unresolved.append(str(cited))
            elif match not in resolved:
                resolved.append(match)
        interpretation["evidence_ids"] = resolved
    if unresolved:
        raise InvalidEvidenceReferenceError(
            f"{agent_name} cited {len(unresolved)} unknown evidence id(s): "
            f"{', '.join(unresolved[:5])}"
        )


def rewrite_synthesizer_evidence_ids(
    result: dict[str, Any],
    allowed: set[str],
    agent_name: str = "synthesizer_node",
) -> None:
    """Resolve the synthesizer's `used_evidence_ids` against the catalogue.

    Args:
        result: JSON synthesis result; `used_evidence_ids` is rewritten in place.
        allowed: Every evidence id retrieved this run.
        agent_name: Name used in the error message.

    Raises:
        InvalidEvidenceReferenceError: If a cited id cannot be resolved.
    """
    unresolved: list[str] = []
    resolved: list[str] = []
    for cited in result.get("used_evidence_ids") or []:
        match = resolve_evidence_id(str(cited), allowed)
        if match is None:
            unresolved.append(str(cited))
        elif match not in resolved:
            resolved.append(match)
    result["used_evidence_ids"] = resolved
    if unresolved:
        raise InvalidEvidenceReferenceError(
            f"{agent_name} cited {len(unresolved)} unknown evidence id(s): "
            f"{', '.join(unresolved[:5])}"
        )


def validate_evidence_references(
    cited: list[str],
    allowed: set[str],
    agent_name: str,
) -> None:
    """Reject citations of evidence the agent never received.

    Args:
        cited: Ids the model returned.
        allowed: Ids supplied in the prompt.
        agent_name: Name used in the error message.

    Raises:
        InvalidEvidenceReferenceError: If any id is unknown.
    """
    invalid = collect_invalid_evidence_ids(cited, allowed)
    if invalid:
        raise InvalidEvidenceReferenceError(
            f"{agent_name} cited {len(invalid)} unknown evidence id(s): "
            f"{', '.join(invalid[:5])}"
        )


def check_perspective(analysis: Any, expected: str, agent_name: str) -> None:
    """Ensure a specialist answered inside its own school.

    Args:
        analysis: Validated model output carrying a `perspective` attribute.
        expected: School the specialist is assigned to.
        agent_name: Node name used in the error message.

    Raises:
        PerspectiveMismatchError: If the model labelled itself as another school.
    """
    if analysis.perspective != expected:
        raise PerspectiveMismatchError(
            f"{agent_name} returned perspective={analysis.perspective!r}, "
            f"expected {expected!r}"
        )


def state_messages(state: PsycheGraphState) -> list[BaseMessage]:
    """Return the conversation as a mutable list.

    Args:
        state: Current graph state.

    Returns:
        The messages of the current thread, oldest first.
    """
    return list(state["messages"])


def supervisor_plan(state: PsycheGraphState) -> dict[str, Any]:
    """Return the current run's Supervisor plan.

    Args:
        state: Current graph state, after the Supervisor node ran.

    Returns:
        The plan as a JSON dict.

    Raises:
        ValueError: If the state carries no plan, which means a specialist ran
            without the Supervisor.
    """
    plan = state.get("supervisor_plan")
    if plan is None:
        raise ValueError("supervisor_plan is missing from the graph state")
    return plan


def supervisor_plan_message(state: PsycheGraphState) -> SystemMessage:
    """Render the Supervisor plan as a system message for a specialist.

    Args:
        state: Current graph state, after the Supervisor node ran.

    Returns:
        A system message listing the plan and the focus lines for every school.
    """
    plan = supervisor_plan(state)
    lines = ["Supervisor 的本次计划："]
    for key in (
        "task_summary",
        "analysis_focus",
        "freudian_focus",
        "object_relations_focus",
        "lacanian_focus",
        "synthesis_goal",
    ):
        lines.append(f"- {key}: {plan.get(key, '')}")
    lines.append(
        "- clinical_diagnosis_requested: "
        f"{bool(plan.get('clinical_diagnosis_requested', False))}"
    )
    lines.append(
        "这些重点是分析方向，不是关于用户的事实；用户没有提供的内容不得当成已知信息。"
    )
    return SystemMessage(content="\n".join(lines))


def specialist_results_mapping(state: PsycheGraphState) -> dict[str, dict[str, Any]]:
    """Return the school results collected so far in this run.

    Args:
        state: Current graph state, after the specialist nodes ran.

    Returns:
        A mapping from school name to that school's JSON result.

    Raises:
        ValueError: If the state carries no school results.
    """
    results = state.get("specialist_results")
    if results is None:
        raise ValueError("specialist_results is missing from the graph state")
    return results


def specialist_results_message(state: PsycheGraphState) -> SystemMessage:
    """Render the three specialist results for the Synthesizer.

    Args:
        state: Current graph state, after all specialist nodes ran.

    Returns:
        A system message containing each school's structured result.
    """
    results = specialist_results_mapping(state)
    blocks: list[str] = ["三个 Specialist 的结构化结果："]
    for perspective in SPECIALIST_PERSPECTIVES:
        label = SCHOOL_LABELS[perspective]
        result = results.get(perspective)
        if not isinstance(result, dict):
            blocks.append(f"\n## {label}\n（缺少结果）")
            continue
        blocks.append(f"\n## {label}")
        blocks.append(f"observations: {result.get('observations', [])}")
        blocks.append(f"summary: {result.get('summary', '')}")
        for interpretation in result.get("interpretations", []) or []:
            if not isinstance(interpretation, dict):
                continue
            blocks.append(f"- perspective: {interpretation.get('perspective', '')}")
            blocks.append(f"  claim: {interpretation.get('claim', '')}")
            blocks.append(f"  textual_basis: {interpretation.get('textual_basis', [])}")
            blocks.append(f"  uncertainty: {interpretation.get('uncertainty', '')}")
        blocks.append(f"limitations: {result.get('limitations', [])}")
        blocks.append(f"questions: {result.get('questions', [])}")
        blocks.append(
            "clinical_diagnosis_refused: "
            f"{bool(result.get('clinical_diagnosis_refused', False))}"
        )
    return SystemMessage(content="\n".join(blocks))


def school_evidence(state: PsycheGraphState, school: str) -> list[EvidenceItem]:
    """Return the retrieved theory passages for one school.

    Args:
        state: Current graph state, after the evidence node ran.
        school: School whose shelf is read.

    Returns:
        Validated evidence items; empty when nothing was retrieved.
    """
    return evidence_items_from_state(state, school)


def allowed_evidence_ids(state: PsycheGraphState, school: str) -> set[str]:
    """Return the ids a specialist is allowed to cite.

    Args:
        state: Current graph state.
        school: School whose evidence was offered to that specialist.

    Returns:
        The set of legal evidence ids for this call.
    """
    return {item.evidence_id for item in school_evidence(state, school)}


def evidence_message(state: PsycheGraphState, school: str) -> SystemMessage:
    """Render one school's theory evidence as a system message.

    Args:
        state: Current graph state, after the evidence node ran.
        school: School whose shelf is read.

    Returns:
        A system message listing the evidence ids, their real source line and the
        passage text, followed by the citation rules. When nothing was retrieved
        it states that plainly instead of pretending there is literature.
    """
    items = school_evidence(state, school)
    if not items:
        return SystemMessage(content=NO_EVIDENCE_NOTICE)
    lines = ["本地理论知识库检索结果（这些是该学派资料，不是用户提供的内容）："]
    for item in items:
        lines.append(
            f"- evidence_id: {item.evidence_id}\n"
            f"  来源: {describe_source(item)}\n"
            f"  原文片段: {item.text}"
        )
    lines.append(EVIDENCE_RULES)
    return SystemMessage(content="\n".join(lines))


def all_evidence_ids(state: PsycheGraphState) -> set[str]:
    """Return every evidence id offered to any specialist this run.

    Args:
        state: Current graph state, after the evidence node ran.

    Returns:
        Union of the three schools' evidence ids.
    """
    ids: set[str] = set()
    for school in SPECIALIST_PERSPECTIVES:
        ids.update(allowed_evidence_ids(state, school))
    return ids


def evidence_catalogue(state: PsycheGraphState) -> list[EvidenceItem]:
    """Return all retrieved evidence items once, de-duplicated by id.

    Args:
        state: Current graph state.

    Returns:
        Evidence items ordered by school, then by retrieval order.
    """
    items: list[EvidenceItem] = []
    seen: set[str] = set()
    for school in SPECIALIST_PERSPECTIVES:
        for item in school_evidence(state, school):
            if item.evidence_id in seen:
                continue
            seen.add(item.evidence_id)
            items.append(item)
    return items


def synthesis_evidence_message(state: PsycheGraphState) -> SystemMessage:
    """Render the evidence catalogue for the Synthesizer.

    Only metadata is shown: the Synthesizer cites passages the specialists
    already used, it does not re-read the corpus.

    Args:
        state: Current graph state, after the evidence node ran.

    Returns:
        A system message with each evidence id, its source line and school, plus
        the citation rules.
    """
    items = evidence_catalogue(state)
    if not items:
        return SystemMessage(content=NO_EVIDENCE_NOTICE)
    lines = ["本地理论知识库中的可用文献（只能引用这些 evidence_id）："]
    for item in items:
        lines.append(
            f"- evidence_id: {item.evidence_id} | school={item.school} | "
            f"来源: {describe_source(item)}"
        )
    lines.append(
        "used_evidence_ids 只能从上面的 id 中选择；没有文献支持的结论不要引用。"
        "不要在 final_response 中自己写页码或参考文献格式，系统会统一渲染来源列表。"
    )
    return SystemMessage(content="\n".join(lines))


def draft_result(state: PsycheGraphState) -> dict[str, Any]:
    """Return the current draft synthesis as a JSON dict.

    Args:
        state: Current graph state, after the Synthesizer ran.

    Returns:
        The draft produced by the Synthesizer or the revision step.

    Raises:
        ValueError: If no draft exists, which means the review steps ran before
            the Synthesizer.
    """
    draft = state.get("draft_result")
    if draft is None:
        raise ValueError("draft_result is missing from the graph state")
    return draft


def critique_record(state: PsycheGraphState) -> dict[str, Any]:
    """Return the Critic's verdict as a JSON dict.

    Args:
        state: Current graph state, after the Critic ran.

    Returns:
        The `CritiqueResult` payload.

    Raises:
        ValueError: If the Critic has not run.
    """
    critique = state.get("critique")
    if critique is None:
        raise ValueError("critique is missing from the graph state")
    return critique


def deterministic_issue_records(state: PsycheGraphState) -> list[dict[str, Any]]:
    """Return the code-level findings of this run.

    Args:
        state: Current graph state, after the validator ran.

    Returns:
        The issue payloads, possibly empty.
    """
    return list(state.get("deterministic_issues") or [])


def draft_result_message(state: PsycheGraphState) -> SystemMessage:
    """Render the draft synthesis for review.

    This is the material under review, not instructions: the Critic must treat
    it as a claim to check against the conversation and the evidence.

    Args:
        state: Current graph state, after the Synthesizer ran.

    Returns:
        A system message containing the draft's fields.
    """
    draft = draft_result(state)
    lines = ["待审核的 Synthesizer 草稿（draft_result）："]
    lines.append(f"- common_ground: {draft.get('common_ground', [])}")
    lines.append(f"- differences: {draft.get('differences', [])}")
    lines.append(
        f"- integrated_interpretation: {draft.get('integrated_interpretation', '')}"
    )
    lines.append(f"- limitations: {draft.get('limitations', [])}")
    lines.append(f"- follow_up_questions: {draft.get('follow_up_questions', [])}")
    lines.append(
        "- clinical_diagnosis_refused: "
        f"{bool(draft.get('clinical_diagnosis_refused', False))}"
    )
    lines.append(f"- used_evidence_ids: {draft.get('used_evidence_ids', [])}")
    lines.append(f"- final_response:\n{draft.get('final_response', '')}")
    return SystemMessage(content="\n".join(lines))


def deterministic_issues_message(state: PsycheGraphState) -> SystemMessage:
    """Render the code-level findings for the Critic.

    Args:
        state: Current graph state, after the validator ran.

    Returns:
        A system message listing each finding, or stating that there are none.
    """
    issues = deterministic_issue_records(state)
    if not issues:
        return SystemMessage(
            content="代码级校验（deterministic validation）没有发现问题。"
        )
    lines = [
        "代码级校验（deterministic validation）发现的问题（这些不是你的判断，"
        "是程序对数据的检查结果）："
    ]
    for issue in issues:
        lines.append(
            f"- [{issue.get('severity')}] {issue.get('category')}: "
            f"{issue.get('message')} "
            f"(agent={issue.get('related_agent')}, "
            f"evidence={issue.get('related_evidence_ids', [])})"
        )
    lines.append(
        '任何 severity="error" 的条目都不允许 verdict="pass"；'
        "你需要在 issues 中说明如何修复，或在 revision_instructions 中给出具体修改。"
    )
    return SystemMessage(content="\n".join(lines))


def critique_revision_message(state: PsycheGraphState) -> SystemMessage:
    """Render the Critic's verdict for the revision step.

    Args:
        state: Current graph state, after the Critic ran.

    Returns:
        A system message with the verdict, each issue and the instructions, plus
        the limits of what a revision may do.
    """
    critique = critique_record(state)
    lines = ["Critic 的审核结果（critique）："]
    lines.append(f"- verdict: {critique.get('verdict')}")
    lines.append(f"- summary: {critique.get('summary', '')}")
    lines.append(f"- clinical_safety_ok: {critique.get('clinical_safety_ok')}")
    lines.append(f"- evidence_grounding_ok: {critique.get('evidence_grounding_ok')}")
    lines.append(
        f"- observation_fidelity_ok: {critique.get('observation_fidelity_ok')}"
    )
    issues = critique.get("issues") or []
    if issues:
        lines.append("issues:")
        for issue in issues:
            lines.append(
                f"- [{issue.get('severity')}] {issue.get('category')} "
                f"({issue.get('related_perspective')}): {issue.get('description')}\n"
                f"    affected_claim: {issue.get('affected_claim')}\n"
                f"    evidence: {issue.get('related_evidence_ids', [])}\n"
                f"    revision_instruction: {issue.get('revision_instruction')}"
            )
    instructions = critique.get("revision_instructions") or []
    if instructions:
        lines.append("revision_instructions:")
        for instruction in instructions:
            lines.append(f"- {instruction}")
    lines.append(
        "修订规则：只修复上面指出的问题；不得新增 evidence_id；不得新增用户没有提供的"
        "事实、经历、情绪或诊断；不得引入新的理论来源；可以把过强的结论降级为条件性"
        "表述、标明学派归属、或直接删除。"
    )
    return SystemMessage(content="\n".join(lines))


def rewrite_critic_evidence_ids(
    result: dict[str, Any],
    allowed: set[str],
    agent_name: str = "critic_node",
) -> None:
    """Resolve the Critic's `related_evidence_ids` against the whitelist.

    Args:
        result: JSON critique payload; each issue's `related_evidence_ids` is
            rewritten in place with resolved ids and de-duplicated.
        allowed: Every evidence id retrieved this run.
        agent_name: Name used in the error message.

    Raises:
        InvalidEvidenceReferenceError: If an id cannot be resolved to a
            retrieved passage, so a hallucinated id can never enter the state.
    """
    unresolved: list[str] = []
    for issue in result.get("issues") or []:
        if not isinstance(issue, dict):
            continue
        resolved: list[str] = []
        for cited in issue.get("related_evidence_ids") or []:
            match = resolve_evidence_id(str(cited), allowed)
            if match is None:
                unresolved.append(str(cited))
            elif match not in resolved:
                resolved.append(match)
        issue["related_evidence_ids"] = resolved
    if unresolved:
        raise InvalidEvidenceReferenceError(
            f"{agent_name} cited {len(unresolved)} unknown evidence id(s): "
            f"{', '.join(unresolved[:5])}"
        )
