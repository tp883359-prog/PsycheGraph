"""Finalizer and safe finalizer: the only nodes that write to the chat.

`finalize` is used when the Critic accepted the draft. `safe_finalize` is used
when the revision budget is exhausted and the Critic still wants a revision:
instead of showing a draft that was judged unsupported, it emits a conservative
answer that keeps only what is safe (observations, limitations, questions).

Neither node calls a model.
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from react_agent.agents.context import draft_result, evidence_catalogue
from react_agent.rag.citations import render_sources_section
from react_agent.schemas import EvidenceItem
from react_agent.state import SPECIALIST_PERSPECTIVES, PsycheGraphState

logger = logging.getLogger(__name__)

SAFE_FALLBACK_INTRO = (
    "这次的读解没有达到可以展示的标准，所以我不给出理论结论，"
    "也不会把未被支持的判断写成事实。下面只保留这次分析里仍然成立的部分。"
)
"""Opening line of the fallback answer; names no internal agent or draft."""


def used_evidence_items(
    state: PsycheGraphState, draft: dict[str, Any]
) -> list[EvidenceItem]:
    """Return the evidence items a draft actually cites.

    Args:
        state: Current graph state.
        draft: Draft (or final) synthesis payload.

    Returns:
        The cited items, in citation order; unknown ids are skipped because the
        validator reports them separately.
    """
    catalogue = {item.evidence_id: item for item in evidence_catalogue(state)}
    return [
        catalogue[str(evidence_id)]
        for evidence_id in draft.get("used_evidence_ids") or []
        if str(evidence_id) in catalogue
    ]


def render_visible_answer(draft: dict[str, Any], items: list[EvidenceItem]) -> str:
    """Build the user-visible text from a synthesis payload.

    The prose comes from the model, the source block is rendered from real index
    metadata: a citation list can therefore never contain invented authors,
    titles or page numbers.

    Args:
        draft: Accepted synthesis payload.
        items: Evidence items the synthesis relied on.

    Returns:
        `final_response` plus, when sources were used, the rendered source list.
    """
    response = str(draft.get("final_response", ""))
    sources = render_sources_section(items)
    return f"{response}\n\n{sources}" if sources else response


def _dedupe(values: list[str], limit: int) -> list[str]:
    """Return the first `limit` distinct non-empty strings.

    Args:
        values: Candidate strings.
        limit: Maximum length of the result.

    Returns:
        The de-duplicated strings, order preserved.
    """
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return result[:limit]


def build_safe_fallback(
    state: PsycheGraphState, critique: dict[str, Any]
) -> dict[str, Any]:
    """Build a conservative synthesis payload from data that is safe to show.

    Only three things are kept: what the user actually provided (observations),
    the stated limits of the analysis, and the questions that would move it
    forward. No theory conclusion survives, because the Critic judged the
    synthesis unsupported.

    Args:
        state: Current graph state.
        critique: The last `CritiqueResult` payload (used only for logging).

    Returns:
        A `SynthesisResult`-shaped JSON dict with a fallback answer.
    """
    results = state.get("specialist_results") or {}
    observations: list[str] = []
    limitations: list[str] = []
    questions: list[str] = []
    for school in SPECIALIST_PERSPECTIVES:
        result = results.get(school)
        if not isinstance(result, dict):
            continue
        observations.extend(str(item) for item in result.get("observations") or [])
        limitations.extend(str(item) for item in result.get("limitations") or [])
        questions.extend(str(item) for item in result.get("questions") or [])
    draft = state.get("draft_result") or {}
    limitations.extend(str(item) for item in draft.get("limitations") or [])
    questions.extend(str(item) for item in draft.get("follow_up_questions") or [])

    kept_observations = _dedupe(observations, 6)
    kept_limitations = _dedupe(limitations, 6)
    kept_questions = _dedupe(questions, 3)
    plan = state.get("supervisor_plan") or {}
    diagnosis_requested = bool(plan.get("clinical_diagnosis_requested", False))

    blocks = [SAFE_FALLBACK_INTRO]
    if kept_observations:
        blocks.append(
            "材料中出现的内容：\n"
            + "\n".join(f"- {item}" for item in kept_observations)
        )
    if kept_limitations:
        blocks.append(
            "这次分析的边界：\n" + "\n".join(f"- {item}" for item in kept_limitations)
        )
    if kept_questions:
        blocks.append(
            "如果想继续，需要你补充：\n"
            + "\n".join(f"- {item}" for item in kept_questions)
        )
    if diagnosis_requested:
        blocks.append(
            "关于临床判断：本系统只能做精神分析理论层面的文本讨论，不能进行临床诊断，"
            "也不会判断你是否患有某种障碍。"
        )
    logger.info(
        "safe fallback used after %d issue(s) in the last critique",
        len(critique.get("issues") or []),
    )
    return {
        "common_ground": [],
        "differences": [],
        "integrated_interpretation": "",
        "limitations": kept_limitations,
        "follow_up_questions": kept_questions,
        "clinical_diagnosis_refused": diagnosis_requested,
        "used_evidence_ids": [],
        "final_response": "\n\n".join(blocks),
    }


async def finalize_node(
    state: PsycheGraphState,
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    """Publish the accepted draft as the single user-visible message.

    Args:
        state: Current graph state, after a `pass` verdict.
        config: Optional runnable config (unused: no model call happens here).

    Returns:
        Partial state update with `final_result`, `finalization_status` and one
        `AIMessage`.
    """
    draft = draft_result(state)
    count = int(state.get("revision_count", 0) or 0)
    items = used_evidence_items(state, draft)
    visible = render_visible_answer(draft, items)
    status = "revised_and_passed" if count > 0 else "passed"
    logger.info(
        "finalize: status=%s revisions=%d sources=%d", status, count, len(items)
    )
    return {
        "final_result": draft,
        "finalization_status": status,
        "messages": [AIMessage(content=visible)],
    }


async def safe_finalize_node(
    state: PsycheGraphState,
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    """Publish a conservative fallback when the draft cannot be shown.

    Args:
        state: Current graph state, after the revision budget was exhausted.
        config: Optional runnable config (unused: no model call happens here).

    Returns:
        Partial state update with the fallback `final_result`,
        `finalization_status="safe_fallback"` and one `AIMessage`.
    """
    critique = state.get("critique") or {}
    fallback = build_safe_fallback(state, critique)
    return {
        "final_result": fallback,
        "finalization_status": "safe_fallback",
        "messages": [AIMessage(content=str(fallback["final_response"]))],
    }
