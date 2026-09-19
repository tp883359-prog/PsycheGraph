"""Revision synthesizer: fixes exactly what the Critic named.

A revision does **not** re-run the Supervisor, the evidence node or the three
specialists. The problem Phase 7 targets is in the synthesis layer, and the
evidence was already retrieved this run, so re-running retrieval would only add
latency and could change the grounding under the Critic's feet.

The revision may soften, attribute, delete or state what is missing. It may not
add evidence ids, user facts or new theory.
"""

import logging

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from react_agent.agents.context import (
    all_evidence_ids,
    critique_revision_message,
    draft_result,
    draft_result_message,
    rewrite_synthesizer_evidence_ids,
    specialist_results_message,
    state_messages,
    supervisor_plan_message,
    synthesis_evidence_message,
)
from react_agent.agents.synthesizer import SYNTHESIZER_SYSTEM_PROMPT
from react_agent.llm import build_structured_runnable, invoke_structured
from react_agent.schemas import SynthesisResult
from react_agent.state import PsycheGraphState

logger = logging.getLogger(__name__)

REVISION_ADDENDUM = """
本次是修订（revision），不是第一次综合：
- 只修复 Critic 指出的问题，其余部分尽量保持原样。
- 不得新增 evidence_id，不得新增用户没有提供的事实、经历、情绪或诊断，
  不得引入新的理论来源。
- Critic 指出某个结论缺乏支持时：把它降级为条件性表述、标明它属于哪个学派，
  或直接删除；缺少什么信息就写进 limitations。
- 修订后仍然输出完整的 SynthesisResult（未被指出的字段也要照常填写）。
""".strip()

REVISION_SYSTEM_PROMPT = f"{SYNTHESIZER_SYSTEM_PROMPT}\n\n{REVISION_ADDENDUM}"


def build_revision_runnable() -> object:
    """Return the structured-output runnable for the revision step."""
    return build_structured_runnable(SynthesisResult)


async def revise_synthesis_node(
    state: PsycheGraphState,
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    """Rewrite the draft so that the Critic's issues are fixed.

    Args:
        state: Current graph state; the draft, the critique, the specialist
            results and this run's evidence are read.
        config: Optional LangChain runnable config forwarded to the model.

    Returns:
        Partial state update with the revised draft and an incremented
        `revision_count`.

    Raises:
        InvalidEvidenceReferenceError: If the revision cites evidence that was
            not retrieved this run.
    """
    prompt: list[BaseMessage] = [
        SystemMessage(content=REVISION_SYSTEM_PROMPT),
        supervisor_plan_message(state),
        specialist_results_message(state),
        synthesis_evidence_message(state),
        draft_result_message(state),
        critique_revision_message(state),
        *state_messages(state),
    ]
    _validated, result = await invoke_structured(
        SynthesisResult,
        prompt,
        config=config,
        agent_name="revision_synthesizer",
    )
    rewrite_synthesizer_evidence_ids(result, all_evidence_ids(state))
    previous = draft_result(state)
    count = int(state.get("revision_count", 0) or 0) + 1
    logger.info(
        "revision %d: %d field(s) changed between drafts",
        count,
        sum(1 for key, value in result.items() if previous.get(key) != value),
    )
    return {"draft_result": result, "revision_count": count}
