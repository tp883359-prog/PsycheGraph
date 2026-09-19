"""Critic agent: reviews the synthesis draft without redoing the analysis.

The Critic exists because a real citation is not a correct inference. Phase 6
made it impossible to cite a passage that does not exist; it could not check
whether the passage supports the claim it is attached to. That is a judgement
about reasoning, so it needs a model - but a model with a narrow mandate:

    the Critic reviews, it never analyses.

It may not add theory, may not add facts about the user, may not add evidence
ids and may not re-run retrieval. Its only product is a verdict plus the issues
a revision must fix.
"""

import logging
from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from react_agent.agents.context import (
    all_evidence_ids,
    deterministic_issues_message,
    draft_result_message,
    rewrite_critic_evidence_ids,
    specialist_results_message,
    state_messages,
    supervisor_plan_message,
    synthesis_evidence_message,
)
from react_agent.llm import build_structured_runnable, invoke_structured
from react_agent.schemas import CritiqueResult
from react_agent.state import PsycheGraphState

logger = logging.getLogger(__name__)

CRITIC_SYSTEM_PROMPT = """
你是 PsycheGraph 的 Critic（审核 Agent）。你不是第四个分析师：你不重新做精神分析，
不提出自己的理论读解，不补充用户没有提供的事实，不引用知识库以外的文献。
你只审核 Synthesizer 的草稿（draft_result）。

输入：
- Supervisor 的计划。
- 三个 Specialist 的结构化结果。
- 本地知识库检索到的文献（含 evidence_id 与来源）。
- 代码级校验（deterministic validation）发现的问题。
- 完整对话消息（用户到底写了什么）。
- Synthesizer 的草稿。

审核五个方面（这是你唯一的工作范围）：
1. observation_fidelity：草稿里的“事实”是否真的来自用户？有没有把童年经历、创伤、
   家庭关系、性欲、恐惧、人格特征等用户没有写过的内容当成已观察的事实？
2. evidence_support：文献是否真的支持相应的理论主张？只确认 evidence_id 存在是不够的：
   如果引用的是关于 splitting 的段落，结论却在谈 repression，必须指出这不匹配。
   把理论句子写成对用户的事实判断，同样属于 grounding 问题。
3. theoretical_consistency：三个学派的概念是否被混淆？是否把某个主张错误归因给某个学派？
4. overinterpretation：是否把“一种可能的理解”写成事实判断（例如“这说明你……”）？
   材料很少时，草稿必须保持条件性。
5. clinical_safety：是否出现临床诊断、疾病判断、人格障碍判断、病理概率，或依据极少文本
   推断心理疾病？

判断规则：
- 严格审核，但不要为了显示作用而强行找错。如果草稿有据、学派边界合理、没有安全问题，
  verdict 必须是 "pass"，issues 留空。
- 不要因为“还能写得更详细、更漂亮”就要求修订。只有影响正确性、grounding、理论边界与
  安全的问题才值得标为 severity="error"；措辞建议用 severity="warning"。
- 代码级校验中 severity="error" 的条目不允许 verdict="pass"。
- clinical_safety_ok / evidence_grounding_ok / observation_fidelity_ok 必须如实填写。
- 每条 issue 必须给出可执行的 revision_instruction：降级为条件性表述、标明学派归属、
  删除该结论、或说明缺少什么信息。严禁要求“补充用户没有提供的内容”，
  严禁要求新增 evidence_id 或新的文献。
- related_evidence_ids 只能使用上面真实列出的 evidence_id，一个都不许编造。
- 本阶段最多只允许一次修订，所以只把真正必须修的问题标成 error。
- 不输出任何分数、概率或置信度（例如 0.85、92%）；这些数字没有校准意义。
- 只输出审核结果，不要写给用户的回答。
""".strip()

DETERMINISTIC_TO_CRITIC_CATEGORY: dict[str, str] = {
    "invalid_evidence_reference": "evidence_support",
    "perspective_mismatch": "theoretical_consistency",
    "clinical_safety_mismatch": "clinical_safety",
    "citation_rendering_error": "synthesis_quality",
    "state_consistency": "synthesis_quality",
}
"""How a code-level finding is reported as a critic issue.

A deterministic `error` must never be silently dropped: when the model still
returns `pass`, it is converted into an explicit issue so the verdict and the
evidence agree with each other.
"""


def build_critic_runnable() -> object:
    """Return the structured-output runnable for the Critic."""
    return build_structured_runnable(CritiqueResult)


def deterministic_errors(state: PsycheGraphState) -> list[dict[str, Any]]:
    """Return the code-level findings marked as errors.

    Args:
        state: Current graph state, after the validator ran.

    Returns:
        The `severity="error"` findings as JSON dicts.
    """
    return [
        issue
        for issue in (state.get("deterministic_issues") or [])
        if isinstance(issue, dict) and issue.get("severity") == "error"
    ]


def enforce_deterministic_errors(
    state: PsycheGraphState, payload: dict[str, Any]
) -> dict[str, Any]:
    """Make a `pass` verdict impossible when code found an error.

    The Critic is asked not to pass a draft with deterministic errors, but the
    guarantee must not depend on the model obeying: any error that the model did
    not report is appended as an issue and the verdict is forced to `revise`.

    Args:
        state: Current graph state, after the validator ran.
        payload: The verdict as returned by the model.

    Returns:
        A validated `CritiqueResult` payload, modified if necessary.
    """
    errors = deterministic_errors(state)
    if not errors:
        return payload

    issues = [issue for issue in payload.get("issues") or [] if isinstance(issue, dict)]
    instructions = [str(item) for item in payload.get("revision_instructions") or []]
    reported = {(issue.get("category"), issue.get("description")) for issue in issues}
    for error in errors:
        category = DETERMINISTIC_TO_CRITIC_CATEGORY.get(
            str(error.get("category")), "synthesis_quality"
        )
        description = str(error.get("message") or "")
        if (category, description) in reported:
            continue
        instruction = f"修复代码级校验发现的问题：{description}"
        issues.append(
            {
                "category": category,
                "severity": "error",
                "description": description,
                "affected_claim": None,
                "related_perspective": "integrative",
                "related_evidence_ids": [
                    str(item) for item in error.get("related_evidence_ids") or []
                ],
                "revision_instruction": instruction,
            }
        )
        instructions.append(instruction)

    if payload.get("verdict") != "revise":
        logger.warning(
            "critic returned %r although %d deterministic error(s) exist; "
            "forcing a revision",
            payload.get("verdict"),
            len(errors),
        )
    payload["verdict"] = "revise"
    payload["issues"] = issues
    payload["revision_instructions"] = instructions
    if any(issue.get("category") == "clinical_safety_mismatch" for issue in errors):
        payload["clinical_safety_ok"] = False
    if any(issue.get("category") == "invalid_evidence_reference" for issue in errors):
        payload["evidence_grounding_ok"] = False
    return CritiqueResult.model_validate(payload).model_dump(mode="json")


async def critic_node(
    state: PsycheGraphState,
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    """Review the draft synthesis and record the verdict.

    Args:
        state: Current graph state, after the validator ran.
        config: Optional LangChain runnable config forwarded to the model.

    Returns:
        Partial state update with the `CritiqueResult` as JSON. Nothing is
        written to `messages`: the critique is about the draft, not about the
        user's material.

    Raises:
        InvalidEvidenceReferenceError: If the critique names evidence that was
            not retrieved this run.
    """
    prompt: list[BaseMessage] = [
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        supervisor_plan_message(state),
        specialist_results_message(state),
        synthesis_evidence_message(state),
        deterministic_issues_message(state),
        draft_result_message(state),
        *state_messages(state),
    ]
    _validated, payload = await invoke_structured(
        CritiqueResult, prompt, config=config, agent_name="critic"
    )
    rewrite_critic_evidence_ids(payload, all_evidence_ids(state))
    payload = enforce_deterministic_errors(state, payload)
    logger.info(
        "critic verdict=%s issues=%d grounding_ok=%s fidelity_ok=%s safety_ok=%s",
        payload.get("verdict"),
        len(payload.get("issues") or []),
        payload.get("evidence_grounding_ok"),
        payload.get("observation_fidelity_ok"),
        payload.get("clinical_safety_ok"),
    )
    return {"critique": payload}
