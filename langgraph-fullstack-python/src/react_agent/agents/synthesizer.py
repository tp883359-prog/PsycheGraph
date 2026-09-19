"""Synthesizer agent: combines the three school results into a draft answer.

The Synthesizer does not analyse the material again. It compares the three
specialist results, keeps their disagreements visible, and writes the answer.

Since Phase 7 that answer is a *draft*: it is stored in `draft_result` and the
Critic reviews it before anything reaches the user. The Synthesizer therefore
writes no chat message at all; the finalizer is the only node that does.
"""

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from react_agent.agents.context import (
    all_evidence_ids,
    rewrite_synthesizer_evidence_ids,
    specialist_results_message,
    state_messages,
    supervisor_plan_message,
    synthesis_evidence_message,
)
from react_agent.llm import build_structured_runnable, invoke_structured
from react_agent.schemas import SynthesisResult
from react_agent.state import PsycheGraphState

SYNTHESIZER_SYSTEM_PROMPT = """
你是 PsycheGraph 的 Synthesizer（综合 Agent）。你读 Supervisor 的计划和三个 Specialist
的结果，写出给用户看的回答草稿（draft）。这份草稿之后会被 Critic 审核，只有通过审核
（或经过一次修订）后才会展示给用户。

输入：
- Supervisor 的计划（含三个分析重点与 synthesis_goal）。
- 三个 Specialist 的结构化结果：弗洛伊德视角、客体关系视角、拉康视角。
- 完整对话消息（用户最近说了什么，之前说过什么）。

工作方式：
- 你不重新从零分析。你的材料是这三个结果和用户文本，不是新的联想。
- 保留差异：三个学派对同一材料往往给出不同方向的读解，这是正常的。
  不要强行选一个“正确解释”，也不要把三种读解揉成一段谁也不冒犯的中间话。
- 指出一致之处：如果两个或三个学派在相同的文本线索上得出相近方向，说清楚是哪条线索、
  哪个方向。
- 指出不同之处：说清楚每个学派强调什么、各自解释不了什么，并说明为什么会分歧
  （关注点不同，而不是谁更正确）。
- 不得引入三个 Specialist 都没有提出的新理论结论；可以组织、比较、澄清措辞。
- 不得编造用户没有提供的经历、家庭史、创伤或诊断。理论上不确定的地方保留不确定。
- 没有文献检索能力：不得编造引文、书名、页码。需要说明依据时明确这是理论框架下的概括。
- 用户要求临床诊断时：clinical_diagnosis_refused 设为 true，并在 final_response 中
  明确说明本系统只能做精神分析理论层面的文本讨论，不能进行临床诊断，也不会判断
  用户是否患有某种障碍；可以介绍相关理论概念，但不确认也不排除任何疾病。

字段要求：
- common_ground：至少两个学派共同支持的读解方向（写清依据的文本线索）。
- differences：学派之间的实质差异，点名是哪几个学派在什么问题上不同。
- integrated_interpretation：比较之后的整体读解，包括仍然悬而未决的部分。
- limitations：这次综合的限制（信息不足、文本分析不等于真实生活、没有可核验文献）。
- follow_up_questions：真正会改变分析的问题，用户能回答的那种；不问症状、不做筛查。
- final_response：最终回答。自然、专业、可读的中文，长度与材料相称：
  材料很少（例如只有“我梦见水”）就短，先承认信息不足并问几个关键问题；
  材料丰富或用户要求比较几个学派时可以有结构地展开。
  不要输出 JSON、字段名、schema 名称或函数调用语法，也不要把三个 summary 逐段拼接。
""".strip()


def build_synthesizer_runnable() -> object:
    """Return the structured-output runnable for the Synthesizer."""
    return build_structured_runnable(SynthesisResult)


async def synthesizer_node(
    state: PsycheGraphState,
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    """Combine the specialist results into a reviewable draft.

    The node writes `draft_result` only: no chat message, because an unreviewed
    draft must not enter the conversation history the web UI and the next turn
    both read.

    Args:
        state: Current graph state after all specialists finished.
        config: Optional LangChain runnable config forwarded to the model.

    Returns:
        Partial state update with the draft synthesis as JSON.

    Raises:
        InvalidEvidenceReferenceError: If `used_evidence_ids` names evidence that
            was never retrieved.
    """
    prompt: list[BaseMessage] = [
        SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
        supervisor_plan_message(state),
        specialist_results_message(state),
        synthesis_evidence_message(state),
        *state_messages(state),
    ]
    _validated, result = await invoke_structured(
        SynthesisResult, prompt, config=config, agent_name="synthesizer"
    )
    allowed = all_evidence_ids(state)
    # Same contract as the specialists: ids are resolved against what was
    # retrieved, and an unresolvable citation aborts the run.
    rewrite_synthesizer_evidence_ids(result, allowed)
    return {"draft_result": result}
