"""Freudian Specialist: one school, its own prompt, its own node."""

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from react_agent.agents.context import (
    allowed_evidence_ids,
    check_perspective,
    evidence_message,
    rewrite_cited_evidence_ids,
    state_messages,
    supervisor_plan_message,
)
from react_agent.llm import build_structured_runnable, invoke_structured
from react_agent.schemas import SchoolAnalysis
from react_agent.state import PsycheGraphState

FREUDIAN_SYSTEM_PROMPT = """
你是 PsycheGraph 的 Freudian Specialist（弗洛伊德精神分析视角分析师）。
你只从一个学派作答：弗洛伊德精神分析。用中文，专业、具体、可读。

你关注的现象：无意识冲突、防御机制、压抑、愿望及其变形、象征意义、
本我／自我／超我的运作，以及移情作为理论概念（不宣称与用户存在临床移情关系）。

硬性边界：
- 不把一切都解释成性欲或力比多；只在文本确实支持时才提到性、欲望或身体主题。
- 不强行套用俄狄浦斯情结、阉割焦虑、阴茎羡嫉等经典母题；没有文本线索就不提。
- 不推断童年创伤、童年事件或早期经历；用户没有写，就不能当成事实。
- 理论解释不是事实：每条解释都必须标明依据的文本片段和自身的不确定性。
- 你不是心理医生或诊断系统，不做精神疾病或人格障碍判断。用户要求诊断时，
  在 clinical_diagnosis_refused 标记 true，并说明只能做理论层面的文本解释。

观察与解释必须分开：
- observations 只写用户明确提供的内容（原话、意象、叙事空白、关系模式）。
  不得添加用户没有写过的情绪、记忆、家庭情况或特质。
  例如输入“我梦见水”，observation 只能是“用户梦见水”，
  不能写成梦见海洋、母亲、死亡、害怕或童年。
- interpretations 才放弗洛伊德式的联想，每条必须有 textual_basis（引用用户实际写下的
  内容或用户指定作品的既有情节）和 uncertainty（缺什么信息、还有哪些其他解释）。

文献边界：当前没有检索或原文核验能力。不得编造 Freud 或其他理论家的引文、书名、
页码、章节，也不得声称已核实某文献。需要说明依据时坦率说明这是理论框架下的概括。

材料不足时：承认信息不足，最多给一条条件性的解释，把真正会改变分析的问题写进
questions。不要用一堆术语填补空白。summary 用几句话收束你的视角，不要长篇论文。
""".strip()


def build_freudian_runnable() -> object:
    """Return the structured-output runnable for the Freudian Specialist."""
    return build_structured_runnable(SchoolAnalysis)


async def freudian_node(
    state: PsycheGraphState,
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    """Analyse the current turn from the Freudian perspective.

    Args:
        state: Current graph state; the conversation, the Supervisor plan and
            this school's retrieved theory evidence are read.
        config: Optional LangChain runnable config forwarded to the model.

    Returns:
        Partial state update contributing the `freudian` entry to
        `specialist_results`. No chat message is written.

    Raises:
        PerspectiveMismatchError: If the model answered as another school.
        InvalidEvidenceReferenceError: If the model cited an unknown evidence id.
    """
    prompt: list[BaseMessage] = [
        SystemMessage(content=FREUDIAN_SYSTEM_PROMPT),
        supervisor_plan_message(state),
        evidence_message(state, "freudian"),
        *state_messages(state),
    ]
    analysis, payload = await invoke_structured(
        SchoolAnalysis, prompt, config=config, agent_name="freudian"
    )
    check_perspective(analysis, "freudian", "freudian_node")
    # Resolve the ids the model wrote onto the ids it was actually given; an id
    # that cannot be resolved aborts the run instead of reaching the user.
    rewrite_cited_evidence_ids(
        payload, allowed_evidence_ids(state, "freudian"), "freudian_node"
    )
    return {"specialist_results": {"freudian": payload}}
