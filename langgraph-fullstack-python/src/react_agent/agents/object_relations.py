"""Object Relations Specialist: one school, its own prompt, its own node."""

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

OBJECT_RELATIONS_SYSTEM_PROMPT = """
你是 PsycheGraph 的 Object Relations Specialist（客体关系视角分析师）。
你只从一个学派作答：客体关系理论（Klein、Fairbairn、Winnicott 等传统的共同关切）。
用中文，专业、具体、可读。

你关注的现象：内在客体与客体表征、分裂（全好／全坏）、理想化与贬低、投射与投射性认同、
被内化的关系模式，以及文本里反复出现的关系形态。

硬性边界：
- 不根据几句话判断依恋类型、人格障碍或任何临床诊断；不判断用户的真实关系能力。
- 不编造用户的家庭经历、父母形象或早年照料情况；家庭角色只在文本提到时才讨论，
  而且只能作为文本内容，不是对真实亲属的判断。
- 提到“内在客体”时给出简短解释（例如“内心形成的重要他人形象”），不要只丢术语。
- 理论解释不是事实：每条解释都必须标明依据的文本片段和自身的不确定性。
- 用户要求临床诊断时，在 clinical_diagnosis_refused 标记 true，并说明只能做
  理论层面的文本解释，不做诊断。

观察与解释必须分开：
- observations 只写用户明确提供的内容（原话、意象、叙事空白、关系模式）。
  不得添加用户没有写过的情绪、记忆、家庭情况或特质。
  例如输入“我梦见水”，observation 只能是“用户梦见水”，
  不能写成梦见海洋、母亲、死亡、害怕或童年。
- interpretations 才放客体关系式的联想，每条必须有 textual_basis（引用用户实际写下的
  内容或用户指定作品的既有情节）和 uncertainty（缺什么信息、还有哪些其他解释）。

文献边界：当前没有检索或原文核验能力。不得编造 Klein、Winnicott 等理论家的引文、
书名、页码；不得声称已核实某文献。

材料不足时：承认信息不足，最多给一条条件性的解释，把真正会改变分析的问题写进
questions。不要把“关系”“容纳”“抱持”这类词当成万能解释。summary 用几句话收束，
不要长篇论文。
""".strip()


def build_object_relations_runnable() -> object:
    """Return the structured-output runnable for the Object Relations Specialist."""
    return build_structured_runnable(SchoolAnalysis)


async def object_relations_node(
    state: PsycheGraphState,
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    """Analyse the current turn from the object-relations perspective.

    Args:
        state: Current graph state; the conversation and the Supervisor plan
            are read.
        config: Optional LangChain runnable config forwarded to the model.

    Returns:
        Partial state update contributing the `object_relations` entry to
        `specialist_results`. No chat message is written.
    """
    prompt: list[BaseMessage] = [
        SystemMessage(content=OBJECT_RELATIONS_SYSTEM_PROMPT),
        supervisor_plan_message(state),
        evidence_message(state, "object_relations"),
        *state_messages(state),
    ]
    analysis, payload = await invoke_structured(
        SchoolAnalysis, prompt, config=config, agent_name="object_relations"
    )
    check_perspective(analysis, "object_relations", "object_relations_node")
    rewrite_cited_evidence_ids(
        payload,
        allowed_evidence_ids(state, "object_relations"),
        "object_relations_node",
    )
    return {"specialist_results": {"object_relations": payload}}
