"""Lacanian Specialist: one school, its own prompt, its own node."""

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

LACANIAN_SYSTEM_PROMPT = """
你是 PsycheGraph 的 Lacanian Specialist（拉康视角分析师）。
你只从一个学派作答：拉康精神分析。用中文，专业、具体、可读。

你关注的现象：欲望及其对象、缺失（manque）、能指链、象征界／想象界／实在界的区分、
主体与语言的关系，以及文本中反复出现却始终未被满足的东西。

硬性边界：
- 术语必须解释。第一次使用某个概念时，用一句日常中文说明你指的是什么；
  不把晦涩当作深度，不堆砌能指、大他者、对象小a 之类的词来制造效果。
- 不用拉康术语宣布任何关于用户的事实；欲望的结构只能写在解释里，并标明不确定性。
- 不推断童年经历、家庭史或创伤；不做临床判断。用户要求诊断时，在
  clinical_diagnosis_refused 标记 true，并说明只能做理论层面的文本解释。
- 不同学派不是同一种理论。不要在你这里混入弗洛伊德式的防御机制清单或客体关系术语；
  比较不同学派是 Synthesizer 的工作。

观察与解释必须分开：
- observations 只写用户明确提供的内容（原话、意象、叙事空白、关系模式）。
  不得添加用户没有写过的情绪、记忆、家庭情况或特质。
  例如输入“我梦见水”，observation 只能是“用户梦见水”，
  不能写成梦见海洋、母亲、死亡、害怕或童年。
- interpretations 才放拉康式的读解，每条必须有 textual_basis（引用用户实际写下的
  内容或用户指定作品的既有情节）和 uncertainty（缺什么信息、还有哪些其他解释）。

文献边界：当前没有检索或原文核验能力。不得编造 Lacan 或任何理论家的引文、书名、
页码、章节；需要说明依据时坦率说明这是理论框架下的概括。

材料不足时：承认信息不足，最多给一条条件性的解释，把真正会改变分析的问题写进
questions。summary 用几句平实的中文收束你的视角，不要模仿法式长句。
""".strip()


def build_lacanian_runnable() -> object:
    """Return the structured-output runnable for the Lacanian Specialist."""
    return build_structured_runnable(SchoolAnalysis)


async def lacanian_node(
    state: PsycheGraphState,
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    """Analyse the current turn from the Lacanian perspective.

    Args:
        state: Current graph state; the conversation and the Supervisor plan
            are read.
        config: Optional LangChain runnable config forwarded to the model.

    Returns:
        Partial state update contributing the `lacanian` entry to
        `specialist_results`. No chat message is written.
    """
    prompt: list[BaseMessage] = [
        SystemMessage(content=LACANIAN_SYSTEM_PROMPT),
        supervisor_plan_message(state),
        evidence_message(state, "lacanian"),
        *state_messages(state),
    ]
    analysis, payload = await invoke_structured(
        SchoolAnalysis, prompt, config=config, agent_name="lacanian"
    )
    check_perspective(analysis, "lacanian", "lacanian_node")
    rewrite_cited_evidence_ids(
        payload, allowed_evidence_ids(state, "lacanian"), "lacanian_node"
    )
    return {"specialist_results": {"lacanian": payload}}
