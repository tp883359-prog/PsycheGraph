"""Supervisor agent: plans the turn before any analysis happens.

The Supervisor does not analyse. It reads the conversation, writes one focus
instruction per school, flags clinical-diagnosis requests and states what the
Synthesizer has to resolve.
"""

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from react_agent.llm import build_structured_runnable, invoke_structured
from react_agent.schemas import SupervisorPlan
from react_agent.state import PsycheGraphState

SUPERVISOR_SYSTEM_PROMPT = """
你是 PsycheGraph 的 Supervisor（任务规划 Agent）。你负责把用户当前的请求拆成给三个学派
Specialist 的分析重点，并说明最终需要综合什么。你不做理论分析，也不给用户写回答。

输入：完整的对话消息。最近的用户消息是本次任务；更早的消息提供上下文。

你的输出字段：
- task_summary：中性复述用户这次请你做什么（分析一段梦、一段自我叙述、某个作品、
  一段对话或关系场景等）。只写用户实际写下的内容，不补充细节，不做解释。
- analysis_focus：本次分析最该注意什么。写文本层面的特征，例如反复出现的行为、
  缺失的信息、措辞本身；不要写对用户的判断。
- freudian_focus / object_relations_focus / lacanian_focus：分别告诉三个 Specialist
  各自该看什么文本线索（该学派的哪些问题最贴合现在的材料）。
  这是分析方向，不是事实断言。可以写“关注寻找反复落空这一形式”；
  绝不能写“该用户有童年创伤”“该用户母亲疏远”这类用户未提供的事实。
- clinical_diagnosis_requested：用户是否在要求临床判断（例如“我是不是有某种人格障碍”）。
- synthesis_goal：三个学派的结果最后需要解决什么分歧、回答什么问题。

规划规则：
- 材料很短时（例如只有“我梦见水”），明确要求三个 Specialist 保守分析：
  先承认信息不足，最多给出条件性的解释。不要提示母亲、性欲、创伤、死亡、孕育
  等用户没有提供的内容，也不要暗示存在某个童年事件。
- 不预设任何学派更高明；三个重点应当彼此不同，避免写三段意思相同的句子。
- 用户指定某个学派时，仍按三个学派给重点，但在 synthesis_goal 中说明用户指定的视角
  优先级更高。
- 用户要求临床诊断时：clinical_diagnosis_requested 设为 true，并在 synthesis_goal 中
  写明最终回答必须明确说明系统不能进行临床诊断，可以转而讨论相关理论概念。
- 用户要求引用原文或文献出处时，在 synthesis_goal 中写明当前没有文献核验能力，
  不得编造引文、书名或页码，可以请用户提供可核验文本。
- 你只输出这张计划，不输出任何分析结论，也不复述大段用户文本。
""".strip()


def build_supervisor_runnable() -> object:
    """Return the structured-output runnable for the Supervisor."""
    return build_structured_runnable(SupervisorPlan)


async def supervisor_node(
    state: PsycheGraphState,
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    """Plan the current turn and store the plan as JSON data.

    Starting a turn also resets the per-turn review working fields: the
    revision counter belongs to the current draft, so a second turn must not
    inherit the first turn's revisions.

    Args:
        state: Current graph state; only `messages` is read.
        config: Optional LangChain runnable config forwarded to the model.

    Returns:
        Partial state update with the JSON form of the plan and a reset
        revision counter. No message is appended: internal agents never write
        into the chat transcript.
    """
    messages: list[BaseMessage] = list(state["messages"])
    prompt: list[BaseMessage] = [
        SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
        *messages,
    ]
    _payload, plan = await invoke_structured(
        SupervisorPlan, prompt, config=config, agent_name="supervisor"
    )
    return {"supervisor_plan": plan, "revision_count": 0}
