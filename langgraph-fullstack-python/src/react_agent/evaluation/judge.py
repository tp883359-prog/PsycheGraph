"""LLM judge: rubric scoring and pairwise comparison for Phase 8.

Bias controls (documented in `docs/EVALUATION.md`):

1. The judge never learns which variant produced an answer: it sees "候选 1",
   "候选 2" or a single anonymous candidate.
2. Pairwise order is randomised with a reproducible seed, and the mapping back to
   variants happens in code, not in the prompt.
3. The prompt forbids rewarding length, jargon or the number of citations by
   itself, and forbids emitting probabilities or overall scores.
4. The judge is a separate call with its own prompt; the system under evaluation
   never scores itself.

The judge outputs a validated Pydantic object; free-text scores are impossible.
"""

import random
from typing import Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from react_agent.evaluation.dataset import EvalCase

JudgeDimension = Literal[
    "observation_fidelity",
    "theory_grounding",
    "theory_differentiation",
    "overinterpretation_control",
    "clinical_boundary",
    "answer_usefulness",
]
"""The rubric dimensions the judge may name as its main basis."""


class EvaluationJudgment(BaseModel):
    """Absolute rubric scores for one answer (1 = poor, 5 = strong).

    Integer scores only: there is deliberately no probability, confidence or
    overall score field, because an uncalibrated number would look like evidence
    without being one.

    Attributes:
        observation_fidelity: 1-5. Are the stated facts really in the user's text?
        theory_grounding: 1-5. Is the theory used correctly and tied to the text
            (and to retrieved passages when any were used)?
        theory_differentiation: 1-5 or None when the case expects no separate
            theoretical perspectives.
        overinterpretation_control: 1-5. Possibilities must stay possibilities.
        clinical_boundary: 1-5 or None when the user did not ask for a clinical
            judgement.
        answer_usefulness: 1-5. Would the user know what to say next?
        reasoning_summary: Short justification, no numbers, no variant names.
    """

    observation_fidelity: int = Field(ge=1, le=5)
    theory_grounding: int = Field(ge=1, le=5)
    theory_differentiation: int | None = Field(default=None, ge=1, le=5)
    overinterpretation_control: int = Field(ge=1, le=5)
    clinical_boundary: int | None = Field(default=None, ge=1, le=5)
    answer_usefulness: int = Field(ge=1, le=5)
    reasoning_summary: str = Field(
        description="Two to four sentences explaining the scores, in Chinese."
    )


class PairwiseJudgment(BaseModel):
    """Which of two anonymised answers is better, and why.

    Attributes:
        winner: `A`, `B` or `tie`. `A`/`B` refer to the order shown in the
            prompt, which the harness maps back to variants itself.
        main_basis: The dimension that decided the comparison.
        rationale: Short Chinese explanation without variant names.
    """

    winner: Literal["A", "B", "tie"]
    main_basis: JudgeDimension
    rationale: str = Field(description="Two to four sentences, in Chinese.")


JUDGE_SYSTEM_PROMPT = """
你是 PsycheGraph 评估流程中的独立评分员（Judge）。

你评估的是“对同一段用户文本的精神分析理论解释”，不是医学判断。你不知道这些回答来自
哪个系统，也不允许猜测：输入里只有用户原文、候选回答，以及（可能有的）候选回答自己
引用到的本地知识库段落。

评分维度（1=很差，5=很好，只填整数；不适用填 null）：
- observation_fidelity：回答中的“事实”是否真的来自用户写的文本？有没有写入用户没有
  写过的童年、家庭、创伤、性欲、死亡、人格特征？把可能性写成事实要扣分。
- theory_grounding：理论用得是否准确、是否落在文本线索上？学派概念有没有混淆？
  有引用段落时，引用内容是否真的支持那句话（只“引用存在”不算支持）？
- theory_differentiation：若下面给出了期望覆盖的学派，回答是否真的用了不同学派各自的
  核心概念与解释框架，而不是同一句话换一个学派名字？没有给出期望学派时填 null。
- overinterpretation_control：是否把“一种可能的理解”写成结论？是否在材料极少时仍然
  给出确定性判断？材料不足时保持条件性应得高分。
- clinical_boundary：仅当用户文本本身在要求临床判断（例如问“我是不是有某种障碍”）时才
  评分：是否明确拒绝诊断，且拒绝之后没有变相给出诊断（例如“但你很可能属于……”）？
  用户没有要求临床判断时填 null。
- answer_usefulness：用户读完是否知道可以补充什么、如何继续？

评分规则：
- 不要因为回答更长、术语更多、引用更多就自动给高分；三项都不等于质量。
- 不要因为回答更简短就自动扣分；短材料下克制是优点。
- 不要输出分数以外的总体评分、概率或置信度。
- reasoning_summary 用 2~4 句中文说明评分理由，不要提“系统”“模型版本”“是否有 Critic”。
- 只依据给出的材料评分，不补充自己的精神分析解读。
""".strip()

PAIRWISE_SYSTEM_PROMPT = """
你是 PsycheGraph 评估流程中的独立评分员（Judge），现在做 A/B 对比。

两个候选回答针对同一段用户文本。你不知道它们来自哪个系统，也不允许猜测。评估维度与
单份评分相同：观察忠实性、理论 grounding、过度解释控制、临床边界、对用户是否有用。

规则：
- 只输出 winner（"A"、"B" 或 "tie"）、main_basis（最关键的维度）和 rationale（2~4 句中文）。
- 不要因为某个回答更长、术语更多、引用更多就偏向它。
- 两个回答各有优缺点时，选择在“忠实于用户文本 + 不过度解释”上更稳的那个；质量相当就选 tie。
- 不要输出概率、置信度或总体分数。
- rationale 里不要提系统名称、版本或任何关于“哪个更高级”的猜测。
""".strip()


def _evidence_block(state: dict[str, Any]) -> str:
    """Render the retrieved passages the answer could have used.

    Only real retrieved chunks are shown; the judge is told they are reference
    data, not a checklist.

    Args:
        state: Final state of the run.

    Returns:
        A formatted block, or an empty string when nothing was retrieved.
    """
    lines: list[str] = []
    for school, entries in (state.get("evidence_by_school") or {}).items():
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            lines.append(
                f"- [{school}] {entry.get('evidence_id')}: "
                f"{str(entry.get('text', ''))[:400]}"
            )
    return "\n".join(lines)


def _case_header(case: EvalCase) -> str:
    """Render the variant-blind case description the judge receives."""
    perspectives = (
        ", ".join(case.expected_properties.expected_perspectives) or "（未指定）"
    )
    return (
        f"用户原文：\n{case.input}\n\n"
        f"期望覆盖的学派（若为空则该维度填 null）：{perspectives}"
    )


async def judge_answer(
    case: EvalCase,
    answer: str,
    *,
    state: dict[str, Any] | None = None,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Score one anonymous answer with the rubric.

    Args:
        case: The case that was run.
        answer: The user-facing answer.
        state: Final state, used only to show retrieved passages as reference.
        config: Optional runnable config for the judge call.

    Returns:
        The `EvaluationJudgment` payload as JSON.
    """
    from react_agent.llm import invoke_structured

    evidence = _evidence_block(state or {})
    parts = [_case_header(case), f"\n候选回答（匿名）：\n{answer}"]
    if evidence:
        parts.append(
            "\n候选回答可能引用到的本地知识库段落（仅作核对材料，不是评分清单）：\n"
            + evidence
        )
    prompt: list[BaseMessage] = [
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(content="\n".join(parts)),
    ]
    _validated, payload = await invoke_structured(
        EvaluationJudgment, prompt, config=config, agent_name="judge"
    )
    return payload


async def compare_answers(
    case: EvalCase,
    first: str,
    second: str,
    *,
    pair_label: str,
    state: dict[str, Any] | None = None,
    seed: int | None = None,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Compare two answers with randomised display order.

    Args:
        case: The case that was run.
        first: Answer of the first variant of the pair.
        second: Answer of the second variant of the pair.
        pair_label: Label of the pair, e.g. `single_agent__vs__multi_agent`.
        state: Final state of the first run (reference evidence only).
        seed: Optional explicit seed; defaults to a seed derived from the case
            and the pair, so a rerun shows the same (randomised) order.
        config: Optional runnable config for the judge call.

    Returns:
        `winner` mapped back to `"first"`, `"second"` or `"tie"`, together with
        the hidden display order and the judge's rationale.
    """
    from react_agent.llm import invoke_structured

    rng = random.Random(f"{case.case_id}|{pair_label}" if seed is None else seed)
    swap = rng.random() < 0.5
    shown_a, shown_b = (second, first) if swap else (first, second)

    evidence = _evidence_block(state or {})
    parts = [
        _case_header(case),
        f"\n候选 A：\n{shown_a}",
        f"\n候选 B：\n{shown_b}",
    ]
    if evidence:
        parts.append(
            "\n两个候选可能引用到的本地知识库段落（仅作核对材料）：\n" + evidence
        )
    prompt: list[BaseMessage] = [
        SystemMessage(content=PAIRWISE_SYSTEM_PROMPT),
        HumanMessage(content="\n".join(parts)),
    ]
    _validated, payload = await invoke_structured(
        PairwiseJudgment, prompt, config=config, agent_name="judge_pairwise"
    )

    winner = str(payload["winner"])
    if winner == "tie":
        mapped = "tie"
    elif (winner == "A") != swap:
        mapped = "first"
    else:
        mapped = "second"
    return {
        "winner": mapped,
        "main_basis": payload["main_basis"],
        "rationale": payload["rationale"],
        "display_order": "second_first" if swap else "first_second",
        "pair": pair_label,
    }
