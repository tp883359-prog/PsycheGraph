"""The four evaluation variants, built from the production agents.

    A  single_agent      one analyst, the Phase 3/4 baseline contract
    B  multi_agent       Supervisor -> 3 Specialists -> Synthesizer
    C  multi_agent_rag   B plus the retrieval `evidence` node
    D  full_system       the production graph (validator + Critic + finalizer)

No business logic is duplicated: B and C wire the *production* node functions
into a smaller graph, and D is the production graph object itself. Building a
variant never mutates production code, and importing this module has no side
effects on `react_agent.graph`.
"""

from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from react_agent.agents import (
    evidence_node,
    freudian_node,
    lacanian_node,
    object_relations_node,
    supervisor_node,
    synthesizer_node,
)
from react_agent.agents.finalizer import render_visible_answer, used_evidence_items
from react_agent.llm import invoke_structured
from react_agent.prompts import PSYCHOANALYTIC_ANALYST_SYSTEM_PROMPT
from react_agent.schemas import TheoryInterpretation
from react_agent.state import PsycheGraphState

VariantName = Literal[
    "single_agent",
    "multi_agent",
    "multi_agent_rag",
    "full_system",
]
"""Names accepted by the evaluation runner."""

VARIANTS: tuple[VariantName, ...] = (
    "single_agent",
    "multi_agent",
    "multi_agent_rag",
    "full_system",
)
"""Canonical variant order for reports."""


class PsychoanalyticAnalysis(BaseModel):
    """The single-analyst output contract (Phase 3/4 baseline).

    Reconstructed for the evaluation so variant A produces the same kind of
    fields as the multi-agent pipeline (observations, interpretations,
    limitations, questions, refusal flag, final answer). It is used by the
    baseline variant only and is not part of the production graph.
    """

    observations: list[str] = Field(
        description="Only what the user actually wrote; no added facts."
    )
    interpretations: list[TheoryInterpretation] = Field(
        default_factory=list,
        description="Theory-based readings with their own uncertainty notes.",
    )
    limitations: list[str] = Field(
        default_factory=list, description="Limits of this reading."
    )
    follow_up_questions: list[str] = Field(
        default_factory=list, description="At most three questions that would matter."
    )
    clinical_diagnosis_refused: bool = Field(
        default=False,
        description="True when the user asked for a clinical judgement and it was refused.",
    )
    final_response: str = Field(
        description="The natural-language answer, consistent with the fields above."
    )


async def single_analyst_node(
    state: PsycheGraphState,
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    """Answer with one analyst model call (variant A).

    Args:
        state: Current graph state; only `messages` is read.
        config: Optional runnable config forwarded to the model.

    Returns:
        Partial state update with the analysed payload as `final_result` and one
        AI message.
    """
    prompt: list[BaseMessage] = [
        SystemMessage(content=PSYCHOANALYTIC_ANALYST_SYSTEM_PROMPT),
        *list(state["messages"]),
    ]
    _validated, payload = await invoke_structured(
        PsychoanalyticAnalysis, prompt, config=config, agent_name="single_analyst"
    )
    answer = str(payload.get("final_response", ""))
    return {
        "final_result": payload,
        "finalization_status": "baseline",
        "messages": [AIMessage(content=answer)],
    }


async def publish_draft_node(
    state: PsycheGraphState,
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    """Publish the synthesizer's draft without any review (variants B and C).

    B and C have no Critic, so the draft *is* the answer. The source block is
    rendered by the production helper, so citation metadata is comparable with
    variant D. No model call happens here.

    Args:
        state: State after the synthesizer wrote `draft_result`.
        config: Unused; kept for node-signature parity.

    Returns:
        Partial state update with `final_result` and one AI message.
    """
    draft = dict(state.get("draft_result") or {})
    items = used_evidence_items(state, draft)
    answer = render_visible_answer(draft, items)
    status = "unreviewed" if items else "unreviewed_no_sources"
    return {
        "final_result": draft,
        "finalization_status": status,
        "messages": [AIMessage(content=answer)],
    }


def build_single_agent_graph() -> Any:
    """Build variant A: START -> analyst -> END."""
    builder = StateGraph(PsycheGraphState)
    builder.add_node("analyst", single_analyst_node)
    builder.add_edge(START, "analyst")
    builder.add_edge("analyst", END)
    return builder.compile()


def build_multi_agent_graph(*, with_rag: bool) -> Any:
    """Build variant B (`with_rag=False`) or C (`with_rag=True`).

    Args:
        with_rag: Whether the retrieval `evidence` node runs between the
            Supervisor and the specialists.

    Returns:
        The compiled graph.
    """
    builder = StateGraph(PsycheGraphState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("freudian", freudian_node)
    builder.add_node("object_relations", object_relations_node)
    builder.add_node("lacanian", lacanian_node)
    builder.add_node("synthesizer", synthesizer_node)
    builder.add_node("publish", publish_draft_node)

    builder.add_edge(START, "supervisor")
    if with_rag:
        builder.add_node("evidence", evidence_node)
        builder.add_edge("supervisor", "evidence")
        for school in ("freudian", "object_relations", "lacanian"):
            builder.add_edge("evidence", school)
    else:
        for school in ("freudian", "object_relations", "lacanian"):
            builder.add_edge("supervisor", school)
    for school in ("freudian", "object_relations", "lacanian"):
        builder.add_edge(school, "synthesizer")
    builder.add_edge("synthesizer", "publish")
    builder.add_edge("publish", END)
    return builder.compile()


@dataclass(frozen=True)
class Variant:
    """One evaluation arm.

    Attributes:
        name: Variant identifier used in results and reports.
        description: One-line description for reports.
        uses_rag: Whether retrieval runs.
        uses_critic: Whether the deterministic validator, Critic and finalizer run.
        build: Factory returning a fresh compiled graph (or the production graph).
    """

    name: VariantName
    description: str
    uses_rag: bool
    uses_critic: bool
    build: Any


def _full_system_graph() -> Any:
    """Return the production graph object itself, unchanged.

    Imported lazily so that this module can be imported without building the
    production graph, and so the object identity can be asserted in tests.
    """
    from react_agent.graph import graph

    return graph


def get_variant(name: str) -> Variant:
    """Return the variant definition for `name`.

    Args:
        name: One of `VARIANTS`.

    Returns:
        The variant.

    Raises:
        KeyError: If the name is unknown.
    """
    registry: dict[str, Variant] = {
        "single_agent": Variant(
            name="single_agent",
            description="Single psychoanalytic analyst (Phase 3/4 baseline)",
            uses_rag=False,
            uses_critic=False,
            build=build_single_agent_graph,
        ),
        "multi_agent": Variant(
            name="multi_agent",
            description="Supervisor + three specialists + synthesizer, no retrieval, no Critic",
            uses_rag=False,
            uses_critic=False,
            build=lambda: build_multi_agent_graph(with_rag=False),
        ),
        "multi_agent_rag": Variant(
            name="multi_agent_rag",
            description="Supervisor + evidence + three specialists + synthesizer, no Critic",
            uses_rag=True,
            uses_critic=False,
            build=lambda: build_multi_agent_graph(with_rag=True),
        ),
        "full_system": Variant(
            name="full_system",
            description="Production graph: evidence + specialists + validator + Critic",
            uses_rag=True,
            uses_critic=True,
            build=_full_system_graph,
        ),
    }
    if name not in registry:
        raise KeyError(
            f"unknown variant {name!r}; expected one of {', '.join(VARIANTS)}"
        )
    return registry[name]


def initial_state(case_input: str) -> PsycheGraphState:
    """Return the state a variant is invoked with.

    Every variant receives the same single user message, which is what keeps the
    comparison fair.

    Args:
        case_input: The case's user text.

    Returns:
        A state with exactly one human message.
    """
    return {"messages": [HumanMessage(content=case_input)]}
