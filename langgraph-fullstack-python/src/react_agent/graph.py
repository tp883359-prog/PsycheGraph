"""PsycheGraph: explicit multi-agent StateGraph with retrieval and review.

Phase 5 built the multi-agent pipeline, Phase 6 inserted local retrieval before
the specialists run, and Phase 7 adds a review stage after the synthesis:

    START
      -> supervisor                       (1 model call)
      -> evidence                         (0 model calls, local index only)
      -> freudian | object_relations | lacanian   (3 model calls, concurrent)
      -> synthesizer                      (1 model call, writes draft_result)
      -> deterministic_validator          (0 model calls, code only)
      -> critic                           (1 model call)
      -> route:
           pass                  -> finalize       -> END
           revise, budget left   -> revise_synthesis -> deterministic_validator
           revise, budget spent  -> safe_finalize  -> END

The three specialists run in the same superstep, so they execute concurrently;
LangGraph itself guarantees that the synthesizer only runs once all three have
written their result. No locks, sleeps or manual `asyncio.gather` are involved.

A normal turn costs six DeepSeek calls; one revision costs eight. The evidence
node, the validator and both finalizers never call a model, and the revision
step reuses the evidence of the current run instead of re-retrieving.

The loop is bounded by `MAX_REVISION`, so the graph terminates for every input:
after the budget is spent, a `revise` verdict routes to `safe_finalize` instead
of back into the synthesizer.
"""

from langgraph.graph import END, START, StateGraph

from react_agent.agents import (
    critic_node,
    deterministic_validator_node,
    evidence_node,
    finalize_node,
    freudian_node,
    lacanian_node,
    object_relations_node,
    revise_synthesis_node,
    safe_finalize_node,
    supervisor_node,
    synthesizer_node,
)
from react_agent.config import MAX_REVISION
from react_agent.state import PsycheGraphState

SPECIALIST_NODES: tuple[str, ...] = (
    "freudian",
    "object_relations",
    "lacanian",
)
"""Specialist nodes fanned out from the evidence step and fanned in to the synthesizer."""


def route_after_critic(state: PsycheGraphState) -> str:
    """Decide where the run goes after the Critic.

    Args:
        state: Current graph state, after the Critic wrote `critique`.

    Returns:
        `"finalize"` when the draft passed, `"revise_synthesis"` when the Critic
        asked for a revision and the budget is not spent yet, and
        `"safe_finalize"` once `MAX_REVISION` revisions have been used. The
        third branch is what makes the loop finite for every input.
    """
    critique = state.get("critique") or {}
    verdict = str(critique.get("verdict", "revise"))
    revision_count = int(state.get("revision_count", 0) or 0)
    if verdict == "pass":
        return "finalize"
    if revision_count < MAX_REVISION:
        return "revise_synthesis"
    return "safe_finalize"


builder = StateGraph(PsycheGraphState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("evidence", evidence_node)
builder.add_node("freudian", freudian_node)
builder.add_node("object_relations", object_relations_node)
builder.add_node("lacanian", lacanian_node)
builder.add_node("synthesizer", synthesizer_node)
builder.add_node("deterministic_validator", deterministic_validator_node)
builder.add_node("critic", critic_node)
builder.add_node("revise_synthesis", revise_synthesis_node)
builder.add_node("finalize", finalize_node)
builder.add_node("safe_finalize", safe_finalize_node)

builder.add_edge(START, "supervisor")
builder.add_edge("supervisor", "evidence")
for node_name in SPECIALIST_NODES:
    builder.add_edge("evidence", node_name)
    builder.add_edge(node_name, "synthesizer")
builder.add_edge("synthesizer", "deterministic_validator")
builder.add_edge("deterministic_validator", "critic")
builder.add_conditional_edges(
    "critic",
    route_after_critic,
    {
        "finalize": "finalize",
        "revise_synthesis": "revise_synthesis",
        "safe_finalize": "safe_finalize",
    },
)
builder.add_edge("revise_synthesis", "deterministic_validator")
builder.add_edge("finalize", END)
builder.add_edge("safe_finalize", END)

graph = builder.compile()
