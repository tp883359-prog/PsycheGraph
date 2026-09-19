"""Graph state for PsycheGraph.

Phase 5 turns the single analyst into Supervisor -> three parallel Specialists ->
Synthesizer, so the state separates three kinds of data:

`messages` is the only chat contract the web UI sees: one human turn and, at
most, one final AI answer per run. `supervisor_plan`, `specialist_results` and
`final_result` are working products of the current run, stored as plain JSON
data (never Pydantic instances) so that checkpoints stay serialisable.

The parallel specialists write into `specialist_results` through a merged-dict
reducer, so each school owns its own key and no update can overwrite another.

Phase 7 adds the reviewed draft lifecycle between the two:

    synthesizer -> draft_result
    deterministic_validator -> deterministic_issues
    critic -> critique
    revise_synthesis -> draft_result (again) + revision_count + 1
    finalize / safe_finalize -> final_result + exactly one AIMessage

`draft_result` and `critique` are internal: they never reach `messages`. The
finalizer is the only node in the whole graph that appends an AI message, so a
chat history can never contain a draft the user has not seen.
"""

import operator
from typing import Annotated, Any, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from react_agent.schemas import SpecialistPerspective


class PsycheGraphState(TypedDict):
    """State shared across the PsycheGraph nodes.

    Attributes:
        messages: Chat history. The reducer merges incoming messages instead of
            overwriting them, so the LangGraph service can append new turns to
            a checkpointed thread. This is the only field the web UI observes,
            and only the finalizer writes into it.
        supervisor_plan: The Supervisor's plan for the current run, as a JSON
            dict. Regenerated on every run, so a stale plan cannot leak into
            the next turn.
        evidence_by_school: Retrieved theory passages for the current run, keyed
            by school. Produced by the evidence node from the local knowledge
            base; each entry is a JSON dict following the `EvidenceItem` schema.
            Never holds embeddings, model objects or store clients.
        evidence_meta: Small JSON summary of the retrieval pass (whether an index
            was available, how long retrieval took, how many chunks each school
            received). Used for the performance record and for telling the user
            when no local literature backed an answer.
        specialist_results: School results for the current run, keyed by school
            name. Written by three nodes in the same graph step, therefore the
            reducer merges dictionaries instead of overwriting them.
        draft_result: The Synthesizer's structured result as a JSON dict. It is a
            draft: the Critic may still reject it, so it is never shown to the
            user and never written into `messages`.
        deterministic_issues: Problems found by code before the Critic runs, as
            JSON dicts following `DeterministicIssue` (invalid evidence ids,
            perspective mismatches, clinical-flag contradictions, dangling
            citation labels, leaked messages).
        critique: The Critic's verdict as a JSON dict following `CritiqueResult`.
            Also internal: it describes the draft, not the material.
        revision_count: How many times the synthesis has already been revised in
            this run. Bounded by `MAX_REVISION`; the routing function uses it to
            guarantee the graph terminates.
        final_result: The structured result the user is finally shown, either the
            accepted draft or the conservative fallback, as a JSON dict. Only
            `final_result["final_response"]` reaches the chat.
        finalization_status: `passed`, `revised_and_passed` or `safe_fallback`.
            Records which exit the run took, for reports and tests.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    supervisor_plan: NotRequired[dict[str, Any]]
    evidence_by_school: NotRequired[dict[str, list[dict[str, Any]]]]
    evidence_meta: NotRequired[dict[str, Any]]
    specialist_results: NotRequired[Annotated[dict[str, dict[str, Any]], operator.ior]]
    draft_result: NotRequired[dict[str, Any]]
    deterministic_issues: NotRequired[list[dict[str, Any]]]
    critique: NotRequired[dict[str, Any]]
    revision_count: NotRequired[int]
    final_result: NotRequired[dict[str, Any]]
    finalization_status: NotRequired[str]


SPECIALIST_PERSPECTIVES: tuple[SpecialistPerspective, ...] = (
    "freudian",
    "object_relations",
    "lacanian",
)
"""Schools that must each produce exactly one entry in `specialist_results`."""
