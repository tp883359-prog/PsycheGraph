"""UI event contract for the Phase 9 web layer.

Raw LangGraph stream parts never reach the browser. `react_agent.web.streaming`
reads them and this module defines the small, fully typed event set the SSE
endpoint may send:

    LangGraph raw stream
        -> StreamTranslator (streaming.py)
        -> WorkflowEvent (this module)
        -> sse_message() HTML frames
        -> browser

Two rules make the contract safe:

* `WorkflowEvent.metadata` is a closed dataclass, so a new field can only be
  added here, deliberately, with a safe value.
* Node status is a fixed literal set; the UI never infers a state from a
  colour or from the presence of a DOM node.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence

WorkflowStatus = Literal[
    "waiting",
    "running",
    "completed",
    "revising",
    "failed",
    "skipped",
]
"""Status a workflow node can be in. `revising` is the active revision step."""

EventType = Literal["workflow", "answer", "sources", "error", "close"]
"""SSE event names the web layer emits."""

STATUS_TEXT: Mapping[WorkflowStatus, str] = {
    "waiting": "等待",
    "running": "运行中",
    "completed": "已完成",
    "revising": "修订中",
    "failed": "失败",
    "skipped": "未触发",
}
"""Human-readable status label; the UI shows text, not only a colour."""

STATUS_SYMBOL: Mapping[WorkflowStatus, str] = {
    "waiting": "○",
    "running": "◐",
    "completed": "✓",
    "revising": "↻",
    "failed": "✕",
    "skipped": "–",
}
"""Text symbol shown next to the status; keeps the state readable without colour."""


@dataclass(frozen=True)
class NodeSpec:
    """UI description of one graph node.

    Attributes:
        name: Graph node name, exactly as it appears in `graph.py`.
        label: Short Chinese name shown in the workflow panel.
        running_message: Text shown while the node runs.
        phase: Panel grouping (`plan`, `evidence`, `theory`, `synthesis`,
            `review`, `publish`).
        optional: True for nodes that only run on some paths; they start as
            `skipped` instead of `waiting`.
        skipped_message: Explanation shown while an optional node is not used.
    """

    name: str
    label: str
    running_message: str
    phase: str
    optional: bool = False
    skipped_message: str = ""


SPECIALIST_NAMES: tuple[str, ...] = ("freudian", "object_relations", "lacanian")
"""The three theory specialists; they run concurrently in one graph step."""

NODE_SPECS: tuple[NodeSpec, ...] = (
    NodeSpec("supervisor", "分析任务", "正在解析本次请求", "plan"),
    NodeSpec("evidence", "检索理论材料", "正在检索本地知识库", "evidence"),
    NodeSpec("freudian", "弗洛伊德视角", "正在做弗洛伊德式读解", "theory"),
    NodeSpec("object_relations", "客体关系视角", "正在做客体关系读解", "theory"),
    NodeSpec("lacanian", "拉康视角", "正在做拉康式读解", "theory"),
    NodeSpec("synthesizer", "综合三个理论视角", "正在综合三个理论视角", "synthesis"),
    NodeSpec(
        "deterministic_validator",
        "确定性检查",
        "正在做确定性检查",
        "review",
    ),
    NodeSpec("critic", "质量审核", "正在做质量审核", "review"),
    NodeSpec(
        "revise_synthesis",
        "修订综合回答",
        "正在修订综合回答",
        "review",
        optional=True,
        skipped_message="本次审核未要求修订",
    ),
    NodeSpec("finalize", "完成", "正在生成最终回答", "publish"),
    NodeSpec(
        "safe_finalize",
        "采用保守回答",
        "正在生成保守回答",
        "publish",
        optional=True,
        skipped_message="质量审核通过，未使用保守回答",
    ),
)
"""Every node of the production graph, in the order the panel shows them."""

NODE_SPEC_BY_NAME: Mapping[str, NodeSpec] = {spec.name: spec for spec in NODE_SPECS}

PHASE_LABELS: Mapping[str, str] = {
    "plan": "规划",
    "evidence": "证据",
    "theory": "并行理论视角",
    "synthesis": "综合",
    "review": "审查",
    "publish": "发布",
}
"""Group titles used by the workflow panel."""

SCHOOL_LABELS: Mapping[str, str] = {
    "freudian": "弗洛伊德",
    "object_relations": "客体关系",
    "lacanian": "拉康",
    "general": "通用",
}
"""School names shown to the reader."""

ANSWER_NODES: tuple[str, ...] = ("finalize", "safe_finalize")
"""Only these nodes may produce the chat answer."""

SOURCES_NODE = "evidence"
"""Only this node may produce the evidence panel."""

FINALIZATION_LABELS: Mapping[str, str] = {
    "passed": "质量审核：通过",
    "revised_and_passed": "质量审核：修订一次后通过",
    "safe_fallback": "质量审核：未通过，采用保守回答",
}
"""Reader-facing quality status derived from `finalization_status`."""

SAFE_FALLBACK_MESSAGE = "质量检查未能通过，系统采用了更保守的回答。"
"""Fallback note; deliberately says nothing about internal failures."""

RUN_FAILED_MESSAGE = "本次分析未完成，请重试。"
"""Generic run failure text; the real exception stays in the server log."""

ERROR_MESSAGES: Mapping[str, str] = {
    "run_failed": RUN_FAILED_MESSAGE,
    "stream_interrupted": "连接被中断，本次分析可能没有完成。",
}
"""Safe error codes and their reader-facing text."""

EXCEPTION_CODES: Mapping[str, str] = {
    "ValidationError": "run_failed",
    "RuntimeError": "run_failed",
    "BlockingError": "run_failed",
    "GraphRecursionError": "run_failed",
    "TimeoutError": "run_failed",
}
"""Exception names that map to a known safe code; everything else is generic."""


@dataclass(frozen=True)
class TheoryTag:
    """One theory perspective shown next to the final answer.

    Attributes:
        school: Internal school name.
        label: Reader-facing school label.
        completed: Whether the specialist finished in this run. A specialist
            that did not finish is shown as incomplete instead of being hidden.
    """

    school: str
    label: str
    completed: bool


@dataclass(frozen=True)
class CitationRef:
    """One citation label of the answer, mapped to real evidence metadata.

    Attributes:
        label: Short label such as `E1` (without brackets).
        evidence_id: Real id of the cited passage, used to locate its card.
        school: School of the cited passage.
        description: Bibliographic line built from real index metadata only.
    """

    label: str
    evidence_id: str
    school: str
    description: str


@dataclass(frozen=True)
class SourceCard:
    """One evidence card of the evidence panel, sanitized for the browser.

    Attributes:
        evidence_id: Real chunk id.
        label: School label of the card ("弗洛伊德", ...).
        school: Internal school name.
        title: Work title or document title, when the index carries one.
        author: Author, when the index carries one.
        year: Publication year, when the index carries one.
        page: Real page number, when the loader read one.
        section: Section heading, when the source file has one.
        excerpt: Short, truncated passage excerpt.
        detail: Longer passage text, shown only when the reader expands the card.
        source_label: Sanitized source id (never an absolute path).
    """

    evidence_id: str
    school: str
    label: str
    title: str | None
    author: str | None
    year: int | None
    page: int | None
    section: str | None
    excerpt: str
    detail: str
    source_label: str | None


@dataclass(frozen=True)
class SourcesPayload:
    """Payload of a `sources` event.

    Attributes:
        cards: Sanitized evidence cards, grouped by school in the panel.
        counts: Number of cards per school.
        available: Whether the local index answered this run.
        reason: Safe degradation reason (`disabled`, `no-index`, ...), or None.
    """

    cards: tuple[SourceCard, ...]
    counts: Mapping[str, int]
    available: bool
    reason: str | None


@dataclass(frozen=True)
class AnswerPayload:
    """Payload of an `answer` event.

    Attributes:
        text: The synthesis prose (`final_result.final_response`). Rendered with
            escaping; never inserted as raw HTML.
        status: `finalization_status` as stored by the graph.
        status_label: Reader-facing quality line.
        revision_count: How many revisions this run used.
        theory_tags: Theory perspectives with their completion state.
        citations: Citation labels mapped to real evidence ids.
    """

    text: str
    status: str
    status_label: str
    revision_count: int
    theory_tags: tuple[TheoryTag, ...]
    citations: tuple[CitationRef, ...]


@dataclass(frozen=True)
class EventMetadata:
    """Whitelisted detail attached to one UI event.

    Every field is optional and every field is a small, pre-computed value; no
    raw payload, prompt, draft, critique or path can travel through it.

    Attributes:
        duration_seconds: Real wall-clock duration of the node, when measured.
        evidence_count: Number of retrieved passages.
        evidence_counts: Retrieved passages per school.
        verdict: Critic verdict (`pass` / `revise`); never the critique text.
        issues: Number of issues reported by the validator or the critic.
        revision_count: Revisions used so far.
        finalization_status: One of the graph's finalization states.
        specialists: Specialist nodes that finished in this run.
        attempt: Critic pass number (1 for the first review, 2 after a revision).
        answer: Payload for `answer` events.
        sources: Payload for `sources` events.
        error_code: Safe error code for `error` events.
    """

    duration_seconds: float | None = None
    evidence_count: int | None = None
    evidence_counts: Mapping[str, int] | None = None
    verdict: str | None = None
    issues: int | None = None
    revision_count: int | None = None
    finalization_status: str | None = None
    specialists: tuple[str, ...] = ()
    attempt: int | None = None
    answer: AnswerPayload | None = None
    sources: SourcesPayload | None = None
    error_code: str | None = None


@dataclass(frozen=True)
class WorkflowEvent:
    """One UI event, the only shape the SSE endpoint emits.

    Attributes:
        event_type: SSE event name (`workflow`, `answer`, `sources`, `error`,
            `close`).
        node: Graph node this event belongs to, when there is one.
        status: Node status, for `workflow` events.
        label: Reader-facing node or event label.
        message: Short, safe status text.
        run_id: LangGraph run id, once known.
        timestamp: ISO-8601 UTC timestamp of the event.
        elapsed_seconds: Seconds since the run was accepted.
        metadata: Whitelisted detail (see `EventMetadata`).
    """

    event_type: EventType
    node: str | None = None
    status: WorkflowStatus | None = None
    label: str = ""
    message: str = ""
    run_id: str | None = None
    timestamp: str = ""
    elapsed_seconds: float = 0.0
    metadata: EventMetadata = field(default_factory=EventMetadata)


def node_spec(name: str) -> NodeSpec | None:
    """Return the UI description of a graph node.

    Args:
        name: Graph node name from a stream event.

    Returns:
        The spec, or None when the name is unknown (unknown nodes are ignored
        instead of being displayed with raw names).
    """
    return NODE_SPEC_BY_NAME.get(name)


def status_text(status: WorkflowStatus | None) -> str:
    """Return the reader-facing text of a status.

    Args:
        status: Status value or None.

    Returns:
        Chinese status text; an empty string for None.
    """
    if status is None:
        return ""
    return STATUS_TEXT[status]


def status_symbol(status: WorkflowStatus | None) -> str:
    """Return the text symbol of a status.

    Args:
        status: Status value or None.

    Returns:
        A single character; an empty string for None.
    """
    if status is None:
        return ""
    return STATUS_SYMBOL[status]


def school_label(school: str) -> str:
    """Return the reader-facing label of a school.

    Args:
        school: Internal school name.

    Returns:
        Chinese label, falling back to the raw name for unknown schools.
    """
    return SCHOOL_LABELS.get(school, school)


def finalization_label(status: str) -> str:
    """Return the reader-facing quality line of a finalization status.

    Args:
        status: `finalization_status` from the graph.

    Returns:
        Chinese quality line.
    """
    return FINALIZATION_LABELS.get(status, "质量审核：状态未知")


def error_message(code: str) -> str:
    """Return the safe reader-facing text of an error code.

    Args:
        code: Safe error code produced by the adapter.

    Returns:
        Chinese error text.
    """
    return ERROR_MESSAGES.get(code, RUN_FAILED_MESSAGE)


def phases_of(specs: Sequence[NodeSpec]) -> tuple[str, ...]:
    """Return the distinct phases of a node spec sequence, in order.

    Args:
        specs: Node specs.

    Returns:
        Phase keys, first-seen order preserved.
    """
    seen: list[str] = []
    for spec in specs:
        if spec.phase not in seen:
            seen.append(spec.phase)
    return tuple(seen)
