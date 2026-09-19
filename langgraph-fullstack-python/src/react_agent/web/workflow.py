"""Workflow panel: the live view of the graph's node states.

The panel shows *states*, never reasoning: a node label, one of the fixed
statuses (`waiting`, `running`, `completed`, `revising`, `failed`, `skipped`),
a short message and the real duration measured from the stream's task events.

Updates are rendered as a log line plus an out-of-band replacement of the
affected row, so the checklist stays correct while the log keeps the order in
which things happened (which is how the revision path becomes visible).
"""

from __future__ import annotations

from typing import Mapping

from fasthtml.common import (
    Div,
    Li,
    Span,
    Ul,
)

from react_agent.web.events import (
    NODE_SPECS,
    PHASE_LABELS,
    NodeSpec,
    WorkflowEvent,
    WorkflowStatus,
    phases_of,
    status_symbol,
    status_text,
)
from react_agent.web.nodes import FT, children

PARALLEL_PHASE_NOTE = "三个视角并行执行，各自独立完成"
"""Explains why three rows of the theory phase move at the same time."""


def format_duration(seconds: float | None) -> str:
    """Format a real duration for the panel.

    Args:
        seconds: Duration in seconds, or None when it was not measured.

    Returns:
        A short string such as `3.2s`, or an empty string.
    """
    if seconds is None:
        return ""
    if seconds < 0.05:
        return "<0.1s"
    return f"{seconds:.1f}s"


def status_badge(status: WorkflowStatus, message: str = "") -> FT:
    """Render the status as text plus symbol (never colour alone).

    Args:
        status: Node status.
        message: Optional message shown next to the status.

    Returns:
        A span node.
    """
    return Span(
        children(
            Span(status_symbol(status), cls="status-symbol", aria_hidden="true"),
            Span(status_text(status), cls="status-text"),
            Span(message, cls="status-message") if message else None,
        ),
        cls=f"status status--{status}",
    )


def workflow_row(
    spec: NodeSpec,
    status: WorkflowStatus,
    message: str = "",
    duration: float | None = None,
    *,
    oob: bool = False,
) -> FT:
    """Render one node row of the workflow checklist.

    Args:
        spec: Node description.
        status: Current status.
        message: Short status message.
        duration: Real duration in seconds, when measured.
        oob: Attach `hx-swap-oob` so the row replaces its counterpart when it
            arrives as part of an SSE payload.

    Returns:
        A list item node.
    """
    duration_text = format_duration(duration)
    attributes: dict[str, str] = {}
    if oob:
        attributes["hx_swap_oob"] = f"outerHTML:#wf-{spec.name}"
    return Li(
        children(
            Span(status_symbol(status), cls="wf-symbol", aria_hidden="true"),
            Div(
                children(
                    Div(spec.label, cls="wf-label"),
                    Div(message, cls="wf-message") if message else None,
                ),
                cls="wf-text",
            ),
            Span(
                children(
                    Span(status_text(status), cls="wf-status-text"),
                    Span(duration_text, cls="wf-duration") if duration_text else None,
                ),
                cls="wf-state",
            ),
        ),
        id=f"wf-{spec.name}",
        cls=f"wf-row wf-row--{status}",
        data_node=spec.name,
        data_status=status,
        role="listitem",
        aria_label=f"{spec.label}：{status_text(status)}",
        **attributes,
    )


def initial_status(spec: NodeSpec) -> WorkflowStatus:
    """Return the status a node starts with in a fresh run.

    Args:
        spec: Node description.

    Returns:
        `skipped` for optional nodes (they only run on some paths) and
        `waiting` for the rest.
    """
    return "skipped" if spec.optional else "waiting"


def initial_message(spec: NodeSpec) -> str:
    """Return the message a node starts with in a fresh run.

    Args:
        spec: Node description.

    Returns:
        The optional-node explanation, or an empty string while waiting.
    """
    return spec.skipped_message if spec.optional else ""


def workflow_panel(
    statuses: Mapping[str, WorkflowStatus] | None = None,
    *,
    oob: bool = False,
) -> FT:
    """Render the whole workflow panel (checklist plus empty log).

    Args:
        statuses: Optional initial status overrides, keyed by node name.
        oob: Attach `hx-swap-oob` so a fresh panel replaces the current one when
            a new run starts.

    Returns:
        The panel section.
    """
    overrides = statuses or {}
    groups: list[FT] = []
    for phase in phases_of(NODE_SPECS):
        specs = [spec for spec in NODE_SPECS if spec.phase == phase]
        rows = [
            workflow_row(
                spec,
                overrides.get(spec.name, initial_status(spec)),
                "" if spec.name in overrides else initial_message(spec),
            )
            for spec in specs
        ]
        groups.append(
            Div(
                children(
                    Div(
                        children(
                            Span(PHASE_LABELS[phase], cls="wf-phase-title"),
                            Span(PARALLEL_PHASE_NOTE, cls="wf-phase-note")
                            if phase == "theory"
                            else None,
                        ),
                        cls="wf-phase-head",
                    ),
                    Ul(
                        *rows,
                        cls="wf-list",
                        aria_label=f"{PHASE_LABELS[phase]}阶段节点",
                    ),
                ),
                cls="wf-group",
            )
        )
    attributes: dict[str, str] = {}
    if oob:
        attributes["hx_swap_oob"] = "outerHTML:#workflow-panel"
    return Div(
        children(
            Div(
                children(
                    Span("Agent Workflow", cls="panel-title"),
                    Span("节点状态，不含模型内部推理", cls="panel-hint"),
                ),
                cls="panel-head",
            ),
            *groups,
            Div(
                children(
                    Span("事件流", cls="panel-subtitle"),
                    Span("按发生顺序追加", cls="panel-hint"),
                ),
                cls="panel-head panel-head--sub",
            ),
            Div(
                id="workflow-log",
                cls="wf-log",
                role="log",
                aria_live="polite",
                aria_relevant="additions text",
            ),
        ),
        id="workflow-panel",
        cls="panel panel--workflow",
        aria_label="Agent 执行流程",
        **attributes,
    )


def workflow_log_line(event: WorkflowEvent) -> FT:
    """Render one chronological log entry for a workflow event.

    Args:
        event: UI event.

    Returns:
        A div node appended to the log.
    """
    duration = format_duration(event.metadata.duration_seconds)
    status = event.status or "running"
    return Div(
        children(
            Span(f"{event.elapsed_seconds:.0f}s", cls="wf-log-time"),
            Span(event.label, cls="wf-log-label"),
            Span(event.message, cls="wf-log-message"),
            Span(duration, cls="wf-log-duration") if duration else None,
        ),
        cls=f"wf-log-line wf-log-line--{status}",
    )


def workflow_event_fragment(event: WorkflowEvent) -> FT:
    """Render a `workflow` event as a log line plus an out-of-band row update.

    Args:
        event: UI event with `node` set.

    Returns:
        A fragment: the log line is swapped into the log, the row replaces its
        counterpart in the checklist.
    """
    parts: list[FT] = [workflow_log_line(event)]
    spec = next((item for item in NODE_SPECS if item.name == event.node), None)
    if spec is not None:
        parts.append(
            workflow_row(
                spec,
                event.status or "waiting",
                event.message,
                event.metadata.duration_seconds,
                oob=True,
            )
        )
    return Div(*parts, cls="wf-event")
