"""Event to HTML fragments: the last step before the SSE frame.

Each `WorkflowEvent` becomes one fragment, and the route wraps it with
FastHTML's `sse_message`. The mapping from event type to fragment is the only
place that decides what a named SSE event looks like in the browser:

    workflow -> log line + out-of-band row + status line
    sources  -> inner body of the evidence panel
    answer   -> assistant bubble appended to the chat
    error    -> system bubble appended to the chat
    close    -> final status bar + out-of-band composer restore
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from fasthtml.common import Div, sse_message, to_xml

from react_agent.web.citations import build_sources_payload, citation_refs
from react_agent.web.components import (
    answer_fragment,
    close_fragment,
    error_fragment,
    evidence_body,
    progress_fragment,
)
from react_agent.web.events import CitationRef, SourcesPayload, WorkflowEvent
from react_agent.web.nodes import FT, children
from react_agent.web.workflow import workflow_event_fragment


def citations_for_state(values: Any) -> tuple[CitationRef, ...]:
    """Rebuild the citation links of the newest answer from checkpoint state.

    Args:
        values: Thread state values.

    Returns:
        Citation references, or an empty tuple when the state holds no
        finalized result. Only real `used_evidence_ids` are mapped.
    """
    if not isinstance(values, Mapping):
        return ()
    final = values.get("final_result")
    if not isinstance(final, Mapping):
        return ()
    return citation_refs(
        final.get("used_evidence_ids"), sources_for_state(values).cards
    )


def sources_for_state(values: Any) -> SourcesPayload:
    """Rebuild the evidence panel payload from checkpoint state.

    Args:
        values: Thread state values.

    Returns:
        Sanitized sources of the newest run; empty when the state has none.
    """
    if not isinstance(values, Mapping):
        return build_sources_payload({}, None)
    return build_sources_payload(
        values.get("evidence_by_school"), values.get("evidence_meta")
    )


def event_fragment(
    event: WorkflowEvent,
    *,
    thread_id: str,
    answer_seen: bool = False,
) -> FT | None:
    """Render one UI event as the HTML the browser swaps in.

    Args:
        event: Sanitized UI event.
        thread_id: Current thread (needed by the close event).
        answer_seen: Whether an answer or error bubble was already appended.

    Returns:
        The fragment, or None for events that carry no content.
    """
    if event.event_type == "workflow":
        return Div(
            children(
                workflow_event_fragment(event),
                progress_fragment(event),
            ),
        )
    if event.event_type == "sources":
        return evidence_body(event.metadata.sources)
    if event.event_type == "answer":
        return answer_fragment(event)
    if event.event_type == "error":
        return error_fragment(event)
    if event.event_type == "close":
        return close_fragment(event, thread_id, remove_placeholder=not answer_seen)
    return None


def frame_for(
    event: WorkflowEvent,
    *,
    thread_id: str,
    answer_seen: bool = False,
) -> str | None:
    """Render one UI event as a complete SSE frame.

    Args:
        event: Sanitized UI event.
        thread_id: Current thread.
        answer_seen: Whether an answer or error bubble was already appended.

    Returns:
        A `text/event-stream` frame, or None when the event has no content.
    """
    fragment = event_fragment(event, thread_id=thread_id, answer_seen=answer_seen)
    if fragment is None:
        return None
    return str(sse_message(fragment, event=event.event_type))


def empty_stream_frame() -> str:
    """Return the frame sent when there is nothing to stream.

    Returns:
        A `close` frame, which makes the browser close the EventSource.
    """
    return str(sse_message(Div("没有正在运行的分析。"), event="close"))


def frame_text(frames: Sequence[str]) -> str:
    """Join frames into one event-stream body (used by tests).

    Args:
        frames: SSE frames.

    Returns:
        The concatenated stream body.
    """
    return "".join(frames)


def debug_render(fragment: Any) -> str:
    """Render a fragment to HTML for tests and logs.

    Args:
        fragment: FastHTML node.

    Returns:
        The rendered HTML.
    """
    return str(to_xml(fragment))
