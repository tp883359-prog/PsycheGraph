"""LangGraph stream to UI event translation.

This module is the only place that reads raw LangGraph stream parts. It turns
them into `WorkflowEvent`s that carry nothing but whitelisted values, so the
SSE endpoint can never forward:

* `draft_result`, `critique`, `supervisor_plan` or `specialist_results` bodies,
* token chunks of structured output,
* raw exception payloads,
* local file paths or vector store internals.

What the browser does receive is a node name, a status from a fixed set, a
short Chinese message and a few small counters, plus (once per run) the answer
prose and the sanitized evidence cards.

Two stream modes are requested: `tasks` (node start/result, which gives real
per-node timing) and `updates` (the payload that says *what* a node produced).
Token streaming (`messages`) is deliberately not requested: with structured
output it produces thousands of tool-argument chunks and the visible answer
only exists after the finalizer runs.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, AsyncIterator, Callable, Mapping, Sequence

from react_agent.web.citations import build_sources_payload, citation_refs
from react_agent.web.events import (
    ANSWER_NODES,
    EXCEPTION_CODES,
    NODE_SPECS,
    SAFE_FALLBACK_MESSAGE,
    SOURCES_NODE,
    SPECIALIST_NAMES,
    AnswerPayload,
    CitationRef,
    EventMetadata,
    EventType,
    SourcesPayload,
    TheoryTag,
    WorkflowEvent,
    WorkflowStatus,
    error_message,
    finalization_label,
    node_spec,
    school_label,
)

logger = logging.getLogger(__name__)

STREAM_MODES: tuple[str, ...] = ("tasks", "updates")
"""LangGraph stream modes the web layer asks for."""

DEFAULT_ASSISTANT_ID = "agent"
"""Assistant id from `langgraph.json`."""

PLANNED_SPECIALISTS = len(SPECIALIST_NAMES)
"""How many theory perspectives a run plans to consult."""


@dataclass
class _NodeState:
    """Bookkeeping for one node of the current run."""

    started_at: float | None = None
    update: Mapping[str, Any] | None = None
    reported: bool = False


class StreamTranslator:
    """Turn raw LangGraph stream parts into sanitized UI events.

    One instance handles one run. The translator is synchronous and free of I/O
    so unit tests can replay recorded event sequences directly.

    Attributes:
        run_id: LangGraph run id, filled in from the `metadata` control event.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.perf_counter,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        """Create a translator.

        Args:
            clock: Monotonic clock used for durations (injectable for tests).
            now: Wall-clock factory used for event timestamps.
        """
        self._clock = clock
        self._now = now or (lambda: datetime.now(UTC))
        self._started = clock()
        self._states: dict[str, _NodeState] = {}
        self._sources: SourcesPayload | None = None
        self._citations: tuple[CitationRef, ...] = ()
        self._completed_specialists: list[str] = []
        self._revision_count = 0
        self._critic_runs = 0
        self._answer_sent = False
        self.run_id: str | None = None

    # ------------------------------------------------------------------ public

    @property
    def answer_emitted(self) -> bool:
        """Whether a visible answer has already been produced for this run."""
        return self._answer_sent

    @property
    def sources(self) -> SourcesPayload | None:
        """Sanitized evidence of the current run, once the evidence node ran."""
        return self._sources

    def handle(self, event: str, data: Any) -> list[WorkflowEvent]:
        """Translate one raw stream part.

        Args:
            event: SSE event name from the LangGraph SDK.
            data: Raw payload (never forwarded, only read).

        Returns:
            Zero or more UI events, in the order they should be shown.
        """
        if event == "metadata":
            self._remember_run_id(data)
            return []
        if event == "tasks":
            return self._handle_task(data)
        if event == "updates":
            return self._handle_updates(data)
        if event == "error":
            return self._handle_stream_error(data)
        return []

    def failure(self, exc: BaseException) -> list[WorkflowEvent]:
        """Report a failed run without leaking the exception.

        Args:
            exc: Exception raised while streaming.

        Returns:
            `failed` events for the nodes that were running, then one `error`.
        """
        code = EXCEPTION_CODES.get(type(exc).__name__, "run_failed")
        logger.error("web run failed with %s", type(exc).__name__, exc_info=exc)
        events: list[WorkflowEvent] = []
        for name, state in self._states.items():
            spec = node_spec(name)
            if spec is None or state.reported or state.started_at is None:
                continue
            state.reported = True
            events.append(
                self._workflow(
                    name,
                    "failed",
                    message="这一步没有完成",
                    metadata=EventMetadata(duration_seconds=self._duration(state)),
                )
            )
        events.append(
            self._event(
                "error",
                node=None,
                status=None,
                label="运行失败",
                message=error_message(code),
                metadata=EventMetadata(error_code=code),
            )
        )
        return events

    def recover(self, values: Any) -> list[WorkflowEvent]:
        """Emit a missing answer from checkpointed state.

        Used when the stream ended without a finalizer update (for example the
        browser reconnected after the run finished). Only real state is used;
        nothing is invented.

        Args:
            values: Thread state `values` mapping.

        Returns:
            An `answer` event when the state holds a finalized result and no
            answer was sent yet, otherwise an empty list.
        """
        if self._answer_sent or not isinstance(values, Mapping):
            return []
        final = values.get("final_result")
        if not isinstance(final, Mapping):
            return []
        if self._sources is None:
            self._sources = build_sources_payload(
                values.get("evidence_by_school"), values.get("evidence_meta")
            )
            self._citations = citation_refs(
                final.get("used_evidence_ids"), self._sources.cards
            )
        self._remember_specialists(values.get("specialist_results"))
        self._revision_count = (
            _as_int(values.get("revision_count")) or self._revision_count
        )
        status = str(values.get("finalization_status") or "passed")
        return [self._answer_event(final, status)]

    def finish(self) -> list[WorkflowEvent]:
        """Close the run: mark untouched nodes and emit the final `close`.

        Returns:
            `skipped` events for nodes that never ran, then one `close` event.
        """
        events: list[WorkflowEvent] = []
        for spec in NODE_SPECS:
            state = self._states.get(spec.name)
            if state is not None and state.reported:
                continue
            if state is not None and state.started_at is not None:
                events.append(
                    self._workflow(
                        spec.name,
                        "skipped",
                        message="本次未执行",
                        metadata=EventMetadata(),
                    )
                )
                continue
            message = spec.skipped_message or "本次未执行"
            events.append(
                self._workflow(
                    spec.name,
                    "skipped",
                    message=message,
                    metadata=EventMetadata(),
                )
            )
        events.append(
            self._event(
                "close",
                node=None,
                status=None,
                label="结束",
                message="本次分析已结束",
                metadata=EventMetadata(
                    finalization_status=self._last_finalization_status(),
                    revision_count=self._revision_count,
                ),
            )
        )
        return events

    # ------------------------------------------------------------------ helpers

    def _last_finalization_status(self) -> str | None:
        for name in ANSWER_NODES:
            state = self._states.get(name)
            if state is not None and isinstance(state.update, Mapping):
                status = state.update.get("finalization_status")
                if isinstance(status, str):
                    return status
        return None

    def _event(
        self,
        event_type: EventType,
        *,
        node: str | None,
        status: WorkflowStatus | None,
        label: str,
        message: str,
        metadata: EventMetadata | None = None,
    ) -> WorkflowEvent:
        return WorkflowEvent(
            event_type=event_type,
            node=node,
            status=status,
            label=label,
            message=message,
            run_id=self.run_id,
            timestamp=self._now().isoformat(timespec="seconds"),
            elapsed_seconds=round(self._clock() - self._started, 3),
            metadata=metadata or EventMetadata(),
        )

    def _workflow(
        self,
        name: str,
        status: WorkflowStatus,
        *,
        message: str,
        metadata: EventMetadata | None = None,
    ) -> WorkflowEvent:
        spec = node_spec(name)
        label = spec.label if spec is not None else name
        resolved = metadata or EventMetadata(
            duration_seconds=self._duration(self._states.get(name))
        )
        return self._event(
            "workflow",
            node=name,
            status=status,
            label=label,
            message=message,
            metadata=resolved,
        )

    def _duration(self, state: _NodeState | None) -> float | None:
        if state is None or state.started_at is None:
            return None
        return round(max(self._clock() - state.started_at, 0.0), 3)

    def _remember_run_id(self, data: Any) -> None:
        if isinstance(data, Mapping):
            run_id = data.get("run_id")
            if isinstance(run_id, str) and run_id:
                self.run_id = run_id

    def _remember_specialists(self, results: Any) -> None:
        if not isinstance(results, Mapping):
            return
        for school in SPECIALIST_NAMES:
            if school in results and school not in self._completed_specialists:
                self._completed_specialists.append(school)

    def _handle_task(self, data: Any) -> list[WorkflowEvent]:
        if not isinstance(data, Mapping):
            return []
        name = data.get("name")
        if not isinstance(name, str):
            return []
        spec = node_spec(name)
        if spec is None:
            return []
        state = self._states.setdefault(name, _NodeState())
        if "result" in data:
            if data.get("error"):
                state.reported = True
                return [
                    self._workflow(
                        name,
                        "failed",
                        message=f"{spec.label}没有完成",
                    )
                ]
            return []
        state.started_at = self._clock()
        if name == "critic":
            self._critic_runs += 1
        status: WorkflowStatus = "revising" if name == "revise_synthesis" else "running"
        message = spec.running_message
        if name == "critic" and self._critic_runs > 1:
            message = f"第二次审核（第 {self._critic_runs} 次质量审核）"
        return [
            self._workflow(
                name,
                status,
                message=message,
                metadata=EventMetadata(
                    attempt=self._critic_runs if name == "critic" else None
                ),
            )
        ]

    def _handle_updates(self, data: Any) -> list[WorkflowEvent]:
        if not isinstance(data, Mapping):
            return []
        events: list[WorkflowEvent] = []
        for name, raw in data.items():
            if not isinstance(name, str):
                continue
            spec = node_spec(name)
            if spec is None:
                continue
            payload = raw if isinstance(raw, Mapping) else {}
            state = self._states.setdefault(name, _NodeState())
            state.update = payload
            if name == SOURCES_NODE:
                events.extend(self._evidence_events(name, payload))
            message, metadata = self._completion(name, payload)
            state.reported = True
            events.append(
                self._workflow(name, "completed", message=message, metadata=metadata)
            )
            if name in ANSWER_NODES:
                events.append(
                    self._answer_event(
                        payload.get("final_result"),
                        str(payload.get("finalization_status") or "passed"),
                    )
                )
        return events

    def _evidence_events(
        self, name: str, payload: Mapping[str, Any]
    ) -> list[WorkflowEvent]:
        self._sources = build_sources_payload(
            payload.get("evidence_by_school"), payload.get("evidence_meta")
        )
        return [
            self._event(
                "sources",
                node=name,
                status="completed",
                label="理论依据",
                message=self._evidence_message(),
                metadata=EventMetadata(
                    sources=self._sources,
                    evidence_count=len(self._sources.cards),
                    evidence_counts=self._sources.counts,
                ),
            )
        ]

    def _evidence_message(self) -> str:
        sources = self._sources
        if sources is None or not sources.cards:
            return "本地知识库没有可用材料，本次回答不引用文献"
        return f"找到 {len(sources.cards)} 条证据（按学派分组）"

    def _completion(
        self, name: str, payload: Mapping[str, Any]
    ) -> tuple[str, EventMetadata]:
        spec = node_spec(name)
        label = spec.label if spec is not None else name
        metadata = EventMetadata(
            duration_seconds=self._duration(self._states.get(name))
        )
        if name == "supervisor":
            self._revision_count = _as_int(payload.get("revision_count")) or 0
            return (
                f"已规划 {PLANNED_SPECIALISTS} 个理论视角的分析任务",
                EventMetadata(
                    duration_seconds=metadata.duration_seconds,
                    revision_count=self._revision_count,
                ),
            )
        if name == SOURCES_NODE:
            return self._evidence_message(), metadata
        if name in SPECIALIST_NAMES:
            self._remember_specialists(payload.get("specialist_results"))
            count = _interpretation_count(payload)
            suffix = f" · {count} 条读解" if count else ""
            return f"完成{suffix}", metadata
        if name == "synthesizer":
            used = _used_evidence_count(payload)
            suffix = f" · 引用 {used} 条证据" if used else ""
            return f"综合完成{suffix}", metadata
        if name == "deterministic_validator":
            issues = _issue_count(payload.get("deterministic_issues"))
            if issues:
                return (
                    f"发现 {issues} 个问题",
                    EventMetadata(
                        duration_seconds=metadata.duration_seconds, issues=issues
                    ),
                )
            return "检查通过", metadata
        if name == "critic":
            critique = payload.get("critique")
            verdict = None
            issues = None
            if isinstance(critique, Mapping):
                raw_verdict = critique.get("verdict")
                verdict = str(raw_verdict) if isinstance(raw_verdict, str) else None
                issues = _issue_count(critique.get("issues"))
            attempt = self._critic_runs or 1
            if verdict == "revise":
                message = "发现需要修订的问题，建议修订"
            elif verdict == "pass":
                message = "审核通过"
            else:
                message = "审核结束"
            if attempt > 1:
                message = f"第二次审核：{message}"
            return (
                message,
                EventMetadata(
                    duration_seconds=metadata.duration_seconds,
                    verdict=verdict,
                    issues=issues,
                    attempt=attempt,
                ),
            )
        if name == "revise_synthesis":
            self._revision_count = _as_int(payload.get("revision_count")) or (
                self._revision_count + 1
            )
            return (
                "已按审核意见完成修订",
                EventMetadata(
                    duration_seconds=metadata.duration_seconds,
                    revision_count=self._revision_count,
                ),
            )
        if name == "safe_finalize":
            self._revision_count = (
                _as_int(payload.get("revision_count")) or self._revision_count
            )
            return SAFE_FALLBACK_MESSAGE, EventMetadata(
                duration_seconds=metadata.duration_seconds,
                finalization_status=str(payload.get("finalization_status") or ""),
                revision_count=self._revision_count,
            )
        if name == "finalize":
            self._revision_count = (
                _as_int(payload.get("revision_count")) or self._revision_count
            )
            return "回答已生成", EventMetadata(
                duration_seconds=metadata.duration_seconds,
                finalization_status=str(payload.get("finalization_status") or ""),
                revision_count=self._revision_count,
            )
        return f"{label}完成", metadata

    def _answer_event(self, final: Any, status: str) -> WorkflowEvent:
        self._answer_sent = True
        draft = final if isinstance(final, Mapping) else {}
        text = str(draft.get("final_response") or "")
        sources = self._sources
        self._citations = citation_refs(
            draft.get("used_evidence_ids"), sources.cards if sources else ()
        )
        tags = tuple(
            TheoryTag(
                school=school,
                label=school_label(school),
                completed=school in self._completed_specialists,
            )
            for school in SPECIALIST_NAMES
        )
        payload = AnswerPayload(
            text=text,
            status=status,
            status_label=finalization_label(status),
            revision_count=self._revision_count,
            theory_tags=tags,
            citations=self._citations,
        )
        node = "safe_finalize" if status == "safe_fallback" else "finalize"
        return self._event(
            "answer",
            node=node,
            status="completed",
            label="最终回答",
            message=payload.status_label,
            metadata=EventMetadata(
                answer=payload,
                finalization_status=status,
                revision_count=self._revision_count,
                specialists=tuple(self._completed_specialists),
            ),
        )

    def _handle_stream_error(self, data: Any) -> list[WorkflowEvent]:
        name = ""
        if isinstance(data, Mapping):
            raw = data.get("error") or data.get("error_type") or data.get("type")
            if isinstance(raw, str):
                name = raw.split(":", 1)[0].split("(", 1)[0].strip()
        code = EXCEPTION_CODES.get(name, "run_failed")
        logger.error("run stream reported an error event: %s", name or "unknown")
        events = [
            self._workflow(
                node_name,
                "failed",
                message="这一步没有完成",
            )
            for node_name, state in self._states.items()
            if state.started_at is not None and not state.reported
        ]
        for state in self._states.values():
            state.reported = True
        events.append(
            self._event(
                "error",
                node=None,
                status=None,
                label="运行失败",
                message=error_message(code),
                metadata=EventMetadata(error_code=code),
            )
        )
        return events


def _as_int(value: Any) -> int | None:
    """Return an int for a numeric value, else None."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _issue_count(value: Any) -> int | None:
    """Return the number of issues in a list payload."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value)
    return None


def _interpretation_count(payload: Mapping[str, Any]) -> int | None:
    """Return how many interpretations a specialist reported."""
    results = payload.get("specialist_results")
    if not isinstance(results, Mapping):
        return None
    total = 0
    found = False
    for result in results.values():
        if not isinstance(result, Mapping):
            continue
        items = result.get("interpretations")
        if isinstance(items, Sequence) and not isinstance(items, (str, bytes)):
            total += len(items)
            found = True
    return total if found else None


def _used_evidence_count(payload: Mapping[str, Any]) -> int | None:
    """Return how many evidence ids a synthesis draft cites."""
    draft = payload.get("draft_result")
    if not isinstance(draft, Mapping):
        return None
    used = draft.get("used_evidence_ids")
    if isinstance(used, Sequence) and not isinstance(used, (str, bytes)):
        return len(used)
    return None


async def run_workflow_events(
    client: Any,
    *,
    thread_id: str,
    message: str,
    assistant_id: str = DEFAULT_ASSISTANT_ID,
    translator: StreamTranslator | None = None,
) -> AsyncIterator[WorkflowEvent]:
    """Run the graph and yield sanitized UI events for one user message.

    A run is created *and* streamed in this single request, so no event can be
    missed between "create" and "join" and the browser never needs a second
    endpoint.

    Args:
        client: LangGraph SDK client (in-process or HTTP).
        thread_id: Thread the message belongs to.
        message: User message text.
        assistant_id: Assistant id from `langgraph.json`.
        translator: Optional translator (injectable for tests).

    Yields:
        UI events, always ending with exactly one `close` event.
    """
    active = translator or StreamTranslator()
    try:
        stream = client.runs.stream(
            thread_id,
            assistant_id,
            input={"messages": [{"type": "human", "content": message}]},
            stream_mode=list(STREAM_MODES),
        )
        async for part in stream:
            for event in active.handle(part.event, part.data):
                yield event
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - any failure becomes a safe event
        for event in active.failure(exc):
            yield event
    if not active.answer_emitted:
        values = await _thread_state_values(client, thread_id)
        for event in active.recover(values):
            yield event
    for event in active.finish():
        yield event


async def _thread_state_values(client: Any, thread_id: str) -> Any:
    """Read the thread state values, tolerating any client error.

    Args:
        client: LangGraph SDK client.
        thread_id: Thread to read.

    Returns:
        The `values` mapping of the latest checkpoint, or None.
    """
    try:
        state = await client.threads.get_state(thread_id)
    except Exception as exc:  # noqa: BLE001 - recovery is best effort
        logger.warning("could not read thread state: %s", type(exc).__name__)
        return None
    if isinstance(state, Mapping):
        return state.get("values")
    return None
