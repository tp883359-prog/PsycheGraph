"""Test doubles and recorded event shapes for the Phase 9 web tests.

The event shapes below were captured from a real `langgraph dev` run with
`scripts/inspect_stream.py` (stream modes `tasks` + `updates`), so the web
adapter is tested against what the installed server actually sends:

    metadata   {"run_id": ..., "attempt": 1}
    tasks      start:  {"id", "name", "input", "metadata", "triggers"}
               result: {"id", "name", "error", "interrupts", "result"}
    updates    {"<node>": <partial state>}

No test here calls a model or the network.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from langgraph_sdk.schema import StreamPart

Event = tuple[str, Any]
"""One recorded (event name, payload) pair."""

DRAFT_TEXT = "草稿内容：你很可能被母亲压抑，这是未审核的草稿。"
"""Internal draft prose; it must never reach the browser."""

CRITIQUE_TEXT = "审核意见：结论缺乏证据支持，需要修订。"
"""Internal critique prose; it must never reach the browser."""

FAKE_ANSWER = "目前只知道你梦见了水。可以多说一点梦里的场景吗？"
"""A plausible final answer."""

WINDOWS_PATH = r"C:\Users\demo\Desktop\PsycheGraph\data\secret\notes.md"
"""A local path that must never appear in any UI event."""


def make_evidence_item(
    evidence_id: str,
    school: str,
    *,
    text: str = "梦的工作把愿望改写成意象，这是一种伪装。",
    source_path: str = WINDOWS_PATH,
    **overrides: Any,
) -> dict[str, Any]:
    """Build one evidence entry shaped like the retriever's output.

    Args:
        evidence_id: Chunk id.
        school: School shelf.
        text: Passage text.
        source_path: Local path (the UI must drop it).
        **overrides: Extra or replacement fields.

    Returns:
        The evidence dict.
    """
    payload: dict[str, Any] = {
        "evidence_id": evidence_id,
        "school": school,
        "text": text,
        "source_id": evidence_id.rsplit("_", 1)[0],
        "title": "项目自编测试材料",
        "author": "PsycheGraph test fixture",
        "work_title": "项目自编测试材料",
        "year": None,
        "page": None,
        "section": "梦的工作与愿望",
        "source_path": source_path,
        "retrieval_score": 0.42,
    }
    payload.update(overrides)
    return payload


def evidence_by_school() -> dict[str, list[dict[str, Any]]]:
    """Build a small three-school evidence mapping.

    Returns:
        Mapping from school to evidence entries.
    """
    return {
        "freudian": [make_evidence_item("freud_dreams_000001", "freudian")],
        "object_relations": [
            make_evidence_item("object_rel_000002", "object_relations")
        ],
        "lacanian": [make_evidence_item("lacan_signifier_000003", "lacanian")],
    }


def specialist_payload(school: str, *, claims: int = 2) -> dict[str, Any]:
    """Build one specialist's `specialist_results` contribution.

    Args:
        school: School name.
        claims: Number of interpretations.

    Returns:
        The partial state update of that specialist.
    """
    return {
        "specialist_results": {
            school: {
                "perspective": school,
                "observations": [f"{school} 观察：材料只有一个意象。"],
                "interpretations": [
                    {
                        "perspective": school,
                        "claim": f"草稿解读 {index}",
                        "textual_basis": ["我梦见水"],
                        "uncertainty": "材料不足。",
                        "evidence_ids": [],
                    }
                    for index in range(claims)
                ],
                "limitations": ["材料不足。"],
                "questions": ["水是什么状态？"],
                "summary": f"{school} 小结。",
            }
        }
    }


def draft_payload(
    *, used: Sequence[str] = (), body: str = FAKE_ANSWER
) -> dict[str, Any]:
    """Build a synthesis payload (`draft_result` / `final_result`).

    Args:
        used: Cited evidence ids.
        body: Visible prose.

    Returns:
        The synthesis dict.
    """
    return {
        "common_ground": "三个学派都注意到材料很少。",
        "differences": "弗洛伊德谈愿望，拉康谈能指。",
        "integrated_interpretation": "只能给出条件性读解。",
        "limitations": ["缺少场景与感受。"],
        "follow_up_questions": ["水的状态？"],
        "clinical_diagnosis_refused": False,
        "used_evidence_ids": list(used),
        "final_response": body,
    }


def critique_payload(*, verdict: str, issue_count: int = 1) -> dict[str, Any]:
    """Build a `critique` payload whose prose must stay internal.

    Args:
        verdict: `pass` or `revise`.
        issue_count: Number of issues reported.

    Returns:
        The critique dict.
    """
    return {
        "verdict": verdict,
        "issues": [
            {
                "category": "evidence_support",
                "severity": "error",
                "description": CRITIQUE_TEXT,
                "affected_claim": "该证据证明用户正在使用压抑。",
                "related_perspective": "freudian",
                "related_evidence_ids": [],
                "revision_instruction": "降级为条件性表述。",
            }
        ]
        * issue_count,
        "summary": CRITIQUE_TEXT,
        "revision_instructions": [CRITIQUE_TEXT],
        "clinical_safety_ok": True,
        "evidence_grounding_ok": True,
        "observation_fidelity_ok": True,
    }


def task_start(name: str, index: int = 1) -> Event:
    """Build a `tasks` start event.

    Args:
        name: Node name.
        index: Task index used for the id.

    Returns:
        The recorded event pair.
    """
    return (
        "tasks",
        {
            "id": f"task-{index}",
            "name": name,
            "input": {"messages": ["..."]},
            "metadata": {},
            "triggers": [f"branch:to:{name}"],
        },
    )


def task_result(name: str, index: int = 1, *, error: Any = None) -> Event:
    """Build a `tasks` result event.

    Args:
        name: Node name.
        index: Task index used for the id.
        error: Optional task error.

    Returns:
        The recorded event pair.
    """
    return (
        "tasks",
        {
            "id": f"task-{index}",
            "name": name,
            "error": error,
            "interrupts": [],
            "result": {"messages": []},
        },
    )


def update(node: str, payload: Mapping[str, Any]) -> Event:
    """Build an `updates` event for one node.

    Args:
        node: Node name.
        payload: Partial state update.

    Returns:
        The recorded event pair.
    """
    return ("updates", {node: dict(payload)})


def metadata_event(run_id: str = "run-1") -> Event:
    """Build the `metadata` control event.

    Args:
        run_id: Run id the server reports.

    Returns:
        The recorded event pair.
    """
    return ("metadata", {"run_id": run_id, "attempt": 1})


def normal_sequence(*, cited: Sequence[str] = ()) -> list[Event]:
    """Build a complete pass sequence: all nodes, no revision.

    Args:
        cited: Evidence ids the synthesis cites.

    Returns:
        The recorded event sequence.
    """
    return [
        metadata_event(),
        task_start("supervisor"),
        update(
            "supervisor",
            {"supervisor_plan": {"task_summary": "分析水梦"}, "revision_count": 0},
        ),
        task_result("supervisor"),
        task_start("evidence"),
        update(
            "evidence",
            {
                "evidence_by_school": evidence_by_school(),
                "evidence_meta": {
                    "available": True,
                    "reason": None,
                    "elapsed_seconds": 2.3,
                    "counts": {"freudian": 1, "object_relations": 1, "lacanian": 1},
                },
            },
        ),
        task_result("evidence"),
        task_start("freudian", 2),
        task_start("object_relations", 3),
        task_start("lacanian", 4),
        update("freudian", specialist_payload("freudian")),
        task_result("freudian", 2),
        update("object_relations", specialist_payload("object_relations")),
        task_result("object_relations", 3),
        update("lacanian", specialist_payload("lacanian")),
        task_result("lacanian", 4),
        task_start("synthesizer", 5),
        update("synthesizer", {"draft_result": draft_payload(used=cited)}),
        task_result("synthesizer", 5),
        task_start("deterministic_validator", 6),
        update("deterministic_validator", {"deterministic_issues": []}),
        task_result("deterministic_validator", 6),
        task_start("critic", 7),
        update("critic", {"critique": critique_payload(verdict="pass")}),
        task_result("critic", 7),
        task_start("finalize", 8),
        update(
            "finalize",
            {
                "final_result": draft_payload(used=cited),
                "finalization_status": "passed",
                "messages": [{"type": "ai", "content": FAKE_ANSWER}],
            },
        ),
        task_result("finalize", 8),
    ]


def revision_sequence(*, cited: Sequence[str] = ()) -> list[Event]:
    """Build a sequence that revises once before finalizing.

    Args:
        cited: Evidence ids the synthesis cites.

    Returns:
        The recorded event sequence.
    """
    return [
        metadata_event("run-revise"),
        task_start("supervisor"),
        update(
            "supervisor",
            {"supervisor_plan": {"task_summary": "分析"}, "revision_count": 0},
        ),
        task_result("supervisor"),
        task_start("evidence"),
        update(
            "evidence",
            {
                "evidence_by_school": evidence_by_school(),
                "evidence_meta": {
                    "available": True,
                    "reason": None,
                    "counts": {"freudian": 1, "object_relations": 1, "lacanian": 1},
                },
            },
        ),
        task_result("evidence"),
        task_start("freudian", 2),
        update("freudian", specialist_payload("freudian")),
        task_result("freudian", 2),
        task_start("object_relations", 3),
        update("object_relations", specialist_payload("object_relations")),
        task_result("object_relations", 3),
        task_start("lacanian", 4),
        update("lacanian", specialist_payload("lacanian")),
        task_result("lacanian", 4),
        task_start("synthesizer", 5),
        update(
            "synthesizer", {"draft_result": draft_payload(used=cited, body=DRAFT_TEXT)}
        ),
        task_result("synthesizer", 5),
        task_start("deterministic_validator", 6),
        update("deterministic_validator", {"deterministic_issues": []}),
        task_result("deterministic_validator", 6),
        task_start("critic", 7),
        update("critic", {"critique": critique_payload(verdict="revise")}),
        task_result("critic", 7),
        task_start("revise_synthesis", 8),
        update(
            "revise_synthesis",
            {"draft_result": draft_payload(used=cited), "revision_count": 1},
        ),
        task_result("revise_synthesis", 8),
        task_start("deterministic_validator", 9),
        update("deterministic_validator", {"deterministic_issues": []}),
        task_result("deterministic_validator", 9),
        task_start("critic", 10),
        update("critic", {"critique": critique_payload(verdict="pass")}),
        task_result("critic", 10),
        task_start("finalize", 11),
        update(
            "finalize",
            {
                "final_result": draft_payload(used=cited),
                "finalization_status": "revised_and_passed",
                "messages": [{"type": "ai", "content": FAKE_ANSWER}],
            },
        ),
        task_result("finalize", 11),
    ]


def fallback_sequence() -> list[Event]:
    """Build a sequence that ends in `safe_finalize`.

    Returns:
        The recorded event sequence.
    """
    return [
        metadata_event("run-fallback"),
        task_start("supervisor"),
        update("supervisor", {"revision_count": 0}),
        task_result("supervisor"),
        task_start("evidence"),
        update(
            "evidence",
            {
                "evidence_by_school": evidence_by_school(),
                "evidence_meta": {
                    "available": False,
                    "reason": "no-index",
                    "counts": {},
                },
            },
        ),
        task_result("evidence"),
        task_start("synthesizer", 5),
        update("synthesizer", {"draft_result": draft_payload(body=DRAFT_TEXT)}),
        task_result("synthesizer", 5),
        task_start("deterministic_validator", 6),
        update("deterministic_validator", {"deterministic_issues": []}),
        task_result("deterministic_validator", 6),
        task_start("critic", 7),
        update("critic", {"critique": critique_payload(verdict="revise")}),
        task_result("critic", 7),
        task_start("revise_synthesis", 8),
        update("revise_synthesis", {"revision_count": 1}),
        task_result("revise_synthesis", 8),
        task_start("critic", 9),
        update("critic", {"critique": critique_payload(verdict="revise")}),
        task_result("critic", 9),
        task_start("safe_finalize", 10),
        update(
            "safe_finalize",
            {
                "final_result": draft_payload(
                    body="保守回答：只保留材料中出现的内容。"
                ),
                "finalization_status": "safe_fallback",
                "messages": [{"type": "ai", "content": "保守回答"}],
            },
        ),
        task_result("safe_finalize", 10),
    ]


def parts(sequence: Iterable[Event]) -> list[StreamPart]:
    """Wrap recorded events into SDK stream parts.

    Args:
        sequence: Recorded event pairs.

    Returns:
        Stream parts as the SDK yields them.
    """
    return [StreamPart(event=event, data=data, id=None) for event, data in sequence]


class FakeRuns:
    """Replay a recorded sequence in place of `client.runs`."""

    def __init__(
        self, sequence: Sequence[Event], *, error: BaseException | None = None
    ) -> None:
        """Store the sequence to replay.

        Args:
            sequence: Recorded event pairs.
            error: Optional exception raised mid-stream.
        """
        self.sequence = list(sequence)
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def stream(
        self,
        thread_id: str,
        assistant_id: str,
        *,
        input: Any = None,  # noqa: A002 - mirrors the SDK signature
        stream_mode: Any = None,
    ) -> Any:
        """Return an async generator over the recorded events.

        Args:
            thread_id: Thread id.
            assistant_id: Assistant id.
            input: Run input.
            stream_mode: Requested stream modes.

        Returns:
            An async iterator of stream parts.
        """
        self.calls.append(
            {
                "thread_id": thread_id,
                "assistant_id": assistant_id,
                "input": input,
                "stream_mode": list(stream_mode or []),
            }
        )

        async def generator() -> Any:
            for part in parts(self.sequence):
                yield part
            if self.error is not None:
                raise self.error

        return generator()


class FakeThreads:
    """Minimal `client.threads` replacement."""

    def __init__(
        self,
        *,
        values: Mapping[str, Any] | None = None,
        threads: Sequence[Mapping[str, Any]] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Store the canned thread data.

        Args:
            values: State `values` returned by `get_state`.
            threads: Records returned by `search`.
            metadata: Thread metadata returned by `get`.
        """
        self.values = dict(values or {})
        self.thread_records = list(threads)
        self.metadata = dict(metadata or {})
        self.created: list[dict[str, Any]] = []
        self.updates: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Mapping[str, Any]:
        """Record a thread creation."""
        self.created.append(dict(kwargs))
        return {
            "thread_id": kwargs.get("thread_id"),
            "metadata": kwargs.get("metadata"),
        }

    async def get_state(self, thread_id: str) -> Mapping[str, Any]:
        """Return the canned state."""
        return {"values": dict(self.values)}

    async def search(
        self, metadata: Any = None, limit: int = 50
    ) -> Sequence[Mapping[str, Any]]:
        """Return the canned thread list."""
        return list(self.thread_records)

    async def get(self, thread_id: str) -> Mapping[str, Any]:
        """Return canned thread metadata."""
        return {"thread_id": thread_id, "metadata": dict(self.metadata)}

    async def update(
        self, thread_id: str, *, metadata: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Merge metadata and remember the call."""
        self.updates.append(dict(metadata))
        self.metadata.update(metadata)
        return {"thread_id": thread_id, "metadata": dict(self.metadata)}


class FakeClient:
    """Client double with `runs` and `threads` parts."""

    def __init__(
        self,
        sequence: Sequence[Event] = (),
        *,
        state: Mapping[str, Any] | None = None,
        threads: Sequence[Mapping[str, Any]] = (),
        metadata: Mapping[str, Any] | None = None,
        error: BaseException | None = None,
    ) -> None:
        """Build the double.

        Args:
            sequence: Events `runs.stream` replays.
            state: State values for `threads.get_state`.
            threads: Thread records for `threads.search`.
            metadata: Thread metadata for `threads.get` / `update`.
            error: Exception raised after the recorded events.
        """
        self.runs = FakeRuns(sequence, error=error)
        self.threads = FakeThreads(values=state, threads=threads, metadata=metadata)
