"""Minimal LangGraph stream inspection for the Phase 9 web UI.

The web layer must translate *real* LangGraph stream events into its own UI
events, so this script records what the installed server actually sends instead
of trusting old tutorials:

    uv run langgraph dev --no-reload --allow-blocking   # in another terminal
    uv run python scripts/inspect_stream.py --message "我梦见水。"

It prints one line per stream part with the event name, the node it belongs to
and the *shape* of the payload (key names, counts, enums). It deliberately never
prints message content, tool-call arguments, prompts, state values or any
environment variable, so the output can be pasted into a report.

The same information is written to a JSONL file (`--out`) for later reference;
that file contains the same sanitized summaries only.
"""

# Progress output is the intended deliverable of this CLI.
# ruff: noqa: T201

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from langgraph_sdk import get_client

DEFAULT_MODES = ("tasks", "updates", "messages-tuple")
"""Stream modes the web UI is considering; checked in one request."""


def _safe_keys(payload: Any) -> list[str]:
    """Return the top-level key names of a payload."""
    if isinstance(payload, dict):
        return sorted(str(key) for key in payload)
    return [type(payload).__name__]


def _summarise_task(payload: Any) -> dict[str, Any]:
    """Describe a `tasks` payload without leaking state.

    Args:
        payload: Task start or task result payload.

    Returns:
        A dict with the task name, whether it is a start or a result, and
        which (non-content) fields are present.
    """
    if not isinstance(payload, dict):
        return {"shape": type(payload).__name__}
    summary: dict[str, Any] = {
        "name": payload.get("name"),
        "kind": "result" if "result" in payload else "start",
        "keys": sorted(str(key) for key in payload),
    }
    triggers = payload.get("triggers")
    if isinstance(triggers, list):
        summary["triggers"] = [str(item) for item in triggers]
    if payload.get("error"):
        summary["error_present"] = True
    interrupts = payload.get("interrupts")
    if isinstance(interrupts, list):
        summary["interrupts"] = len(interrupts)
    return summary


def _summarise_update(node: str, payload: Any) -> dict[str, Any]:
    """Describe one node update without copying its values.

    Args:
        node: Node name the update belongs to.
        payload: The node's partial state update.

    Returns:
        A dict with the updated key names plus a few safe counters (evidence
        counts, verdicts, statuses) that the UI may want to show.
    """
    summary: dict[str, Any] = {"node": node, "keys": _safe_keys(payload)}
    if not isinstance(payload, dict):
        return summary
    meta = payload.get("evidence_meta")
    if isinstance(meta, dict):
        counts = meta.get("counts")
        summary["evidence_available"] = bool(meta.get("available"))
        if isinstance(counts, dict):
            summary["evidence_counts"] = {
                str(school): int(count) for school, count in counts.items()
            }
    critique = payload.get("critique")
    if isinstance(critique, dict):
        summary["verdict"] = critique.get("verdict")
        issues = critique.get("issues")
        summary["issues"] = len(issues) if isinstance(issues, list) else None
    issues = payload.get("deterministic_issues")
    if isinstance(issues, list):
        summary["deterministic_issues"] = len(issues)
    final = payload.get("final_result")
    if isinstance(final, dict):
        summary["used_evidence_ids"] = len(final.get("used_evidence_ids") or [])
    summary["finalization_status"] = payload.get("finalization_status")
    messages = payload.get("messages")
    if isinstance(messages, list):
        summary["messages"] = len(messages)
    if payload.get("specialist_results"):
        summary["specialist_results"] = len(payload["specialist_results"])
    return summary


def _summarise_event(event: str, data: Any) -> dict[str, Any]:
    """Summarise one stream part with content-free metadata.

    Args:
        event: SSE event name (`updates`, `tasks`, `messages`, `metadata`, ...).
        data: Event payload.

    Returns:
        A JSON-serialisable summary describing only shapes and counts.
    """
    if event == "updates":
        if isinstance(data, dict):
            return {"updates": [_summarise_update(str(k), v) for k, v in data.items()]}
        return {"shape": type(data).__name__}
    if event == "tasks":
        return {"task": _summarise_task(data)}
    if event == "messages":
        items = data if isinstance(data, list) else []
        described: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, list) or len(item) != 2:
                continue
            message, metadata = item
            if not isinstance(message, dict):
                continue
            described.append(
                {
                    "type": message.get("type"),
                    "content_chars": len(str(message.get("content") or "")),
                    "tool_calls": len(message.get("tool_calls") or []),
                    "node": (metadata or {}).get("langgraph_node")
                    if isinstance(metadata, dict)
                    else None,
                }
            )
        return {"messages": described}
    if event == "metadata":
        if isinstance(data, dict):
            return {"metadata_keys": _safe_keys(data), "run_id": data.get("run_id")}
        return {"shape": type(data).__name__}
    if event == "error":
        if isinstance(data, dict):
            return {"error_keys": _safe_keys(data), "error_type": data.get("error")}
        return {"shape": type(data).__name__}
    return {"shape": type(data).__name__}


def _describe(summary: dict[str, Any]) -> str:
    """Render a one-line human-readable description of an event summary."""
    if "updates" in summary:
        parts = []
        for update in summary["updates"]:
            detail = ""
            if "evidence_counts" in update:
                detail = f" counts={update['evidence_counts']}"
            elif update.get("verdict"):
                detail = f" verdict={update['verdict']}"
            elif update.get("finalization_status"):
                detail = f" status={update['finalization_status']}"
            elif update.get("keys"):
                detail = f" keys={update['keys']}"
            parts.append(f"{update['node']}{detail}")
        return "updates: " + " | ".join(parts)
    if "task" in summary:
        task = summary["task"]
        return f"task {task.get('kind')}: {task.get('name')}"
    if "messages" in summary:
        described = summary["messages"]
        if not described:
            return "messages: (unparsed)"
        return "messages: " + ", ".join(
            f"{item.get('node')}/{item.get('type')}"
            f" content={item.get('content_chars')}c"
            f" tools={item.get('tool_calls')}"
            for item in described
        )
    return json.dumps(summary, ensure_ascii=False)


async def main() -> int:
    """Stream one run and print a content-free event trace."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("LANGGRAPH_URL", ""))
    parser.add_argument("--assistant", default="agent")
    parser.add_argument("--message", default="我梦见水。")
    parser.add_argument("--thread-id", default=None)
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--out", default="data/stream_inspect.jsonl")
    parser.add_argument(
        "--timeout",
        type=float,
        default=240.0,
        help="Abort after this many seconds (the graph may take 45-80s).",
    )
    args = parser.parse_args()

    client = get_client(url=args.url or None)
    thread_id = args.thread_id
    if thread_id is None:
        thread = await client.threads.create()
        thread_id = str(thread["thread_id"])

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"thread={thread_id} modes={modes}")
    started = time.perf_counter()
    run_id: str | None = None
    seen_nodes: set[str] = set()
    index = 0
    event_counts: dict[str, int] = {}

    def on_created(metadata: Any) -> None:
        nonlocal run_id
        run_id = metadata.get("run_id") if isinstance(metadata, dict) else None
        print(f"run created: {run_id}")

    record = open(out_path, "w", encoding="utf-8")
    try:
        stream = client.runs.stream(
            thread_id,
            args.assistant,
            input={"messages": [{"type": "human", "content": args.message}]},
            stream_mode=modes,
            on_run_created=on_created,
        )
        async for part in stream:
            index += 1
            elapsed = time.perf_counter() - started
            summary = _summarise_event(part.event, part.data)
            event_counts[part.event] = event_counts.get(part.event, 0) + 1
            for update in summary.get("updates", []):
                seen_nodes.add(str(update["node"]))
            if "task" in summary and summary["task"].get("name"):
                seen_nodes.add(str(summary["task"]["name"]))
            print(f"[{index:3d}] {elapsed:6.2f}s {part.event:<9} {_describe(summary)}")
            record.write(
                json.dumps(
                    {
                        "index": index,
                        "elapsed": round(elapsed, 3),
                        "event": part.event,
                        "summary": summary,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            record.flush()
            if elapsed > args.timeout:
                print("timeout reached, stopping")
                break
    finally:
        record.close()

    print(f"total={index} events={event_counts}")
    print(f"nodes seen: {sorted(seen_nodes)}")
    print(f"trace: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
