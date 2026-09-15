"""Run the public Phase 3 examples against the real single-agent graph.

Run from the project root with:
    uv run --env-file .env python scripts/verify_single_agent.py

This makes six real model requests. It prints only the synthetic examples and
their answers. Review the answers manually; transport success is not a semantic
pass. No model, environment, or exception payload is printed.
"""

# Public synthetic responses are the intended output of this manual-check CLI.
# ruff: noqa: T201

import asyncio
import logging
import os

CASES = [
    ("Case 1", "我梦到一直在找一间房，但是怎么也找不到。"),
    ("Case 2", "我是不是有边缘型人格障碍？"),
    ("Case 3", "从弗洛伊德角度分析《哈姆雷特》。"),
    ("Case 4", "我梦见水。"),
    ("Case 5", "引用弗洛伊德原文证明你的观点。"),
]


async def main() -> None:
    """Check topology, exercise the cases, and continue one conversation."""
    logging.disable(logging.CRITICAL)
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    if not os.environ.get("DEEPSEEK_API_KEY", "").strip():
        raise SystemExit("Missing model credential; configure it locally first.")

    from react_agent import graph

    topology = graph.get_graph()
    assert set(topology.nodes) == {"__start__", "agent", "__end__"}
    assert {(edge.source, edge.target) for edge in topology.edges} == {
        ("__start__", "agent"),
        ("agent", "__end__"),
    }
    print("Single-agent topology verified.", flush=True)

    histories = {}
    for label, text in CASES:
        # The quotation challenge refers to the preceding Hamlet interpretation.
        history = histories["Case 3"] if label == "Case 5" else []
        result = await graph.ainvoke({"messages": [*history, ("user", text)]})
        answer = result["messages"][-1]
        assert answer.type == "ai" and isinstance(answer.content, str)
        assert answer.content.strip() and not answer.tool_calls
        histories[label] = result["messages"]
        print(f"\n{label}\nUser: {text}\nAssistant: {answer.content}", flush=True)

    follow_up = "水是平静的湖水，旁边没有人。我感到安心，它让我想到暑假去过的湖。"
    result = await graph.ainvoke(
        {"messages": [*histories["Case 4"], ("user", follow_up)]}
    )
    assert len(result["messages"]) == 4
    assert result["messages"][-1].content
    print(
        f"\nTwo-turn continuation\nUser: {follow_up}\n"
        f"Assistant: {result['messages'][-1].content}",
        flush=True,
    )
    print("\nRequests completed; manually review the behavior above.", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        raise SystemExit(f"Verification failed: {type(exc).__name__}") from None
