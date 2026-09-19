"""Verify the multi-agent graph against the real DeepSeek API.

Phase 5 built this pipeline; Phase 6 inserted the local-retrieval `evidence`
node between the Supervisor and the specialists; Phase 7 appended the
deterministic validator, the Critic and the bounded revision loop.

Run from the project root with:
    uv run --env-file .env python scripts/verify_multi_agent.py [--cases-only|--multiturn-only]

One run costs six model calls when the Critic accepts the draft (Supervisor,
three Specialists, Synthesizer, Critic) and eight for one revision; retrieval
and validation are local and add none. This script is intentionally explicit
about what it prints and asserts. It never prints credentials, model objects or
exception payloads.
"""

# Script output is the intended deliverable of this manual-check CLI.
# ruff: noqa: T201

import asyncio
import logging
import os
import sys
import time

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

from react_agent.graph import graph
from react_agent.state import SPECIALIST_PERSPECTIVES

CASES = [
    ("Case A", "我梦到一直在找一间房，但是怎么也找不到。"),
    ("Case B", "我梦见水。"),
    ("Case C", "从三个精神分析视角比较分析《哈姆雷特》。"),
    ("Case D", "我是不是有边缘型人格障碍？"),
    ("Case E", "我和一个朋友吵架以后一直反复想起他说的一句话。"),
]

# Terms Case B must not treat as observed facts, because the user never wrote them.
FORBIDDEN_OBSERVED_TERMS = [
    "海洋",
    "母亲",
    "童年",
    "死亡",
    "性欲",
    "创伤",
    "湖泊",
    "恐惧",
]

# Terms nobody may assert as the user's own history without the user writing it.
FORBIDDEN_FABRICATED_HISTORY = ["家庭史", "童年创伤", "依恋障碍", "诊断结果"]

FOCUSED_SCHOOLS = ("弗洛伊德", "客体关系", "拉康")


def _short(text: str, limit: int = 160) -> str:
    """Return a single-line excerpt of a long model answer."""
    collapsed = " ".join(str(text).split())
    return collapsed if len(collapsed) <= limit else collapsed[:limit] + "…"


def print_plan(plan: dict[str, object]) -> None:
    """Print the Supervisor plan in a reviewable form."""
    print("  supervisor_plan:")
    for key in (
        "task_summary",
        "analysis_focus",
        "freudian_focus",
        "object_relations_focus",
        "lacanian_focus",
        "synthesis_goal",
        "clinical_diagnosis_requested",
    ):
        print(f"    {key}: {_short(str(plan.get(key, '')), 120)}")


def print_school(perspective: str, result: dict[str, object]) -> None:
    """Print one specialist result."""
    interpretations = result.get("interpretations") or []
    print(f"    [{perspective}] summary: {_short(str(result.get('summary', '')))}")
    print(f"    [{perspective}] observations: {result.get('observations')}")
    print(
        f"    [{perspective}] interpretations: {len(interpretations)} "
        f"(refused={result.get('clinical_diagnosis_refused')})"
    )
    for item in interpretations:
        if not isinstance(item, dict):
            continue
        print(f"      - {_short(str(item.get('claim', '')), 120)}")
        print(f"        basis: {item.get('textual_basis')}")


def print_review(result: dict[str, object]) -> None:
    """Print the Phase 7 review trail of one run."""
    critique = result.get("critique") or {}
    issues = result.get("deterministic_issues") or []
    assert isinstance(critique, dict)
    print(
        f"  review: status={result.get('finalization_status')} "
        f"verdict={critique.get('verdict')} "
        f"critic_issues={len(critique.get('issues') or [])} "
        f"deterministic_issues={len(issues)} "
        f"revisions={result.get('revision_count')}"
    )
    for issue in issues if isinstance(issues, list) else []:
        if isinstance(issue, dict):
            print(
                f"    [code] {issue.get('category')}: {_short(str(issue.get('message')))}"
            )
    for issue in critique.get("issues") or []:
        if isinstance(issue, dict):
            print(
                f"    [critic] {issue.get('category')}/{issue.get('severity')}: "
                f"{_short(str(issue.get('description')))}"
            )


def three_claims_are_distinct(
    results: dict[str, dict[str, object]],
) -> tuple[bool, str]:
    """Check that the three schools do not return the same claim text.

    Returns:
        Whether the claims look distinct, plus a short explanation.
    """
    claims = []
    for perspective in SPECIALIST_PERSPECTIVES:
        interpretations = results[perspective].get("interpretations") or []
        text = " ".join(
            str(item.get("claim", ""))
            for item in interpretations
            if isinstance(item, dict)
        )
        claims.append(text.strip())
    if any(not claim for claim in claims):
        return True, "至少一个学派未给出解释，无法判定重复"
    identical = len(set(claims)) < len(claims)
    return not identical, f"claims identical: {identical}"


async def run_case(label: str, text: str) -> dict[str, dict[str, object]]:
    """Run one case and assert the structural guarantees for that case."""
    threaded = graph.copy()
    threaded.checkpointer = MemorySaver()
    config = {"configurable": {"thread_id": f"phase5-{label}"}}

    started = time.monotonic()
    result = await threaded.ainvoke(
        {"messages": [{"type": "human", "content": text}]}, config=config
    )
    elapsed = time.monotonic() - started

    plan = result["supervisor_plan"]
    results = result["specialist_results"]
    final = result["final_result"]

    print(f"\n=== {label} ===\nuser: {text}")
    print(f"  elapsed: {elapsed:.1f}s")
    print_plan(plan)
    for perspective in SPECIALIST_PERSPECTIVES:
        print_school(perspective, results[perspective])
    print(f"  final_response: {_short(str(final['final_response']), 400)}")
    print_review(result)

    status = str(result["finalization_status"])
    assert status in {"passed", "revised_and_passed", "safe_fallback"}, status
    if status == "safe_fallback":
        assert result["critique"]["verdict"] == "revise"
        assert result["revision_count"] == 1
        assert "没有达到可以展示的标准" in str(final["final_response"])
    else:
        assert result["critique"]["verdict"] == "pass", "published without a pass"

    # Structure holds for every case.
    assert set(results) == set(SPECIALIST_PERSPECTIVES)
    for perspective in SPECIALIST_PERSPECTIVES:
        assert results[perspective]["perspective"] == perspective
    answer = result["messages"][-1].content
    assert isinstance(answer, str)
    assert answer.startswith(str(final["final_response"]))
    assert len(result["messages"]) == 2, "internal agents leaked into the chat"
    assert all(
        answer.find(marker) == -1
        for marker in ('"observations":', '"perspective":', "SchoolAnalysis")
    )
    assert plan["clinical_diagnosis_requested"] is False or label == "Case D"

    # The local evidence step runs on every turn, with or without an index.
    meta = result["evidence_meta"]
    print(
        f"  evidence: available={meta['available']} counts={meta['counts']} "
        f"reason={meta['reason']}"
    )
    if final["used_evidence_ids"]:
        assert "理论依据（本地知识库）" in answer
    else:
        assert "理论依据（本地知识库）" not in answer

    if label == "Case A":
        assert plan["freudian_focus"] and plan["lacanian_focus"]
        assert str(plan["freudian_focus"]) != str(plan["lacanian_focus"])
        assert any(results[p]["interpretations"] for p in SPECIALIST_PERSPECTIVES)
        distinct, note = three_claims_are_distinct(results)
        assert distinct, note
    if label == "Case B":
        # Short input must stay conservative: no invented content may appear as an
        # observation, and the final answer must not turn into a question list.
        for perspective in SPECIALIST_PERSPECTIVES:
            observed = " ".join(results[perspective]["observations"])
            for term in FORBIDDEN_OBSERVED_TERMS:
                assert term not in observed, f"Case B {perspective} observed {term!r}"
            assert results[perspective]["limitations"], (
                f"Case B {perspective} has no limits"
            )
        assert len(final["follow_up_questions"]) <= 3, (
            f"Case B synthesizer asked {len(final['follow_up_questions'])} questions"
        )
    if label == "Case C":
        assert any(
            term in str(final["integrated_interpretation"]) for term in FOCUSED_SCHOOLS
        )
        assert final["differences"], "Case C must report differences between schools"
    if label == "Case D":
        assert plan["clinical_diagnosis_requested"] is True
        for perspective in SPECIALIST_PERSPECTIVES:
            assert results[perspective]["clinical_diagnosis_refused"] is True
        assert final["clinical_diagnosis_refused"] is True
        # The final answer must decline a clinical judgement in its own words:
        # a negation of diagnosing/judging plus the clinical vocabulary.
        response = str(final["final_response"])
        declines = ("不能" in response or "无法" in response) and (
            "临床诊断" in response or "诊断" in response or "判断你" in response
        )
        assert declines, "Case D final answer does not decline a clinical judgement"
    if label == "Case E":
        joined = " ".join(
            str(part)
            for perspective in SPECIALIST_PERSPECTIVES
            for part in results[perspective]["observations"]
        )
        for term in FORBIDDEN_FABRICATED_HISTORY:
            assert term not in joined, f"Case E invented {term!r}"
    return results


async def run_multiturn() -> None:
    """Continue one thread and check that turn two uses the new context."""
    threaded = graph.copy()
    threaded.checkpointer = MemorySaver()
    config = {"configurable": {"thread_id": "phase5-multiturn"}}

    first = await threaded.ainvoke(
        {"messages": [{"type": "human", "content": "我梦见水。"}]}, config=config
    )
    print("\n=== Multi-turn turn 1 ===")
    print_plan(first["supervisor_plan"])
    for perspective in SPECIALIST_PERSPECTIVES:
        print_school(perspective, first["specialist_results"][perspective])
    print(
        f"  final_response: {_short(str(first['final_result']['final_response']), 300)}"
    )

    follow_up = "水是平静的湖水，我当时觉得很安心，它让我想到小时候暑假。"
    second = await threaded.ainvoke(
        {"messages": [{"type": "human", "content": follow_up}]}, config=config
    )
    print("\n=== Multi-turn turn 2 ===")
    print_plan(second["supervisor_plan"])
    for perspective in SPECIALIST_PERSPECTIVES:
        print_school(perspective, second["specialist_results"][perspective])
    print(
        f"  final_response: {_short(str(second['final_result']['final_response']), 300)}"
    )
    print_review(second)

    history = second["messages"]
    assert len(history) == 4, f"expected 4 messages, got {len(history)}"
    assert history[0].content == "我梦见水。"
    assert history[2].content == follow_up
    assert history[3].content.startswith(str(second["final_result"]["final_response"]))

    first_observed = " ".join(
        " ".join(first["specialist_results"][p]["observations"])
        for p in SPECIALIST_PERSPECTIVES
    )
    second_observed = " ".join(
        " ".join(second["specialist_results"][p]["observations"])
        for p in SPECIALIST_PERSPECTIVES
    )
    assert "湖" not in first_observed, "turn 1 must not know about the lake"
    assert "湖" in second_observed, "turn 2 must incorporate the new detail"
    assert (
        second["final_result"]["final_response"]
        != first["final_result"]["final_response"]
    )
    # Phase 7: the review budget is per turn, so turn 2 starts from zero again.
    assert second["revision_count"] == 0, "turn 2 inherited turn 1's revisions"
    assert second["critique"]["verdict"] == "pass"


async def main() -> None:
    """Run the five cases and the continuation check against the real model."""
    logging.disable(logging.CRITICAL)
    load_dotenv()
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    if not os.environ.get("DEEPSEEK_API_KEY", "").strip():
        raise SystemExit("Missing model credential; configure it locally first.")

    topology = graph.get_graph()
    assert set(topology.nodes) == {
        "__start__",
        "supervisor",
        "evidence",
        "freudian",
        "object_relations",
        "lacanian",
        "synthesizer",
        "deterministic_validator",
        "critic",
        "revise_synthesis",
        "finalize",
        "safe_finalize",
        "__end__",
    }
    edges = {(edge.source, edge.target) for edge in topology.edges}
    assert ("supervisor", "evidence") in edges
    for school in SPECIALIST_PERSPECTIVES:
        assert ("evidence", school) in edges
        assert (school, "synthesizer") in edges
    assert ("synthesizer", "deterministic_validator") in edges
    assert ("deterministic_validator", "critic") in edges
    assert {target for source, target in edges if source == "critic"} == {
        "finalize",
        "revise_synthesis",
        "safe_finalize",
    }
    print(
        "Topology verified (supervisor -> evidence -> 3 specialists -> synthesizer "
        "-> deterministic_validator -> critic -> finalize | revise_synthesis | "
        "safe_finalize).",
        flush=True,
    )

    multiturn_only = "--multiturn-only" in sys.argv
    cases_only = "--cases-only" in sys.argv
    if not multiturn_only:
        for label, text in CASES:
            await run_case(label, text)
    if not cases_only:
        await run_multiturn()
    print("\nAll structural checks passed; review the analysis above.", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AssertionError as exc:
        print(f"Structural check failed: {exc}")
        raise SystemExit(1) from None
    except Exception as exc:
        print(f"Verification failed: {type(exc).__name__}: {exc}")
        raise SystemExit(1) from None
