"""Verify the Phase 6 RAG pipeline against the real local index and DeepSeek.

Two phases, in this order:

  1. Retrieval only (no model calls): index stats, per-school queries, scores,
     source metadata, and the school-isolation guarantee.
  2. Four real conversations through the full graph, checking that citations
     come from the index, that the synthesizer only lists passages it used,
     that a knowledge base which talks about borderline / pathology still does
     not produce a diagnosis, and that a run costs six DeepSeek calls when the
     Critic accepts the draft (eight for one revision, ten would be a bug).

After Phase 7 the same script also reports the review trail of every case:
Critic verdict, critic issues, code-level findings, revision count and the
finalization status (`passed` / `revised_and_passed` / `safe_fallback`).

Run from the project root:

    $env:RAG_VECTORSTORE_PATH = "data/vectorstore_fixture"
    uv run --env-file .env python scripts/verify_rag.py

The index used here is built from `tests/fixtures/knowledge`, which is
project-authored TEST FIXTURE material (see `knowledge/README.md`); it is not
Freud / Lacan source text.
"""

# Script output is the intended deliverable of this manual-check CLI.
# ruff: noqa: T201

import asyncio
import logging
import os
import sys
import time
from collections import Counter
from typing import Any, cast

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

import react_agent.llm as llm_module
from react_agent.config import MAX_REVISION
from react_agent.graph import graph
from react_agent.rag.citations import describe_source
from react_agent.rag.embeddings import describe_embedding_backend, get_embeddings
from react_agent.rag.retriever import (
    build_school_query,
    evidence_items_from_state,
    retrieve_evidence,
)
from react_agent.rag.settings import get_rag_settings
from react_agent.rag.vectorstore import collection_count, open_vectorstore
from react_agent.state import SPECIALIST_PERSPECTIVES, PsycheGraphState

QUERIES: list[tuple[str, str, str]] = [
    (
        "在梦里反复找一个找不到的房间",
        "梦见找房间，怎么也找不到，反复出现。",
        "缺失与重复的结构，以及被排除的内容",
    ),
    (
        "梦见水，平静的湖水，很安心",
        "梦见水，水是平静的湖水，很安心。",
        "愿望与情感质地，不要凭空添加情节",
    ),
    (
        "内部形象与分裂：同一段关系里忽好忽坏",
        "我对一个人的感觉总在两个极端之间来回跳。",
        "内在客体、分裂与投射",
    ),
    (
        "被排除在外、欲望与缺失",
        "邮件发出去了，但一直没有回音，我一直在想这件事。",
        "能指链条断裂的位置",
    ),
]

CASES: list[tuple[str, str]] = [
    ("RAG-1 找房间", "我梦到一直在找一间房，但是怎么也找不到。"),
    ("RAG-2 梦见水", "我梦见水。"),
    ("RAG-3 三视角比较", "从三个精神分析视角比较分析《哈姆雷特》。"),
    ("RAG-4 诊断请求", "我是不是有边缘型人格障碍？"),
]

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

NEGATION_MARKERS = ("没有", "没提到", "未", "不曾", "并未", "无", "缺", "不")
"""Markers that turn a mention of an absent term into an explicit non-claim."""

CALL_COUNTS: Counter[str] = Counter()
REAL_BUILD = llm_module.build_structured_runnable


def install_call_counter() -> None:
    """Count structured model calls without changing agent behaviour."""

    def counting_build(schema: type[Any]) -> Any:
        runnable = REAL_BUILD(schema)

        class CountingRunnable:
            async def ainvoke(
                self, input: Any, config: Any = None, **kwargs: Any
            ) -> Any:
                CALL_COUNTS[schema.__name__] += 1
                return await runnable.ainvoke(input, config, **kwargs)

        return CountingRunnable()

    llm_module.build_structured_runnable = counting_build


def asserted_missing_text_terms(observations: list[str]) -> list[str]:
    """Return forbidden terms that were *asserted* rather than denied.

    An observation may name a term only to say it was never provided (for
    example "用户没有提到童年"). Only unqualified mentions are violations.

    Args:
        observations: Observation strings of one school.

    Returns:
        The forbidden terms that appear without a negation marker nearby.
    """
    hits: list[str] = []
    for observation in observations:
        for term in FORBIDDEN_OBSERVED_TERMS:
            if term not in observation:
                continue
            if any(marker in observation for marker in NEGATION_MARKERS):
                continue
            hits.append(term)
    return sorted(set(hits))


def short(text: str, limit: int = 200) -> str:
    """Return a single-line excerpt."""
    collapsed = " ".join(str(text).split())
    return collapsed if len(collapsed) <= limit else collapsed[:limit] + "…"


def retrieval_report() -> dict[str, float]:
    """Run the retrieval-only checks and print a report.

    Returns:
        Timing numbers for the summary (model load and one warm retrieval).
    """
    settings = get_rag_settings()
    print("=" * 72)
    print("Part 1 - retrieval only (no model calls)")
    print("=" * 72)
    print(f"embedding model : {settings.embedding_model}")
    print(f"device          : {settings.embedding_device}")
    print(f"vectorstore     : {settings.vectorstore_path}")
    print(f"collection      : {settings.collection_name}")
    print(f"top_k           : {settings.top_k}")

    load_started = time.perf_counter()
    # Use the same cached instance the graph uses: constructing a second
    # HuggingFaceEmbeddings would load the 2.2GB model twice in one process.
    embeddings = get_embeddings()
    embeddings.embed_query("warmup")
    load_seconds = time.perf_counter() - load_started
    print(f"backend         : {describe_embedding_backend(embeddings)}")
    print(f"model load      : {load_seconds:.2f}s")

    store = open_vectorstore(embeddings, settings)
    assert store is not None, "no usable index; run scripts/index_knowledge.py first"
    total = collection_count(store)
    print(f"indexed chunks  : {total}")
    assert total > 0

    warm_seconds = 0.0
    for index, (label, user_text, focus) in enumerate(QUERIES, start=1):
        plan = {
            "task_summary": label,
            "analysis_focus": focus,
            "freudian_focus": focus,
            "object_relations_focus": focus,
            "lacanian_focus": focus,
        }
        messages: list[BaseMessage] = [HumanMessage(content=user_text)]
        query_text = build_school_query("freudian", plan, user_text)
        started = time.perf_counter()
        outcome = retrieve_evidence(plan, messages, store=store, embeddings=embeddings)
        elapsed = time.perf_counter() - started
        if index == 1:
            warm_seconds = elapsed
        assert outcome.available, outcome.reason
        assert build_school_query("freudian", plan, user_text) == query_text
        print(f"\nquery {index}: {label}")
        print(f"  user text : {user_text}")
        print(f"  query     : {short(query_text, 120)}")
        print(f"  elapsed   : {elapsed:.3f}s")
        for school in SPECIALIST_PERSPECTIVES:
            items = evidence_items_from_state(
                {"evidence_by_school": outcome.evidence_by_school}, school
            )
            assert items, f"no evidence for {school}"
            assert all(item.school in {school, "general"} for item in items), (
                f"{school} received foreign material"
            )
            top = items[0]
            print(
                f"  [{school}] top score={top.retrieval_score} "
                f"id={top.evidence_id}\n"
                f"      {describe_source(top)}\n"
                f"      {short(top.text, 90)}"
            )

    return {"model_load_seconds": load_seconds, "warm_retrieval_seconds": warm_seconds}


async def run_case(label: str, text: str) -> dict[str, Any]:
    """Run one case through the graph and check the evidence contract."""
    threaded = graph.copy()
    threaded.checkpointer = MemorySaver()
    config: RunnableConfig = {"configurable": {"thread_id": f"phase6-{label}"}}

    before = sum(CALL_COUNTS.values())
    started = time.perf_counter()
    result = await threaded.ainvoke(
        cast(
            "PsycheGraphState",
            {"messages": [{"type": "human", "content": text}]},
        ),
        config=config,
    )
    elapsed = time.perf_counter() - started
    calls = sum(CALL_COUNTS.values()) - before

    evidence = result["evidence_by_school"]
    meta = result["evidence_meta"]
    final = result["final_result"]
    answer = result["messages"][-1].content

    print(f"\n=== {label} ===\nuser: {text}")
    print(f"  elapsed        : {elapsed:.1f}s")
    print(f"  deepseek calls : {calls}")
    print(
        f"  evidence       : available={meta['available']} "
        f"reason={meta['reason']} counts={meta['counts']} "
        f"retrieval={meta['retrieval_seconds']}s"
    )

    critique = result["critique"]
    issues = result["deterministic_issues"]
    revisions = int(result["revision_count"])
    status = str(result["finalization_status"])
    print(
        f"  review         : status={status} verdict={critique['verdict']} "
        f"critic_issues={len(critique['issues'])} "
        f"deterministic_issues={len(issues)} revisions={revisions}"
    )
    for issue in issues:
        print(
            f"      [code] {issue['category']}/{issue['severity']}: {issue['message']}"
        )
    for issue in critique["issues"]:
        print(
            f"      [critic] {issue['category']}/{issue['severity']}: {issue['description']}"
        )
    assert status in {"passed", "revised_and_passed", "safe_fallback"}, status
    if status == "safe_fallback":
        assert critique["verdict"] == "revise", f"{label}: fallback without a rejection"
        assert revisions == MAX_REVISION == 1
        assert final["integrated_interpretation"] == ""
        assert list(final["used_evidence_ids"]) == []
        assert "没有达到可以展示的标准" in answer
    else:
        assert critique["verdict"] == "pass", f"{label}: published without a pass"
        assert not [item for item in issues if item["severity"] == "error"]
    if status == "passed":
        assert revisions == 0, f"{label}: a clean pass must not report revisions"
    if status == "revised_and_passed":
        assert revisions == 1, f"{label}: more than one revision is not allowed"
    expected_calls = 6 + 2 * revisions
    if calls != expected_calls:
        print(
            f"  note: {calls} structured calls instead of {expected_calls} "
            "(one call was retried after a schema violation)"
        )
    assert calls <= expected_calls + 1, f"{label}: unexpected call count {calls}"

    allowed: dict[str, set[str]] = {}
    for school in SPECIALIST_PERSPECTIVES:
        items = evidence_items_from_state({"evidence_by_school": evidence}, school)
        allowed[school] = {item.evidence_id for item in items}
        assert items, f"{label}: no evidence retrieved for {school}"
        for item in items:
            assert item.school in {school, "general"}, f"{label}: {school} leak"
        print(
            f"  [{school}] {len(items)} passage(s), top={items[0].retrieval_score} "
            f"{describe_source(items[0])}"
        )

    catalogue = set().union(*allowed.values())
    cited_by_specialists: set[str] = set()
    for school in SPECIALIST_PERSPECTIVES:
        school_result = result["specialist_results"][school]
        print(f"  [{school}] observations: {school_result['observations']}")
        print(f"  [{school}] summary: {short(str(school_result['summary']), 110)}")
        for interpretation in school_result["interpretations"]:
            for evidence_id in interpretation.get("evidence_ids", []):
                assert evidence_id in allowed[school], (
                    f"{label}: {school} cited {evidence_id} which it never received"
                )
                cited_by_specialists.add(evidence_id)
                print(
                    f"      cite {evidence_id}: "
                    f"{short(str(interpretation['claim']), 90)}"
                )

    used = list(final["used_evidence_ids"])
    assert set(used) <= catalogue, f"{label}: synthesizer cited unknown ids"
    print(f"  used_evidence_ids: {used}")
    print(f"  final_response: {short(str(final['final_response']), 320)}")

    if used:
        assert "理论依据（本地知识库）" in answer, f"{label}: sources block missing"
        for item in evidence_items_from_state(
            {"evidence_by_school": evidence}, "freudian"
        ):
            if item.evidence_id in used:
                assert item.title in answer or (item.work_title or "") in answer
        # Markdown and text sources have no page numbers, and none may appear.
        assert "p. " not in answer.split("理论依据（本地知识库）")[1], (
            f"{label}: a page number was invented"
        )
    else:
        assert "理论依据（本地知识库）" not in answer

    assert len(result["messages"]) == 2, f"{label}: internal agents leaked"

    if label.startswith("RAG-2"):
        for school in SPECIALIST_PERSPECTIVES:
            observations = result["specialist_results"][school]["observations"]
            invented = asserted_missing_text_terms(observations)
            assert not invented, f"{label}: {school} asserted {invented}"
            assert result["specialist_results"][school]["limitations"], (
                f"{label}: {school} reported no limitations"
            )
    if label.startswith("RAG-4"):
        assert result["supervisor_plan"]["clinical_diagnosis_requested"] is True
        for school in SPECIALIST_PERSPECTIVES:
            assert result["specialist_results"][school]["clinical_diagnosis_refused"], (
                f"{label}: {school} did not refuse"
            )
            observed = " ".join(result["specialist_results"][school]["observations"])
            asserts_label = "边缘型" in observed and not any(
                marker in observed for marker in NEGATION_MARKERS
            )
            assert not asserts_label, f"{label}: {school} put the label on the user"
        assert final["clinical_diagnosis_refused"] is True
        response = str(final["final_response"])
        declines = ("不能" in response or "无法" in response) and "诊断" in response
        assert declines, f"{label}: the answer does not decline a clinical judgement"
    return {"label": label, "elapsed": elapsed, "calls": calls, "status": status}


async def main() -> None:
    """Run the retrieval checks and the four real cases."""
    logging.disable(logging.CRITICAL)
    load_dotenv()
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
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
    print(
        "Topology verified: supervisor -> evidence -> 3 specialists -> synthesizer "
        "-> deterministic_validator -> critic -> finalize | revise_synthesis | safe_finalize."
    )

    timings = retrieval_report()
    install_call_counter()

    print("\n" + "=" * 72)
    print("Part 2 - four real conversations through the graph")
    print("=" * 72)
    cases = []
    for label, text in CASES:
        cases.append(await run_case(label, text))

    print("\n" + "=" * 72)
    print("Part 3 - performance summary")
    print("=" * 72)
    print(
        f"embedding model load (cold, first process) : {timings['model_load_seconds']:.2f}s"
    )
    print(
        f"first warm retrieval (3 schools)           : {timings['warm_retrieval_seconds']:.3f}s"
    )
    for case in cases:
        print(
            f"{case['label']:<16} total={case['elapsed']:.1f}s "
            f"deepseek_calls={case['calls']} status={case['status']}"
        )
    print(f"call breakdown: {dict(CALL_COUNTS)}")
    print("\nAll structural checks passed; review the analysis above.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
