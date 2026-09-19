"""Orchestration tests: the real graph, with only the chat model replaced."""

import asyncio
from typing import Any, cast

import pytest
from langgraph.checkpoint.memory import MemorySaver

from react_agent.agents.context import InvalidEvidenceReferenceError
from react_agent.config import MAX_REVISION
from react_agent.graph import graph, route_after_critic
from react_agent.llm import STRUCTURED_OUTPUT_METHOD
from react_agent.schemas import (
    CritiqueResult,
    SchoolAnalysis,
    SupervisorPlan,
    SynthesisResult,
)
from react_agent.state import PsycheGraphState
from tests.unit_tests.conftest import (
    FakeChatModel,
    RagIndex,
    default_payload_factory,
    make_critic_issue,
    make_critique,
    make_plan,
    make_school_analysis,
    make_synthesis,
)

SPECIALISTS = ("freudian", "object_relations", "lacanian")


def _threaded_graph() -> Any:
    """Return a copy of the compiled graph that keeps thread state."""
    threaded = graph.copy()
    threaded.checkpointer = MemorySaver()
    return threaded


def _run(fake: FakeChatModel, content: str, thread_id: str = "t1") -> dict[str, Any]:
    threaded = _threaded_graph()
    config = {"configurable": {"thread_id": thread_id}}
    return asyncio.run(
        threaded.ainvoke(
            {"messages": [{"type": "human", "content": content}]}, config=config
        )
    )


def test_graph_is_compiled_and_exposes_the_review_pipeline() -> None:
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


def test_full_run_costs_six_model_calls(fake_model: FakeChatModel) -> None:
    result = _run(fake_model, "我梦见水。")

    names = fake_model.call_names()
    assert names[0] == "SupervisorPlan"
    assert sorted(names[1:4]) == ["SchoolAnalysis"] * 3
    assert names[4] == "SynthesisResult"
    assert names[5] == "CritiqueResult"
    assert len(names) == 6

    assert isinstance(result["supervisor_plan"], dict)
    assert set(result["specialist_results"]) == set(SPECIALISTS)
    assert isinstance(result["draft_result"], dict)
    assert isinstance(result["critique"], dict)
    assert isinstance(result["final_result"], dict)
    assert result["finalization_status"] == "passed"
    assert result["revision_count"] == 0
    assert result["final_result"]["final_response"] in [
        message.content for message in result["messages"] if message.type == "ai"
    ]


def test_each_specialist_is_an_independent_model_call(
    fake_model: FakeChatModel,
) -> None:
    _run(fake_model, "我梦见水。")

    prompts = fake_model.calls_for(SchoolAnalysis)
    assert len(prompts) == 3
    first_messages = [prompt[0].content for prompt in prompts]
    assert len(set(first_messages)) == 3, "specialists shared a system prompt"


def test_specialists_run_concurrently(fake_model: FakeChatModel) -> None:
    fake_model.delay = 0.5
    _run(fake_model, "我梦见水。")

    starts = fake_model.times(SchoolAnalysis, "start")
    finishes = fake_model.times(SchoolAnalysis, "finish")
    assert len(starts) == 3 and len(finishes) == 3

    # Real concurrency: the last specialist call starts before the first ends,
    # and the whole group spans barely more than one model call.
    assert max(starts) < min(finishes), "specialist calls did not overlap"
    assert max(finishes) - min(starts) < 0.9, "three 0.5s calls did not overlap"


def test_synthesizer_waits_for_all_three_specialists(
    fake_model: FakeChatModel,
) -> None:
    fake_model.delay = 0.2
    result = _run(fake_model, "我梦见水。")

    assert all(school in result["specialist_results"] for school in SPECIALISTS)
    synthesizer_start = fake_model.times(SynthesisResult, "start")[0]
    specialist_finishes = fake_model.times(SchoolAnalysis, "finish")
    specialist_starts = fake_model.times(SchoolAnalysis, "start")
    assert len(specialist_finishes) == 3
    assert max(specialist_finishes) <= synthesizer_start
    assert fake_model.times(SupervisorPlan, "finish")[0] <= min(specialist_starts)


def test_only_one_ai_message_per_run(fake_model: FakeChatModel) -> None:
    result = _run(fake_model, "我梦见水。")

    roles = [message.type for message in result["messages"]]
    assert roles == ["human", "ai"]
    for school in SPECIALISTS:
        assert school not in result["messages"][1].content
    assert "observations" not in result["messages"][1].content


def test_second_turn_regenerates_working_fields(fake_model: FakeChatModel) -> None:
    threaded = _threaded_graph()
    config = {"configurable": {"thread_id": "multi-turn"}}

    first = asyncio.run(
        threaded.ainvoke(
            {"messages": [{"type": "human", "content": "我梦见水。"}]}, config=config
        )
    )
    assert first["specialist_results"]["freudian"]["observations"] == ["用户梦见水"]

    fake_model.overrides = {
        SupervisorPlan: make_plan(task_summary="第二轮：用户补充了湖水与安心。"),
        SchoolAnalysis: make_school_analysis(
            "freudian", observations=["用户补充了平静的湖水和安心"]
        ),
        SynthesisResult: make_synthesis(final_response="第二轮回答。"),
    }
    second = asyncio.run(
        threaded.ainvoke(
            {"messages": [{"type": "human", "content": "水是平静的湖水，我很安心。"}]},
            config=config,
        )
    )

    assert [message.type for message in second["messages"]] == [
        "human",
        "ai",
        "human",
        "ai",
    ]
    assert second["final_result"]["final_response"] == "第二轮回答。"
    for school in SPECIALISTS:
        assert second["specialist_results"][school]["observations"] == [
            "用户补充了平静的湖水和安心"
        ]
    assert second["supervisor_plan"]["task_summary"].startswith("第二轮")

    supervisor_prompts = fake_model.calls_for(SupervisorPlan)
    assert len(supervisor_prompts) == 2
    second_prompt = [message.content for message in supervisor_prompts[1]]
    assert second_prompt[-1] == "水是平静的湖水，我很安心。"


def test_checkpoint_restore_has_no_custom_pydantic_types(
    fake_model: FakeChatModel,
) -> None:
    threaded = _threaded_graph()
    config = {"configurable": {"thread_id": "restore"}}
    asyncio.run(
        threaded.ainvoke(
            {"messages": [{"type": "human", "content": "我梦见水。"}]}, config=config
        )
    )

    snapshot = asyncio.run(threaded.aget_state(config))
    values = cast("PsycheGraphState", snapshot.values)
    assert isinstance(values["supervisor_plan"], dict)
    assert isinstance(values["final_result"], dict)
    for school in SPECIALISTS:
        entry = values["specialist_results"][school]
        assert not hasattr(entry, "model_dump"), (
            "a Pydantic object leaked into the state"
        )
        assert type(entry) is dict

    checkpoints = (
        list(
            threaded.checkpointer.list(config)  # type: ignore[union-attr]
        )
        if threaded.checkpointer is not None
        else []
    )
    assert checkpoints, "no checkpoint was written"
    blob = "".join(repr(checkpoint.checkpoint) for checkpoint in checkpoints)
    assert "PsychoanalyticAnalysis" not in blob
    assert "SchoolAnalysis(" not in blob
    assert "SynthesisResult(" not in blob
    assert "Surrogate" not in blob and "pydantic" not in blob.lower()


def test_state_input_contract_is_the_chat_thread(fake_model: FakeChatModel) -> None:
    result = _run(fake_model, "我梦见水。")

    # The web UI only ever sends messages; everything else is graph output.
    assert set(result) >= {
        "messages",
        "supervisor_plan",
        "specialist_results",
        "draft_result",
        "deterministic_issues",
        "critique",
        "revision_count",
        "final_result",
        "finalization_status",
    }
    assert STRUCTURED_OUTPUT_METHOD == "function_calling"


def test_graph_handles_a_clinical_diagnosis_request(fake_model: FakeChatModel) -> None:
    fake_model.overrides = {
        SupervisorPlan: make_plan(clinical_diagnosis_requested=True),
        SchoolAnalysis: make_school_analysis(
            "freudian", clinical_diagnosis_refused=True
        ),
        SynthesisResult: make_synthesis(
            clinical_diagnosis_refused=True,
            final_response="本系统只能做精神分析理论层面的文本讨论，不能进行临床诊断。",
        ),
    }
    result = _run(fake_model, "我是不是有边缘型人格障碍？")

    assert result["supervisor_plan"]["clinical_diagnosis_requested"] is True
    assert result["final_result"]["clinical_diagnosis_refused"] is True
    for school in SPECIALISTS:
        assert (
            result["specialist_results"][school]["clinical_diagnosis_refused"] is True
        )
    assert result["critique"]["clinical_safety_ok"] is True
    assert result["finalization_status"] == "passed"
    assert result["deterministic_issues"] == []


def test_payload_factory_covers_every_schema() -> None:
    assert default_payload_factory(SupervisorPlan)["task_summary"]
    with pytest.raises(AssertionError):
        default_payload_factory(int)


def test_evidence_node_adds_no_model_call(
    fake_model: FakeChatModel, rag_index: RagIndex
) -> None:
    """Retrieval is local: the call count stays at six calls per turn."""
    result = _run(fake_model, "我梦见水，水是平静的湖水。")

    assert len(fake_model.calls) == 6
    assert fake_model.call_names().count("SchoolAnalysis") == 3
    for school in SPECIALISTS:
        assert result["evidence_by_school"][school], f"no evidence for {school}"
    assert result["evidence_meta"]["available"] is True
    assert result["evidence_meta"]["counts"] == {
        school: len(result["evidence_by_school"][school]) for school in SPECIALISTS
    }


def test_retrieved_evidence_is_school_scoped(
    fake_model: FakeChatModel, rag_index: RagIndex
) -> None:
    result = _run(fake_model, "我梦见水，水是平静的湖水。")

    prompts = fake_model.calls_for(SchoolAnalysis)
    assert len(prompts) == 3
    # The specialists run concurrently, so prompts are matched to schools by
    # their system prompt rather than by call order.
    by_school: dict[str, str] = {}
    for prompt in prompts:
        content = "\n".join(str(message.content) for message in prompt)
        for school in SPECIALISTS:
            if {
                "freudian": "Freudian Specialist",
                "object_relations": "Object Relations Specialist",
                "lacanian": "Lacanian Specialist",
            }[school] in content:
                by_school[school] = content
    assert set(by_school) == set(SPECIALISTS)

    chunks = {chunk.metadata["chunk_id"]: chunk for chunk in rag_index.chunks}
    for school, content in by_school.items():
        own_ids = {
            entry["evidence_id"] for entry in result["evidence_by_school"][school]
        }
        assert own_ids, f"no evidence retrieved for {school}"
        for evidence_id in own_ids:
            assert chunks[evidence_id].metadata["school"] in {school, "general"}
            assert evidence_id in content, "prompt is missing its own evidence"
        for other_school in set(SPECIALISTS) - {school}:
            foreign = {
                entry["evidence_id"]
                for entry in result["evidence_by_school"][other_school]
            }
            assert foreign, "the fixture corpus is missing a school"
            leaked = [evidence_id for evidence_id in foreign if evidence_id in content]
            assert not leaked, f"{school} prompt leaked {leaked}"


def test_run_without_a_vectorstore_still_completes(fake_model: FakeChatModel) -> None:
    """No index: the run must degrade, not fail."""
    result = _run(fake_model, "我梦见水。")

    assert len(fake_model.calls) == 6
    assert result["evidence_by_school"] == {school: [] for school in SPECIALISTS}
    assert result["evidence_meta"]["available"] is False
    assert result["evidence_meta"]["reason"] == "disabled"
    assert result["final_result"]["final_response"]


def test_final_message_cites_the_sources_it_used(
    fake_model: FakeChatModel, rag_index: RagIndex
) -> None:
    freudian_ids = [
        chunk.metadata["chunk_id"] for chunk in rag_index.evidence_for("freudian")
    ]
    fake_model.overrides = {
        SynthesisResult: make_synthesis(used_evidence_ids=freudian_ids[:1]),
    }
    result = _run(fake_model, "我梦见水，水是平静的湖水。", thread_id="t-cite")

    answer = result["messages"][-1].content
    assert result["final_result"]["used_evidence_ids"] == freudian_ids[:1]
    assert "理论依据（本地知识库）" in answer
    cited = [
        chunk
        for chunk in rag_index.chunks
        if chunk.metadata["chunk_id"] == freudian_ids[0]
    ]
    assert cited[0].metadata["work_title"] in answer
    assert cited[0].metadata["author"] in answer


def test_uncited_evidence_is_not_listed(
    fake_model: FakeChatModel, rag_index: RagIndex
) -> None:
    result = _run(fake_model, "我梦见水，水是平静的湖水。")

    answer = result["messages"][-1].content
    assert "理论依据（本地知识库）" not in answer
    assert result["final_result"]["used_evidence_ids"] == []


def test_hallucinated_evidence_id_fails_the_run(
    fake_model: FakeChatModel, rag_index: RagIndex
) -> None:
    fake_model.overrides = {
        SynthesisResult: make_synthesis(used_evidence_ids=["made_up_000001"]),
    }
    with pytest.raises(InvalidEvidenceReferenceError, match="made_up_000001"):
        _run(fake_model, "我梦见水。", thread_id="t-hallucination")


def test_evidence_in_the_checkpoint_is_plain_json(
    fake_model: FakeChatModel, rag_index: RagIndex
) -> None:
    threaded = _threaded_graph()
    config = {"configurable": {"thread_id": "evidence-restore"}}
    asyncio.run(
        threaded.ainvoke(
            {"messages": [{"type": "human", "content": "我梦见水。"}]}, config=config
        )
    )

    snapshot = asyncio.run(threaded.aget_state(config))
    values = cast("PsycheGraphState", snapshot.values)
    for school in SPECIALISTS:
        entries = values["evidence_by_school"][school]
        assert entries, f"no evidence stored for {school}"
        for entry in entries:
            assert type(entry) is dict
            assert isinstance(entry["evidence_id"], str)
    blob = "".join(
        repr(checkpoint.checkpoint)
        for checkpoint in threaded.checkpointer.list(config)  # type: ignore[union-attr]
    )
    assert "EvidenceItem(" not in blob


def test_route_after_critic_covers_every_branch() -> None:
    assert (
        route_after_critic({"critique": {"verdict": "pass"}, "revision_count": 0})
        == "finalize"
    )
    assert (
        route_after_critic({"critique": {"verdict": "revise"}, "revision_count": 0})
        == "revise_synthesis"
    )
    assert (
        route_after_critic({"critique": {"verdict": "revise"}, "revision_count": 1})
        == "safe_finalize"
    )
    # A missing critique or counter must still terminate, never loop.
    assert route_after_critic({}) == "revise_synthesis"
    assert route_after_critic(
        {"critique": {"verdict": "revise"}, "revision_count": 5}
    ) == ("safe_finalize")
    assert MAX_REVISION == 1


def test_revise_then_pass_is_recorded_as_revised(
    fake_model: FakeChatModel,
) -> None:
    fake_model.sequences = {
        CritiqueResult: [
            make_critique(
                verdict="revise",
                issues=[make_critic_issue()],
                revision_instructions=["把该结论降级为条件性表述或删除。"],
                evidence_grounding_ok=False,
            ),
            make_critique(summary="修订后不再有过度推断。"),
        ]
    }
    result = _run(fake_model, "我梦见水，水是平静的湖水。", thread_id="t-revise")

    assert result["revision_count"] == 1
    assert result["finalization_status"] == "revised_and_passed"
    assert result["critique"]["verdict"] == "pass"
    assert len(fake_model.calls_for(SynthesisResult)) == 2
    assert len(fake_model.calls_for(CritiqueResult)) == 2
    assert len(fake_model.calls) == 8
    assert [message.type for message in result["messages"]] == ["human", "ai"]
    assert result["final_result"]["final_response"] in result["messages"][-1].content


def test_revision_prompt_contains_the_critique(
    fake_model: FakeChatModel,
) -> None:
    fake_model.sequences = {
        CritiqueResult: [
            make_critique(
                verdict="revise",
                issues=[
                    make_critic_issue(
                        description="DRAFT-CLAIM-UNDER-REVIEW",
                        revision_instruction="INSTRUCTION-TO-FIX",
                    )
                ],
                revision_instructions=["INSTRUCTION-TO-FIX"],
            ),
            make_critique(),
        ]
    }
    _run(fake_model, "我梦见水。", thread_id="t-revise-prompt")

    revision_prompt = fake_model.calls_for(SynthesisResult)[1]
    joined = "\n".join(str(message.content) for message in revision_prompt)
    assert "DRAFT-CLAIM-UNDER-REVIEW" in joined
    assert "INSTRUCTION-TO-FIX" in joined
    assert "只修复 Critic 指出的问题" in joined


def test_second_revise_verdict_uses_the_safe_fallback(
    fake_model: FakeChatModel,
) -> None:
    fake_model.sequences = {
        CritiqueResult: [
            make_critique(
                verdict="revise",
                issues=[make_critic_issue()],
                revision_instructions=["降级或删除该结论。"],
                evidence_grounding_ok=False,
            )
        ]
    }
    result = _run(fake_model, "我梦见水。", thread_id="t-safe")

    assert result["revision_count"] == 1
    assert result["finalization_status"] == "safe_fallback"
    assert result["critique"]["verdict"] == "revise"
    assert len(fake_model.calls) == 8, "the loop must stop after one revision"
    assert result["final_result"]["integrated_interpretation"] == ""
    assert result["final_result"]["common_ground"] == []
    assert result["final_result"]["used_evidence_ids"] == []
    answer = result["messages"][-1].content
    assert "没有达到可以展示的标准" in answer
    assert "splitting" not in answer
    assert [message.type for message in result["messages"]] == ["human", "ai"]


def test_draft_and_critique_never_enter_messages(fake_model: FakeChatModel) -> None:
    draft_texts = ["第一版草稿内容", "修订后的第二版内容"]
    fake_model.sequences = {
        SynthesisResult: [
            make_synthesis(final_response=draft_texts[0]),
            make_synthesis(final_response=draft_texts[1]),
        ],
        CritiqueResult: [
            make_critique(
                verdict="revise",
                summary="CRITIC-SUMMARY-MUST-NOT-LEAK",
                issues=[make_critic_issue()],
            ),
            make_critique(summary="SECOND-CRITIC-SUMMARY"),
        ],
    }
    result = _run(fake_model, "我梦见水。", thread_id="t-leak")

    transcript = "\n".join(str(message.content) for message in result["messages"])
    assert draft_texts[0] not in transcript, "the rejected draft leaked into the chat"
    assert "CRITIC-SUMMARY-MUST-NOT-LEAK" not in transcript
    assert "SECOND-CRITIC-SUMMARY" not in transcript
    # Only the accepted revision is shown.
    assert draft_texts[1] in transcript
    assert len(result["messages"]) == 2


def test_deterministic_error_forces_a_revision_even_when_the_critic_passes(
    fake_model: FakeChatModel,
) -> None:
    """A dangling [E1] label is a code-level error: `pass` cannot override it."""
    fake_model.overrides = {
        SynthesisResult: make_synthesis(
            final_response="这条读解引用 [E1] 作为依据。",
            used_evidence_ids=[],
        )
    }
    fake_model.sequences = {CritiqueResult: [make_critique(verdict="pass")]}
    result = _run(fake_model, "我梦见水。", thread_id="t-forced")

    categories = {issue["category"] for issue in result["critique"]["issues"]}
    assert "synthesis_quality" in categories
    assert result["critique"]["verdict"] == "revise"
    assert result["finalization_status"] == "safe_fallback"
    assert result["revision_count"] == 1


def test_revision_counter_resets_on_the_next_turn(fake_model: FakeChatModel) -> None:
    """Turn 2 must not inherit turn 1's revision count."""
    fake_model.sequences = {
        CritiqueResult: [
            make_critique(verdict="revise", issues=[make_critic_issue()]),
            make_critique(),
        ]
    }
    threaded = _threaded_graph()
    config = {"configurable": {"thread_id": "reset-count"}}

    first = asyncio.run(
        threaded.ainvoke(
            {"messages": [{"type": "human", "content": "我梦见水。"}]}, config=config
        )
    )
    assert first["revision_count"] == 1
    assert first["finalization_status"] == "revised_and_passed"

    second = asyncio.run(
        threaded.ainvoke(
            {"messages": [{"type": "human", "content": "水是平静的湖水。"}]},
            config=config,
        )
    )
    assert second["revision_count"] == 0
    assert second["finalization_status"] == "passed"
    assert [message.type for message in second["messages"]] == [
        "human",
        "ai",
        "human",
        "ai",
    ]


def test_review_trail_is_stored_as_json(fake_model: FakeChatModel) -> None:
    fake_model.sequences = {
        CritiqueResult: [
            make_critique(verdict="revise", issues=[make_critic_issue()]),
            make_critique(),
        ]
    }
    threaded = _threaded_graph()
    config = {"configurable": {"thread_id": "review-trail"}}
    asyncio.run(
        threaded.ainvoke(
            {"messages": [{"type": "human", "content": "我梦见水。"}]}, config=config
        )
    )

    snapshot = asyncio.run(threaded.aget_state(config))
    values = cast("PsycheGraphState", snapshot.values)
    assert type(values["draft_result"]) is dict
    assert type(values["critique"]) is dict
    assert values["revision_count"] == 1
    assert values["finalization_status"] == "revised_and_passed"
    blob = "".join(
        repr(checkpoint.checkpoint)
        for checkpoint in threaded.checkpointer.list(config)  # type: ignore[union-attr]
    )
    assert "CritiqueResult(" not in blob
    assert "CriticIssue(" not in blob
    assert "DeterministicIssue(" not in blob
