"""Tests for the individual agent nodes, with only the model replaced."""

import asyncio
from typing import Any, cast

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

import react_agent.llm as llm_module
from react_agent.agents.context import (
    InvalidEvidenceReferenceError,
    PerspectiveMismatchError,
    cited_evidence_ids,
    collect_invalid_evidence_ids,
    specialist_results_message,
    supervisor_plan_message,
    validate_evidence_references,
)
from react_agent.agents.freudian import FREUDIAN_SYSTEM_PROMPT, freudian_node
from react_agent.agents.lacanian import LACANIAN_SYSTEM_PROMPT, lacanian_node
from react_agent.agents.object_relations import (
    OBJECT_RELATIONS_SYSTEM_PROMPT,
    object_relations_node,
)
from react_agent.agents.supervisor import SUPERVISOR_SYSTEM_PROMPT, supervisor_node
from react_agent.agents.synthesizer import SYNTHESIZER_SYSTEM_PROMPT, synthesizer_node
from react_agent.schemas import SchoolAnalysis, SupervisorPlan, SynthesisResult
from tests.unit_tests.conftest import (
    FakeChatModel,
    make_evidence_by_school,
    make_plan,
    make_school_analysis,
    make_synthesis,
)


def _state_with_plan(**extra: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "messages": [HumanMessage(content="我梦见水。")],
        "supervisor_plan": make_plan(),
    }
    state.update(extra)
    return state


def _three_results() -> dict[str, Any]:
    return {
        perspective: make_school_analysis(perspective)
        for perspective in ("freudian", "object_relations", "lacanian")
    }


def test_supervisor_node_stores_plan_without_writing_messages(
    fake_model: FakeChatModel,
) -> None:
    update = asyncio.run(
        supervisor_node({"messages": [HumanMessage(content="我梦见水。")]})
    )

    assert set(update) == {"supervisor_plan", "revision_count"}
    assert update["revision_count"] == 0
    plan = cast("dict[str, Any]", update["supervisor_plan"])
    assert plan["task_summary"]
    assert plan["clinical_diagnosis_requested"] is False

    prompts = fake_model.calls_for(SupervisorPlan)
    assert len(prompts) == 1
    assert isinstance(prompts[0][0], SystemMessage)
    assert prompts[0][0].content == SUPERVISOR_SYSTEM_PROMPT
    assert [message.content for message in prompts[0][1:]] == ["我梦见水。"]


@pytest.mark.parametrize(
    ("node", "prompt_text", "school"),
    [
        (freudian_node, FREUDIAN_SYSTEM_PROMPT, "freudian"),
        (object_relations_node, OBJECT_RELATIONS_SYSTEM_PROMPT, "object_relations"),
        (lacanian_node, LACANIAN_SYSTEM_PROMPT, "lacanian"),
    ],
)
def test_specialist_node_writes_only_its_own_school(
    fake_model: FakeChatModel, node: Any, prompt_text: str, school: str
) -> None:
    update = asyncio.run(node(_state_with_plan()))

    assert set(update) == {"specialist_results"}
    results = cast("dict[str, dict[str, Any]]", update["specialist_results"])
    assert set(results) == {school}
    assert results[school]["perspective"] == school
    assert "messages" not in update


def test_specialist_prompt_carries_the_supervisor_focus(
    fake_model: FakeChatModel,
) -> None:
    asyncio.run(freudian_node(_state_with_plan()))

    prompt = fake_model.calls_for(SchoolAnalysis)[0]
    assert prompt[0].content == FREUDIAN_SYSTEM_PROMPT
    plan_message = prompt[1].content
    assert make_plan()["freudian_focus"] in plan_message
    assert make_plan()["analysis_focus"] in plan_message
    assert "不是关于用户的事实" in plan_message
    assert prompt[-1].content == "我梦见水。"


def test_specialists_use_their_own_prompt(fake_model: FakeChatModel) -> None:
    asyncio.run(freudian_node(_state_with_plan()))
    asyncio.run(lacanian_node(_state_with_plan()))

    prompts = fake_model.calls_for(SchoolAnalysis)
    assert prompts[0][0].content == FREUDIAN_SYSTEM_PROMPT
    assert prompts[1][0].content == LACANIAN_SYSTEM_PROMPT
    assert prompts[0][0].content != prompts[1][0].content


def test_specialist_rejects_another_school_from_the_model(
    fake_model: FakeChatModel,
) -> None:
    fake_model.raw = True
    fake_model.overrides = {
        SchoolAnalysis: make_school_analysis("lacanian"),
    }
    with pytest.raises(PerspectiveMismatchError):
        asyncio.run(freudian_node(_state_with_plan()))


def test_specialist_requires_the_supervisor_plan(fake_model: FakeChatModel) -> None:
    with pytest.raises(ValueError, match="supervisor_plan"):
        asyncio.run(freudian_node({"messages": [HumanMessage(content="我梦见水。")]}))


def test_synthesizer_writes_only_the_draft(fake_model: FakeChatModel) -> None:
    """An unreviewed draft must never become a chat message."""
    state = _state_with_plan(specialist_results=_three_results())
    update = asyncio.run(synthesizer_node(state))

    assert set(update) == {"draft_result"}
    draft = cast("dict[str, Any]", update["draft_result"])
    assert draft["final_response"] == make_synthesis()["final_response"]
    assert "messages" not in update


def test_synthesizer_prompt_contains_all_three_schools(
    fake_model: FakeChatModel,
) -> None:
    state = _state_with_plan(specialist_results=_three_results())
    asyncio.run(synthesizer_node(state))

    prompt = fake_model.calls_for(SynthesisResult)[0]
    assert prompt[0].content == SYNTHESIZER_SYSTEM_PROMPT
    joined = "\n".join(str(message.content) for message in prompt)
    for school in ("freudian", "object_relations", "lacanian"):
        assert school in joined
        assert f"{school} 的结论。" in joined
    assert make_plan()["synthesis_goal"] in joined
    assert "我梦见水。" in joined


def test_synthesizer_requires_specialist_results(fake_model: FakeChatModel) -> None:
    with pytest.raises(ValueError, match="specialist_results"):
        asyncio.run(synthesizer_node(_state_with_plan()))


def test_unparsable_model_output_raises_a_named_error(
    fake_model: FakeChatModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed structured-output call must name the agent, not a schema."""

    class EmptyRunnable:
        async def ainvoke(self, *_: Any, **__: Any) -> None:
            return None

    monkeypatch.setattr(
        llm_module, "build_structured_runnable", lambda schema: EmptyRunnable()
    )
    with pytest.raises(RuntimeError, match="freudian"):
        asyncio.run(freudian_node(_state_with_plan()))


def test_invalid_structured_output_is_retried_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing required field is a provider slip: retry, then continue."""

    class FlakyRunnable:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, *_: Any, **__: Any) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 1:
                return {"perspective": "freudian", "observations": []}
            return make_school_analysis("freudian")

    runnable = FlakyRunnable()
    monkeypatch.setattr(
        llm_module, "build_structured_runnable", lambda schema: runnable
    )
    update = asyncio.run(freudian_node(_state_with_evidence()))

    assert runnable.calls == 2
    results = cast("dict[str, dict[str, Any]]", update["specialist_results"])
    assert results["freudian"]["summary"] == make_school_analysis("freudian")["summary"]


def test_persistently_invalid_output_reports_the_missing_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenRunnable:
        async def ainvoke(self, *_: Any, **__: Any) -> dict[str, Any]:
            return {"perspective": "freudian", "observations": ["用户梦见水"]}

    monkeypatch.setattr(
        llm_module, "build_structured_runnable", lambda schema: BrokenRunnable()
    )
    with pytest.raises(RuntimeError, match="summary: missing"):
        asyncio.run(freudian_node(_state_with_evidence()))


def test_context_message_renders_each_school() -> None:
    message = specialist_results_message(
        cast("Any", _state_with_plan(specialist_results=_three_results()))
    )
    content = str(message.content)
    assert "freudian" in content
    assert "object_relations" in content
    assert "lacanian" in content
    assert content.count("summary:") == 3


def test_context_message_reports_a_missing_school() -> None:
    partial = _three_results()
    partial.pop("lacanian")
    message = specialist_results_message(
        cast("Any", _state_with_plan(specialist_results=partial))
    )
    assert "缺少结果" in str(message.content)


def test_supervisor_plan_message_lists_all_focus_lines() -> None:
    content = str(supervisor_plan_message(cast("Any", _state_with_plan())).content)
    for key in (
        "task_summary",
        "analysis_focus",
        "freudian_focus",
        "object_relations_focus",
        "lacanian_focus",
        "synthesis_goal",
        "clinical_diagnosis_requested",
    ):
        assert key in content


def _state_with_evidence(
    evidence_by_school: dict[str, list[dict[str, Any]]] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build a specialist/synthesizer state that carries retrieved evidence."""
    state = _state_with_plan(
        evidence_by_school=(
            evidence_by_school
            if evidence_by_school is not None
            else make_evidence_by_school()
        ),
        **extra,
    )
    return state


def test_specialist_prompt_lists_only_its_own_evidence(
    fake_model: FakeChatModel,
) -> None:
    asyncio.run(freudian_node(_state_with_evidence()))

    content = "\n".join(
        str(message.content) for message in fake_model.calls_for(SchoolAnalysis)[0]
    )
    assert "freudian_note_000000" in content
    assert "object_relations_note_000000" not in content
    assert "lacanian_note_000000" not in content
    # The citation contract travels with the passages.
    assert "evidence_ids 只能从下面列出的 evidence_id 中选择" in content
    assert "理论依据" in content or "文献证据使用规则" in content


def test_specialist_prompt_states_when_there_is_no_evidence(
    fake_model: FakeChatModel,
) -> None:
    empty = {school: [] for school in ("freudian", "object_relations", "lacanian")}
    asyncio.run(freudian_node(_state_with_evidence(empty)))

    content = "\n".join(
        str(message.content) for message in fake_model.calls_for(SchoolAnalysis)[0]
    )
    assert "没有可用的本地文献证据" in content
    assert "不得编造任何 evidence_id" in content


def test_specialist_can_cite_the_evidence_it_received(
    fake_model: FakeChatModel,
) -> None:
    fake_model.overrides = {
        SchoolAnalysis: make_school_analysis(
            "freudian",
            interpretations=[
                {
                    "perspective": "freudian",
                    "claim": "可以借助防御机制来读解这段材料。",
                    "textual_basis": ["我梦见水"],
                    "uncertainty": "水的状态未知。",
                    "evidence_ids": ["freudian_note_000000"],
                }
            ],
        )
    }
    update = asyncio.run(freudian_node(_state_with_evidence()))

    results = cast("dict[str, dict[str, Any]]", update["specialist_results"])
    cited = results["freudian"]["interpretations"][0]["evidence_ids"]
    assert cited == ["freudian_note_000000"]


def test_specialist_rejects_an_invented_evidence_id(
    fake_model: FakeChatModel,
) -> None:
    fake_model.overrides = {
        SchoolAnalysis: make_school_analysis(
            "freudian",
            interpretations=[
                {
                    "perspective": "freudian",
                    "claim": "声称有文献支持的解释。",
                    "textual_basis": ["我梦见水"],
                    "uncertainty": "无。",
                    "evidence_ids": ["freud_1915_repression_p42"],
                }
            ],
        )
    }
    with pytest.raises(InvalidEvidenceReferenceError, match="freud_1915"):
        asyncio.run(freudian_node(_state_with_evidence()))


def test_specialist_citing_without_evidence_fails(
    fake_model: FakeChatModel,
) -> None:
    empty = {school: [] for school in ("freudian", "object_relations", "lacanian")}
    fake_model.overrides = {
        SchoolAnalysis: make_school_analysis(
            "freudian",
            interpretations=[
                {
                    "perspective": "freudian",
                    "claim": "没有证据却引用了证据。",
                    "textual_basis": ["我梦见水"],
                    "uncertainty": "无。",
                    "evidence_ids": ["freudian_note_000000"],
                }
            ],
        )
    }
    with pytest.raises(InvalidEvidenceReferenceError):
        asyncio.run(freudian_node(_state_with_evidence(empty)))


def _analysis_citing(evidence_ids: list[str]) -> dict[str, Any]:
    """Build a Freudian analysis whose single interpretation cites given ids."""
    return make_school_analysis(
        "freudian",
        interpretations=[
            {
                "perspective": "freudian",
                "claim": "借助检索到的文献读解这段材料。",
                "textual_basis": ["我梦见水"],
                "uncertainty": "水的状态未知。",
                "evidence_ids": evidence_ids,
            }
        ],
    )


def test_specialist_resolves_an_abbreviated_evidence_id(
    fake_model: FakeChatModel,
) -> None:
    """A dropped prefix is a formatting slip, not invented literature.

    The model cited ``note_000000`` while the real chunk id is
    ``freudian_note_000000``. The citation is mapped back to the offered id
    instead of failing the run, and the stored id is the real one.
    """
    fake_model.overrides = {SchoolAnalysis: _analysis_citing(["note_000000"])}
    update = asyncio.run(freudian_node(_state_with_evidence()))

    results = cast("dict[str, dict[str, Any]]", update["specialist_results"])
    assert results["freudian"]["interpretations"][0]["evidence_ids"] == [
        "freudian_note_000000"
    ]


def test_specialist_resolution_ignores_case_and_punctuation(
    fake_model: FakeChatModel,
) -> None:
    fake_model.overrides = {
        SchoolAnalysis: _analysis_citing(
            ["Freudian_Note_000000.", "freudian_NOTe_000000"]
        )
    }
    update = asyncio.run(freudian_node(_state_with_evidence()))

    results = cast("dict[str, dict[str, Any]]", update["specialist_results"])
    assert results["freudian"]["interpretations"][0]["evidence_ids"] == [
        "freudian_note_000000"
    ]


def test_an_ambiguous_abbreviation_is_rejected(fake_model: FakeChatModel) -> None:
    """When a short id could mean two different chunks, the run fails."""
    ambiguous = {
        "freudian": ["freudian_a_note_000000", "freudian_b_note_000000"],
    }
    state = _state_with_evidence(make_evidence_by_school(ambiguous))
    fake_model.overrides = {SchoolAnalysis: _analysis_citing(["note_000000"])}

    with pytest.raises(InvalidEvidenceReferenceError, match="note_000000"):
        asyncio.run(freudian_node(state))


def test_low_level_evidence_helpers_still_behave() -> None:
    analysis = make_school_analysis(
        "freudian",
        interpretations=[
            {
                "perspective": "freudian",
                "claim": "读解。",
                "textual_basis": ["我梦见水"],
                "uncertainty": "未知。",
                "evidence_ids": ["a", "b"],
            }
        ],
    )
    assert cited_evidence_ids(
        SchoolAnalysis.model_validate({**analysis, "perspective": "freudian"})
    ) == ["a", "b"]
    assert collect_invalid_evidence_ids(["a", "c"], {"a", "b"}) == ["c"]
    validate_evidence_references(["a"], {"a", "b"}, "agent")
    with pytest.raises(InvalidEvidenceReferenceError):
        validate_evidence_references(["c"], {"a", "b"}, "agent")


def test_synthesizer_prompt_lists_the_evidence_catalogue(
    fake_model: FakeChatModel,
) -> None:
    asyncio.run(
        synthesizer_node(_state_with_evidence(specialist_results=_three_results()))
    )

    content = "\n".join(
        str(message.content) for message in fake_model.calls_for(SynthesisResult)[0]
    )
    assert "freudian_note_000000" in content
    assert "lacanian_note_000000" in content
    assert "used_evidence_ids 只能从上面的 id 中选择" in content


def test_synthesizer_keeps_citations_in_the_draft(
    fake_model: FakeChatModel,
) -> None:
    """The draft carries ids; rendering the source block belongs to the finalizer."""
    fake_model.overrides = {
        SynthesisResult: make_synthesis(used_evidence_ids=["freudian_note_000000"]),
    }
    state = _state_with_evidence(specialist_results=_three_results())
    update = asyncio.run(synthesizer_node(state))

    draft = cast("dict[str, Any]", update["draft_result"])
    assert draft["used_evidence_ids"] == ["freudian_note_000000"]
    assert "理论依据（本地知识库）" not in draft["final_response"]
    assert "[E1]" not in draft["final_response"]


def test_synthesizer_cannot_cite_evidence_that_was_not_retrieved(
    fake_model: FakeChatModel,
) -> None:
    fake_model.overrides = {
        SynthesisResult: make_synthesis(used_evidence_ids=["ghost_chunk_000042"]),
    }
    with pytest.raises(InvalidEvidenceReferenceError, match="ghost_chunk_000042"):
        asyncio.run(
            synthesizer_node(_state_with_evidence(specialist_results=_three_results()))
        )


def test_synthesizer_without_evidence_cites_nothing(
    fake_model: FakeChatModel,
) -> None:
    empty = {school: [] for school in ("freudian", "object_relations", "lacanian")}
    update = asyncio.run(
        synthesizer_node(
            _state_with_evidence(empty, specialist_results=_three_results())
        )
    )

    draft = cast("dict[str, Any]", update["draft_result"])
    assert draft["used_evidence_ids"] == []
    assert draft["final_response"] == make_synthesis()["final_response"]
