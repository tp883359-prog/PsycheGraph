"""Finalizer tests: exactly one user-visible message, always.

`finalize` publishes the accepted draft; `safe_finalize` publishes a
conservative fallback when the revision budget is spent and the Critic still
rejects the draft. Neither calls a model.
"""

import asyncio
from typing import Any, cast

from langchain_core.messages import HumanMessage

from react_agent.agents.finalizer import (
    build_safe_fallback,
    finalize_node,
    render_visible_answer,
    safe_finalize_node,
    used_evidence_items,
)
from react_agent.state import SPECIALIST_PERSPECTIVES
from tests.unit_tests.conftest import (
    make_evidence_by_school,
    make_plan,
    make_school_analysis,
    make_synthesis,
)


def _state(**overrides: Any) -> dict[str, Any]:
    """Build a state ready for finalization."""
    state: dict[str, Any] = {
        "messages": [HumanMessage(content="我梦见水。")],
        "supervisor_plan": make_plan(),
        "evidence_by_school": make_evidence_by_school(),
        "specialist_results": {
            school: make_school_analysis(school) for school in SPECIALIST_PERSPECTIVES
        },
        "draft_result": make_synthesis(),
        "critique": {"verdict": "pass", "issues": []},
        "revision_count": 0,
    }
    state.update(overrides)
    return state


def test_finalize_publishes_the_draft_and_one_message() -> None:
    update = asyncio.run(finalize_node(_state()))

    assert set(update) == {"final_result", "finalization_status", "messages"}
    assert update["finalization_status"] == "passed"
    messages = cast("list[Any]", update["messages"])
    assert len(messages) == 1
    assert messages[0].type == "ai"
    assert messages[0].content == make_synthesis()["final_response"]


def test_finalize_marks_a_revised_run() -> None:
    update = asyncio.run(finalize_node(_state(revision_count=1)))

    assert update["finalization_status"] == "revised_and_passed"


def test_finalize_renders_sources_from_real_metadata() -> None:
    draft = make_synthesis(
        final_response="回答正文。", used_evidence_ids=["freudian_note_000000"]
    )
    update = asyncio.run(finalize_node(_state(draft_result=draft)))

    content = cast("list[Any]", update["messages"])[0].content
    assert content.startswith("回答正文。")
    assert "理论依据（本地知识库）" in content
    assert "[E1]" in content
    assert "《测试材料》" in content
    assert "p. " not in content


def test_unknown_used_ids_are_not_rendered() -> None:
    draft = make_synthesis(
        final_response="回答正文。", used_evidence_ids=["ghost_000001"]
    )
    items = used_evidence_items(_state(), draft)

    assert items == []
    assert render_visible_answer(draft, items) == "回答正文。"


def test_safe_finalize_keeps_only_safe_material() -> None:
    update = asyncio.run(safe_finalize_node(_state(revision_count=1)))

    assert update["finalization_status"] == "safe_fallback"
    final = cast("dict[str, Any]", update["final_result"])
    assert final["integrated_interpretation"] == ""
    assert final["common_ground"] == []
    assert final["differences"] == []
    assert final["used_evidence_ids"] == []
    assert final["limitations"] == ["材料不足。", "缺少场景与感受信息。"]
    assert final["follow_up_questions"] == ["水的状态是什么样的？"]
    assert final["clinical_diagnosis_refused"] is False
    # The school observations survive; the theory conclusions do not.
    assert "用户梦见水" in final["final_response"]
    assert "freudian 的一种读解。" not in final["final_response"]
    assert len(cast("list[Any]", update["messages"])) == 1


def test_safe_finalize_notes_the_clinical_boundary_when_asked() -> None:
    state = _state(
        supervisor_plan=make_plan(clinical_diagnosis_requested=True),
        specialist_results={
            school: make_school_analysis(school, clinical_diagnosis_refused=True)
            for school in SPECIALIST_PERSPECTIVES
        },
    )
    fallback = build_safe_fallback(state, {"verdict": "revise", "issues": []})

    assert fallback["clinical_diagnosis_refused"] is True
    assert "不能进行临床诊断" in fallback["final_response"]


def test_safe_finalize_does_not_leak_internal_review_text() -> None:
    critique = {
        "verdict": "revise",
        "issues": [
            {
                "category": "evidence_support",
                "severity": "error",
                "description": "INTERNAL-ISSUE-TEXT",
                "revision_instruction": "INTERNAL-INSTRUCTION-TEXT",
            }
        ],
    }
    update = asyncio.run(safe_finalize_node(_state(critique=critique)))

    content = cast("list[Any]", update["messages"])[0].content
    assert "INTERNAL-ISSUE-TEXT" not in content
    assert "INTERNAL-INSTRUCTION-TEXT" not in content
    assert "Critic" not in content
    assert "draft" not in content


def test_safe_fallback_is_deduplicated_and_bounded() -> None:
    state = _state(
        specialist_results={
            school: make_school_analysis(
                school, questions=["同一个问题？", "同一个问题？"]
            )
            for school in SPECIALIST_PERSPECTIVES
        }
    )
    fallback = build_safe_fallback(state, {})

    # One copy survives from three specialists and the draft, and the order
    # follows the specialists before the draft.
    assert fallback["follow_up_questions"] == ["同一个问题？", "水的状态是什么样的？"]
