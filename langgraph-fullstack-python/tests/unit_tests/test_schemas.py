"""Tests for the multi-agent schemas."""

import json

import pytest
from pydantic import ValidationError

from react_agent.schemas import (
    SchoolAnalysis,
    SpecialistPerspective,
    SupervisorPlan,
    SynthesisResult,
    TheoryInterpretation,
)


def _interpretation(perspective: str = "freudian") -> TheoryInterpretation:
    return TheoryInterpretation(
        perspective=perspective,  # type: ignore[arg-type]
        claim="一种解释是水这一意象可能与情绪状态有关。",
        textual_basis=["我梦见水"],
        uncertainty="用户没有描述水的状态和感受，其他解释同样成立。",
    )


def _school_analysis(perspective: str = "freudian") -> SchoolAnalysis:
    return SchoolAnalysis(
        perspective=perspective,  # type: ignore[arg-type]
        observations=["用户梦见水"],
        interpretations=[_interpretation(perspective)],
        limitations=["只出现一个意象，缺少场景与感受信息。"],
        questions=["水的状态是什么样的？"],
        summary="单凭水这一意象无法确定意义。",
    )


def test_specialist_perspectives_are_the_three_schools() -> None:
    assert SpecialistPerspective.__args__ == (  # type: ignore[attr-defined]
        "freudian",
        "object_relations",
        "lacanian",
    )


def test_interpretation_accepts_each_declared_perspective() -> None:
    for perspective in ("freudian", "object_relations", "lacanian"):
        assert _interpretation(perspective).perspective == perspective


def test_interpretation_rejects_unknown_perspective() -> None:
    with pytest.raises(ValidationError):
        _interpretation("jung")


def test_school_analysis_rejects_integrative_perspective() -> None:
    with pytest.raises(ValidationError):
        SchoolAnalysis(
            perspective="integrative",  # type: ignore[arg-type]
            observations=["用户梦见水"],
            interpretations=[],
            limitations=["信息不足。"],
            questions=[],
            summary="无法确定。",
        )


def test_school_analysis_accepts_each_school() -> None:
    for perspective in ("freudian", "object_relations", "lacanian"):
        assert _school_analysis(perspective).perspective == perspective


def test_schemas_have_no_clinical_or_numeric_fields() -> None:
    forbidden = {
        "diagnosis",
        "mental_health_score",
        "disorder_probability",
        "patient_status",
        "confidence",
        "confidence_score",
    }
    assert forbidden.isdisjoint(SchoolAnalysis.model_fields)
    assert forbidden.isdisjoint(SynthesisResult.model_fields)
    assert forbidden.isdisjoint(SupervisorPlan.model_fields)


def test_supervisor_plan_validation() -> None:
    plan = SupervisorPlan(
        task_summary="用户报告一个关于水的梦。",
        analysis_focus="材料只有一个意象，缺少场景与感受。",
        freudian_focus="关注愿望与象征化，不要预设内容。",
        object_relations_focus="关注是否有关系线索，没有就不展开。",
        lacanian_focus="关注能指与缺失，术语需解释。",
        clinical_diagnosis_requested=False,
        synthesis_goal="给出保守的、条件性的读解并提问。",
    )
    assert plan.clinical_diagnosis_requested is False
    with pytest.raises(ValidationError):
        SupervisorPlan(
            task_summary="用户询问自己是否有某种人格障碍。",
            analysis_focus="临床请求。",
            freudian_focus="只做理论说明。",
            object_relations_focus="只做理论说明。",
            lacanian_focus="只做理论说明。",
            clinical_diagnosis_requested=["yes"],  # type: ignore[arg-type]
            synthesis_goal="明确拒绝诊断。",
        )


def test_synthesis_result_validation() -> None:
    result = SynthesisResult(
        common_ground=["三个学派都注意到材料只有一个意象。"],
        differences=["弗洛伊德强调愿望，拉康强调能指结构。"],
        integrated_interpretation="在信息有限的情况下，只能给出条件性的读解。",
        limitations=["缺少场景与感受信息。"],
        follow_up_questions=["水的状态是什么样的？"],
        clinical_diagnosis_refused=False,
        final_response="目前信息很少，可以多说一点梦里的场景吗？",
    )
    assert result.clinical_diagnosis_refused is False


def test_schemas_convert_to_openai_tools() -> None:
    from langchain_core.utils.function_calling import convert_to_openai_tool

    for schema, expected_required in (
        (
            SupervisorPlan,
            {
                "task_summary",
                "analysis_focus",
                "freudian_focus",
                "object_relations_focus",
                "lacanian_focus",
                "synthesis_goal",
            },
        ),
        (
            SchoolAnalysis,
            {
                "perspective",
                "observations",
                "interpretations",
                "limitations",
                "questions",
                "summary",
            },
        ),
        (
            SynthesisResult,
            {
                "common_ground",
                "differences",
                "integrated_interpretation",
                "limitations",
                "follow_up_questions",
                "final_response",
            },
        ),
    ):
        parameters = convert_to_openai_tool(schema)["function"]["parameters"]
        assert set(parameters["required"]) == expected_required

    perspective = convert_to_openai_tool(SchoolAnalysis)["function"]["parameters"][
        "properties"
    ]["perspective"]
    assert perspective["enum"] == ["freudian", "object_relations", "lacanian"]


def test_json_style_escapes_are_normalized() -> None:
    analysis = SchoolAnalysis(
        perspective="freudian",
        observations=["用户写下一行\\n又写一行"],
        interpretations=[
            TheoryInterpretation(
                perspective="freudian",
                claim="第一行\\n\\n第二行",
                textual_basis=['提到的\\"水\\"'],
                uncertainty="缺少信息\\r\\n无法确定",
            )
        ],
        limitations=["材料不足"],
        questions=["后来呢"],
        summary='他问到\\"水\\"\\n\\n然后停住了',
    )
    assert analysis.observations == ["用户写下一行\n又写一行"]
    assert analysis.interpretations[0].claim == "第一行\n\n第二行"
    assert analysis.interpretations[0].textual_basis == ['提到的"水"']
    assert analysis.summary == '他问到"水"\n\n然后停住了'


def test_school_analysis_dumps_to_json() -> None:
    payload = _school_analysis("lacanian").model_dump(mode="json")
    assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload
    assert isinstance(payload["interpretations"][0]["perspective"], str)
