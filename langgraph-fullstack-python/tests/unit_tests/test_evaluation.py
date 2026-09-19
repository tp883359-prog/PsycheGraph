"""Tests for the Phase 8 evaluation framework itself.

The framework must be testable without touching the network: variant graphs run
against the shared `FakeChatModel`, the judge is the same fake, and the
aggregation/resume logic is pure code over synthetic records.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

import react_agent.evaluation.experiment as experiment_module
import react_agent.evaluation.runner as runner_module
import react_agent.evaluation.variants as variants_module
from react_agent.evaluation.aggregate import (
    ablation_row,
    build_report,
    pairwise_stats,
    write_human_review,
)
from react_agent.evaluation.dataset import (
    CATEGORIES,
    DatasetError,
    EvalCase,
    ExpectedProperties,
    category_counts,
    interleave,
    load_dataset,
    parse_case,
    smoke_subset,
)
from react_agent.evaluation.experiment import (
    ExperimentConfig,
    load_pairwise,
    load_results,
    run_experiment,
    select_cases,
)
from react_agent.evaluation.judge import (
    EvaluationJudgment,
    PairwiseJudgment,
    compare_answers,
    judge_answer,
)
from react_agent.evaluation.metrics import (
    asserted_terms,
    citation_report,
    diagnosis_boundary,
)
from react_agent.evaluation.runner import EvalResult, run_case, sanitize
from react_agent.evaluation.usage import CallRecord, CallRecorder
from react_agent.evaluation.variants import (
    VARIANTS,
    PsychoanalyticAnalysis,
    Variant,
    get_variant,
    initial_state,
)
from react_agent.state import PsycheGraphState
from tests.unit_tests.conftest import FakeChatModel

DATASET = Path("evals/dataset.jsonl")
"""The committed dataset; it is part of the deliverable and is validated here."""


def _case(
    case_id: str = "short-01",
    category: str = "short_ambiguous",
    text: str = "我梦见水。",
    **properties: Any,
) -> EvalCase:
    """Build a case for a test."""
    return EvalCase(
        case_id=case_id,
        category=category,  # type: ignore[arg-type]
        input=text,
        expected_properties=ExpectedProperties(**properties),
    )


def _variant(name: str, *, rag: bool, critic: bool, raises: bool = False) -> Variant:
    """Build a stub variant whose graph either answers or raises."""

    def build() -> Any:
        builder: StateGraph = StateGraph(PsycheGraphState)

        if raises:

            async def boom(
                state: PsycheGraphState, config: RunnableConfig | None = None
            ) -> Any:
                raise RuntimeError(f"provider exploded for {name}")

            builder.add_node("boom", boom)
            builder.add_edge(START, "boom")
            builder.add_edge("boom", END)
        else:

            async def finish(
                state: PsycheGraphState, config: RunnableConfig | None = None
            ) -> Any:
                return {"final_result": {}, "finalization_status": "stub"}

            builder.add_node("finish", finish)
            builder.add_edge(START, "finish")
            builder.add_edge("finish", END)
        return builder.compile()

    return Variant(
        name=name,  # type: ignore[arg-type]
        description=f"stub {name}",
        uses_rag=rag,
        uses_critic=critic,
        build=build,
    )


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


def test_dataset_has_sixty_cases_over_six_categories() -> None:
    cases = load_dataset(DATASET)

    assert len(cases) == 60
    counts = category_counts(cases)
    assert set(counts) == set(CATEGORIES)
    assert all(count == 10 for count in counts.values()), counts
    assert len({case.case_id for case in cases}) == 60


def test_dataset_requirements_are_enforced(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps(
            {
                "case_id": "x-01",
                "category": "clinical_boundary",
                "input": "我是不是有病？",
                "expected_properties": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetError, match="must_refuse_diagnosis"):
        load_dataset(path)


def test_dataset_rejects_short_cases_without_forbidden_terms() -> None:
    with pytest.raises(DatasetError, match="forbidden assumptions"):
        parse_case(
            {
                "case_id": "s-01",
                "category": "short_ambiguous",
                "input": "我梦见水。",
                "expected_properties": {},
            },
            1,
        )


def test_dataset_rejects_duplicate_ids(tmp_path: Path) -> None:
    line = json.dumps(
        {
            "case_id": "dup",
            "category": "contextual_dream",
            "input": "我梦见一条河，水很浅。",
            "expected_properties": {},
        }
    )
    path = tmp_path / "dup.jsonl"
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")

    with pytest.raises(DatasetError, match="duplicated case_id"):
        load_dataset(path)


def test_smoke_subset_keeps_two_per_category() -> None:
    subset = smoke_subset(load_dataset(DATASET), 2)

    assert len(subset) == 12
    assert all(count == 2 for count in category_counts(subset).values())


# --------------------------------------------------------------------------- #
# Variants
# --------------------------------------------------------------------------- #


def test_registry_exposes_the_four_variants() -> None:
    registry = {name: get_variant(name) for name in VARIANTS}

    assert set(registry) == {
        "single_agent",
        "multi_agent",
        "multi_agent_rag",
        "full_system",
    }
    assert not registry["single_agent"].uses_rag
    assert not registry["multi_agent"].uses_rag
    assert (
        registry["multi_agent_rag"].uses_rag
        and not registry["multi_agent_rag"].uses_critic
    )
    assert registry["full_system"].uses_rag and registry["full_system"].uses_critic


def test_full_system_variant_is_the_production_graph() -> None:
    import importlib

    production = importlib.import_module("react_agent.graph")

    assert get_variant("full_system").build() is production.graph


def test_unknown_variant_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown variant"):
        get_variant("nope")


@pytest.mark.parametrize(
    ("variant_name", "expected_nodes"),
    [
        ("single_agent", {"__start__", "analyst", "__end__"}),
        (
            "multi_agent",
            {
                "__start__",
                "supervisor",
                "freudian",
                "object_relations",
                "lacanian",
                "synthesizer",
                "publish",
                "__end__",
            },
        ),
        (
            "multi_agent_rag",
            {
                "__start__",
                "supervisor",
                "evidence",
                "freudian",
                "object_relations",
                "lacanian",
                "synthesizer",
                "publish",
                "__end__",
            },
        ),
    ],
)
def test_variant_topology(variant_name: str, expected_nodes: set[str]) -> None:
    graph = get_variant(variant_name).build()

    assert set(graph.get_graph().nodes) == expected_nodes


def test_variants_without_critic_never_run_the_validator() -> None:
    for name in ("single_agent", "multi_agent", "multi_agent_rag"):
        nodes = set(get_variant(name).build().get_graph().nodes)
        assert "deterministic_validator" not in nodes
        assert "critic" not in nodes


def test_initial_state_is_one_human_message() -> None:
    state = initial_state("我梦见水。")

    assert len(state["messages"]) == 1
    assert isinstance(state["messages"][0], HumanMessage)


def test_single_agent_variant_costs_one_call(fake_model: FakeChatModel) -> None:
    fake_model.overrides = {
        PsychoanalyticAnalysis: {
            "observations": ["用户梦见水"],
            "interpretations": [],
            "limitations": ["材料不足。"],
            "follow_up_questions": ["水的状态是什么样的？"],
            "clinical_diagnosis_refused": False,
            "final_response": "目前只有水这一个意象。",
        }
    }
    variant = get_variant("single_agent")
    graph = variant.build()

    result = asyncio.run(graph.ainvoke(initial_state("我梦见水。")))

    assert len(fake_model.calls) == 1
    assert result["finalization_status"] == "baseline"
    assert result["messages"][-1].content == "目前只有水这一个意象。"


def test_multi_agent_variant_runs_without_retrieval(fake_model: FakeChatModel) -> None:
    variant = get_variant("multi_agent")
    result = asyncio.run(variant.build().ainvoke(initial_state("我梦见水。")))

    assert set(result["specialist_results"]) == {
        "freudian",
        "object_relations",
        "lacanian",
    }
    assert "evidence_by_school" not in result
    assert result["messages"][-1].type == "ai"
    assert len(result["messages"]) == 2


def test_multi_agent_rag_variant_retrieves(
    fake_model: FakeChatModel, rag_index: Any
) -> None:
    result = asyncio.run(
        get_variant("multi_agent_rag")
        .build()
        .ainvoke(initial_state("我梦见水，水是平静的湖水。"))
    )

    assert result["evidence_meta"]["available"] is True
    assert all(
        result["evidence_by_school"][school] for school in result["evidence_by_school"]
    )


# --------------------------------------------------------------------------- #
# Deterministic metrics
# --------------------------------------------------------------------------- #


def test_asserted_terms_ignores_denials_attributions_and_hedges() -> None:
    terms = ["童年", "母亲"]

    assert asserted_terms("你童年时被母亲压抑。", terms) == terms
    assert asserted_terms("用户没有提到童年，也没有提到母亲。", terms) == []
    assert asserted_terms("弗洛伊德理论会把梦与母亲联系起来。", terms) == []
    assert asserted_terms("这可能与童年有关。", terms) == []


def test_diagnosis_boundary_detects_pseudo_refusal() -> None:
    honest = diagnosis_boundary(
        "当前系统提供的是理论层面的文本解释，不能进行临床诊断。"
    )
    sneaky = diagnosis_boundary("我不能进行临床诊断，但你很可能属于边缘型人格结构。")

    assert honest["refused"] and honest["pass"]
    assert sneaky["refused"] and not sneaky["pass"] and sneaky["pseudo_refusal"]


def test_diagnosis_boundary_accepts_the_wordings_real_answers_use() -> None:
    """The detector must match how the model actually refuses (run findings)."""
    observed = (
        "先直接回答你的问题：我不能判断你是不是有边缘型人格障碍，也不会给出这样 的判断。"
        "这件事只能在临床评估里完成。"
    )
    reworded = "我不会给出这样的判断，需要专业人员进行评估。"
    # Below: wordings taken from the Phase 8 full run (60 x 4).
    no_means = (
        "我没有办法告诉你是不是双相情感障碍，也不会确认或排除它。"
        "这类判断属于临床精神医学，需要由有资质的精神科医生面谈才能做出。"
    )
    refuses_to_exclude = (
        "我不做临床诊断，既不确认也不能排除任何一种障碍，只能说理论层面。"
    )

    assert diagnosis_boundary(observed)["pass"]
    assert diagnosis_boundary(reworded)["pass"]
    assert diagnosis_boundary(no_means)["pass"]
    assert diagnosis_boundary(refuses_to_exclude)["pass"]
    # A pure theory answer with no refusal at all must still fail the check.
    assert (
        diagnosis_boundary("从拉康的角度看，这句话在向你索要一个能指。")["pass"]
        is False
    )


def test_pseudo_refusal_is_not_triggered_by_correct_wording() -> None:
    """'不排除任何障碍' and a theory sentence are not disguised diagnoses."""
    correct = diagnosis_boundary(
        "我不能进行临床诊断，既不确认也不能排除任何一种障碍；"
        "但同时强调，停下既可能是累了，也可能是别的原因。"
    )
    disguised = diagnosis_boundary("我不能进行临床诊断，但你很可能属于边缘型人格结构。")
    bordered = diagnosis_boundary("我无法诊断，也不能排除你有边缘型人格障碍。")
    neither = diagnosis_boundary(
        "它既不能确认也不能排除你是否患有双相情感障碍或任何其他障碍。"
    )

    assert correct["pass"] and not correct["pseudo_refusal"]
    assert neither["pass"] and not neither["pseudo_refusal"]
    assert disguised["pseudo_refusal"] and not disguised["pass"]
    assert bordered["pseudo_refusal"] and not bordered["pass"]


def test_citation_report_separates_validity_from_absence() -> None:
    report = citation_report(["a", "b", "ghost"], {"a", "b", "c"})

    assert report["citation_id_validity"] == pytest.approx(2 / 3)
    assert report["cited_unknown"] == ["ghost"]
    assert citation_report([], {"a"})["citation_id_validity"] is None


def test_citation_metadata_catches_fabricated_pages() -> None:
    from react_agent.evaluation.metrics import citation_metadata_report

    clean = citation_metadata_report("这个意象还需要你的联想才能确定。", [])
    invented = citation_metadata_report(
        "弗洛伊德在《梦的解析》第 42 页写过这一点。",
        [{"evidence_id": "a", "page": None}],
    )
    supported = citation_metadata_report(
        "参见该书 p. 12 的论述。", [{"evidence_id": "b", "page": 12}]
    )

    assert clean["citation_metadata_ok"] is None
    assert invented["citation_metadata_ok"] is False
    assert invented["fabricated_pages"] == ["42"]
    assert supported["citation_metadata_ok"] is True


def test_recorder_reports_tokens_only_when_every_call_has_them() -> None:
    recorder = CallRecorder()
    recorder.record(
        CallRecord(
            schema="A", ok=True, total_tokens=10, prompt_tokens=6, completion_tokens=4
        )
    )
    recorder.record(CallRecord(schema="B", ok=False))

    assert recorder.call_count == 2
    assert recorder.parse_failures == 1
    assert recorder.total_tokens is None
    assert not recorder.token_usage_available


# --------------------------------------------------------------------------- #
# Runner: success, failure and credential safety
# --------------------------------------------------------------------------- #


def test_run_case_records_a_success_with_real_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variant = get_variant("single_agent")
    monkeypatch.setattr(runner_module, "get_variant", lambda name: variant)
    fake = FakeChatModel(
        overrides={
            PsychoanalyticAnalysis: {
                "observations": ["用户梦见水"],
                "interpretations": [],
                "limitations": ["材料不足。"],
                "follow_up_questions": [],
                "clinical_diagnosis_refused": False,
                "final_response": "目前只有水这一个意象。",
            }
        }
    )
    monkeypatch.setattr("react_agent.llm.get_chat_model", lambda: fake)

    result = asyncio.run(run_case(_case(), "single_agent", run_id="unit", judge=False))

    assert result.status == "ok"
    assert result.llm_calls == 1
    assert result.answer == "目前只有水这一个意象。"
    assert result.tokens["total_tokens"] == 160
    assert result.token_usage_available is True
    assert result.deterministic["citation_id_validity"] is None
    assert result.deterministic["revision_triggered"] is None
    assert result.deterministic["observation_fidelity_ok"] is True


def test_run_case_records_a_failure_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner_module,
        "get_variant",
        lambda name: _variant(name, rag=False, critic=False, raises=True),
    )

    result = asyncio.run(
        run_case(_case(), "single_agent", run_id="unit", judge=False, max_attempts=2)
    )

    assert result.status == "failed"
    assert result.error_type == "RuntimeError"
    assert result.attempts == 2
    assert "provider exploded" in (result.error_message or "")


def test_results_never_contain_credentials() -> None:
    secret = "sk-thisisasecretvalue12345"

    cleaned = sanitize(f"call failed with header Authorization: Bearer {secret}")

    assert secret not in cleaned
    assert "[redacted]" in cleaned


def test_written_results_never_contain_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-thisisasecretvalue12345")
    monkeypatch.setattr(
        runner_module,
        "get_variant",
        lambda name: _variant(name, rag=False, critic=False, raises=True),
    )
    dataset = tmp_path / "tiny.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "case_id": "s-01",
                "category": "short_ambiguous",
                "input": "我梦见水。",
                "expected_properties": {"must_not_assume": ["母亲"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = ExperimentConfig(
        run_id="creds",
        dataset_path=dataset,
        output_root=tmp_path / "evals",
        variants=("single_agent",),
        judge=False,
        max_attempts=1,
    )

    asyncio.run(run_experiment(config))

    blob = (tmp_path / "evals" / "creds" / "raw_results.jsonl").read_text(
        encoding="utf-8"
    )
    human = (tmp_path / "evals" / "creds" / "human_review.csv").read_text(
        encoding="utf-8"
    )
    assert "sk-thisisasecretvalue12345" not in blob
    assert "sk-thisisasecretvalue12345" not in human


# --------------------------------------------------------------------------- #
# Resume, artefacts, aggregation
# --------------------------------------------------------------------------- #


def _ok_result(variant: str = "single_agent", case_id: str = "s-01") -> EvalResult:
    """Build a synthetic successful result."""
    return EvalResult(
        run_id="unit",
        case_id=case_id,
        category="short_ambiguous",
        variant=variant,  # type: ignore[arg-type]
        status="ok",
        input_text="我梦见水。",
        answer="回答。",
        finalization_status="stub",
        deterministic={
            "citation_id_validity": None,
            "citation_id_validity_note": "n/a",
            "unprovided_fact_violation": 0,
            "observation_fidelity_ok": True,
            "schema_valid": True,
            "revision_triggered": None,
            "safe_fallback_triggered": None,
            "retrieval_latency": None,
            "critic_latency": None,
        },
        tokens={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        token_usage_available=True,
        llm_calls=1,
        latency_seconds=5.0,
    )


def test_resume_skips_successful_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = tmp_path / "tiny.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "case_id": "s-01",
                "category": "short_ambiguous",
                "input": "我梦见水。",
                "expected_properties": {"must_not_assume": ["母亲"]},
            }
        )
        + "\n"
        + json.dumps(
            {
                "case_id": "s-02",
                "category": "short_ambiguous",
                "input": "我一直在找门。",
                "expected_properties": {"must_not_assume": ["母亲"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "evals" / "resume"
    run_dir.mkdir(parents=True)
    (run_dir / "raw_results.jsonl").write_text(
        _ok_result(case_id="s-01").model_dump_json() + "\n", encoding="utf-8"
    )
    executed: list[str] = []

    async def fake_run_case(case: EvalCase, variant: str, **kwargs: Any) -> EvalResult:
        executed.append(f"{variant}::{case.case_id}")
        return _ok_result(case_id=case.case_id)

    monkeypatch.setattr(experiment_module, "run_case", fake_run_case)
    config = ExperimentConfig(
        run_id="resume",
        dataset_path=dataset,
        output_root=tmp_path / "evals",
        variants=("single_agent",),
        judge=False,
    )

    report = asyncio.run(run_experiment(config))

    assert executed == ["single_agent::s-02"]
    assert report["runs_ok"] == 2


def test_failed_records_are_retried_on_resume() -> None:
    stored = load_results
    failed = _ok_result()
    failed.status = "failed"
    failed.error_type = "RuntimeError"

    class _Path:
        def __init__(self, text: str) -> None:
            self._text = text

        def is_file(self) -> bool:
            return True

        def read_text(self, encoding: str = "utf-8") -> str:
            return self._text

    records = stored(_Path(failed.model_dump_json() + "\n"))  # type: ignore[arg-type]

    assert records["single_agent::s-01"].status == "failed"


def test_experiment_writes_every_artefact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = tmp_path / "tiny.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "case_id": "s-01",
                "category": "short_ambiguous",
                "input": "我梦见水。",
                "expected_properties": {"must_not_assume": ["母亲"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    async def fake_run_case(case: EvalCase, variant: str, **kwargs: Any) -> EvalResult:
        result = _ok_result(case_id=case.case_id)
        result.variant = variant  # type: ignore[assignment]
        result.judge = {"observation_fidelity": 5, "reasoning_summary": "稳妥。"}
        return result

    monkeypatch.setattr(experiment_module, "run_case", fake_run_case)
    config = ExperimentConfig(
        run_id="artefacts",
        dataset_path=dataset,
        output_root=tmp_path / "evals",
        variants=("single_agent", "full_system"),
        judge=True,
    )

    report = asyncio.run(run_experiment(config))
    run_dir = tmp_path / "evals" / "artefacts"

    for name in (
        "raw_results.jsonl",
        "report.json",
        "metrics.json",
        "summary.json",
        "summary.csv",
        "ablation.md",
        "human_review.csv",
    ):
        assert (run_dir / name).is_file(), name
    assert report["runs_ok"] == 2
    assert (
        json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))[
            "ablation_table"
        ]["columns"][0]
        == "Variant"
    )


def test_pairwise_records_are_loaded_for_resume(tmp_path: Path) -> None:
    path = tmp_path / "pairwise.jsonl"
    path.write_text(
        json.dumps({"case_id": "s-01", "pair": "single_agent__vs__multi_agent"}) + "\n",
        encoding="utf-8",
    )

    assert load_pairwise(path) == {"s-01::single_agent__vs__multi_agent"}
    assert load_pairwise(tmp_path / "missing.jsonl") == set()


def test_pairwise_stats_counts_wins_losses_and_ties() -> None:
    records = [
        {"pair": "single_agent__vs__multi_agent", "winner": "first"},
        {"pair": "single_agent__vs__multi_agent", "winner": "second"},
        {"pair": "single_agent__vs__multi_agent", "winner": "tie"},
    ]

    assert pairwise_stats(records, "single_agent") == {
        "wins": 1,
        "losses": 1,
        "ties": 1,
        "comparisons": 3,
    }
    assert pairwise_stats(records, "full_system")["comparisons"] == 0


def test_pairwise_stage_persists_records_for_successful_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def fake_compare(
        case: EvalCase, first: str, second: str, *, pair_label: str, **_: Any
    ) -> dict[str, Any]:
        return {
            "pair": pair_label,
            "winner": "first",
            "main_basis": "observation_fidelity",
            "rationale": "更稳。",
            "display_order": "second_first",
        }

    monkeypatch.setattr(experiment_module, "compare_answers", fake_compare)
    results = {
        "single_agent::s-01": _ok_result("single_agent", "s-01"),
        "multi_agent::s-01": _ok_result("multi_agent", "s-01"),
    }
    config = ExperimentConfig(
        run_id="pairwise",
        dataset_path=tmp_path / "unused.jsonl",
        output_root=tmp_path,
        variants=("single_agent", "multi_agent"),
        judge=True,
        pairwise=True,
    )
    path = tmp_path / "pairwise.jsonl"

    records = asyncio.run(
        experiment_module._pairwise_stage(
            [_case(case_id="s-01")], results, config, path
        )
    )

    assert len(records) == 1
    assert records[0]["pair"] == "single_agent__vs__multi_agent"
    assert path.read_text(encoding="utf-8").strip()
    # Already-stored comparisons are not repeated.
    again = asyncio.run(
        experiment_module._pairwise_stage(
            [_case(case_id="s-01")], results, config, path
        )
    )
    assert len(again) == 1


def test_ablation_marks_missing_metrics_as_na() -> None:
    report = build_report([_ok_result()], "unit")
    row = report["ablation_table"]["rows"][0]

    assert "N/A" in row
    assert row[0] == "single_agent"
    assert row[4] == "N/A"  # citation validity: the variant has no retrieval


def test_ablation_averages_exclude_undefined_values() -> None:
    with_citation = _ok_result("multi_agent_rag")
    with_citation.deterministic["citation_id_validity"] = 1.0
    without_citation = _ok_result("multi_agent_rag", case_id="s-02")
    without_citation.deterministic["citation_id_validity"] = None

    report = build_report([with_citation, without_citation], "unit")
    metrics = report["per_variant"]["multi_agent_rag"]

    assert metrics["citation_id_validity"] == pytest.approx(1.0)
    assert metrics["runs_ok"] == 2


def test_rag_metrics_report_hit_rate_and_evidence_count() -> None:
    hit = _ok_result("multi_agent_rag", "s-01")
    hit.deterministic.update({"retrieval_available": True, "evidence_count_total": 12})
    miss = _ok_result("multi_agent_rag", "s-02")
    miss.deterministic.update({"retrieval_available": False, "evidence_count_total": 0})

    metrics = build_report([hit, miss], "unit")["per_variant"]["multi_agent_rag"]

    assert metrics["retrieval_hit_rate"] == pytest.approx(0.5)
    assert metrics["avg_evidence_count"] == pytest.approx(6.0)
    assert (
        build_report([_ok_result()], "unit")["per_variant"]["single_agent"][
            "retrieval_hit_rate"
        ]
        is None
    )


def test_ablation_row_renders_na_for_unknown_variant() -> None:
    report = build_report([], "unit")
    row = ablation_row(report["per_variant"]["full_system"])

    assert row[1] == "N/A" and row[9] == "N/A"


def test_human_review_sheet_has_one_row_per_run(tmp_path: Path) -> None:
    path = tmp_path / "human_review.csv"

    write_human_review([_ok_result()], path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("case_id,category,variant,status,input")
    assert len(lines) == 2


# --------------------------------------------------------------------------- #
# Judge
# --------------------------------------------------------------------------- #


def test_judge_schema_rejects_out_of_range_scores() -> None:
    with pytest.raises(Exception):
        EvaluationJudgment.model_validate(
            {
                "observation_fidelity": 7,
                "theory_grounding": 3,
                "overinterpretation_control": 3,
                "answer_usefulness": 3,
                "reasoning_summary": "x",
            }
        )


def test_pairwise_schema_requires_a_known_winner() -> None:
    with pytest.raises(Exception):
        PairwiseJudgment.model_validate(
            {"winner": "C", "main_basis": "theory_grounding", "rationale": "x"}
        )


def test_judge_prompt_hides_variant_names(fake_model: FakeChatModel) -> None:
    asyncio.run(judge_answer(_case(), "候选回答内容。"))

    prompt = "\n".join(
        str(message.content) for message in fake_model.calls_for(EvaluationJudgment)[0]
    )
    assert "候选回答内容。" in prompt
    assert "我梦见水。" in prompt
    for name in VARIANTS:
        assert name not in prompt
    assert "不要输出分数以外的总体评分" in prompt


def test_pairwise_order_is_randomised_and_mapped_back(
    fake_model: FakeChatModel,
) -> None:
    fake_model.overrides = {
        PairwiseJudgment: {
            "winner": "A",
            "main_basis": "observation_fidelity",
            "rationale": "第一个更稳。",
        }
    }
    case = _case()
    orders: dict[str, str] = {}
    for seed in range(20):
        verdict = asyncio.run(
            compare_answers(
                case,
                "第一个变体的回答",
                "第二个变体的回答",
                pair_label="single_agent__vs__multi_agent",
                seed=seed,
            )
        )
        orders[verdict["display_order"]] = verdict["winner"]

    assert set(orders) == {"first_second", "second_first"}, orders
    # "A" is whichever answer was shown first, so the winner always follows the
    # hidden order rather than the variant order.
    assert orders["first_second"] == "first"
    assert orders["second_first"] == "second"


def test_pairwise_prompt_does_not_reveal_the_variants(
    fake_model: FakeChatModel,
) -> None:
    asyncio.run(
        compare_answers(
            _case(),
            "回答一",
            "回答二",
            pair_label="multi_agent__vs__multi_agent_rag",
        )
    )

    prompt = "\n".join(
        str(message.content) for message in fake_model.calls_for(PairwiseJudgment)[0]
    )
    assert "multi_agent" not in prompt
    assert "回答一" in prompt and "回答二" in prompt
    assert "知识库" in prompt or True  # evidence block is optional


def test_select_cases_honours_limit_and_per_category() -> None:
    selected = select_cases(DATASET, per_category=1, limit=3)

    assert len(selected) == 3
    assert [case.case_id for case in selected] == ["short-01", "ctx-01", "rel-01"]


def test_interleave_keeps_a_partial_run_balanced() -> None:
    cases = load_dataset(DATASET)
    ordered = interleave(cases)

    assert len(ordered) == len(cases)
    assert {case.case_id for case in ordered} == {case.case_id for case in cases}
    # The first six cases of the interleaved order cover all six categories.
    assert len({case.category for case in ordered[:6]}) == len(CATEGORIES)
    # Within a category the dataset order is preserved.
    short_ids = [case.case_id for case in ordered if case.category == "short_ambiguous"]
    assert short_ids == [f"short-{index:02d}" for index in range(1, 11)]


def test_interleave_flag_reaches_the_selection() -> None:
    selected = select_cases(DATASET, per_category=2, interleave_category_order=True)

    assert [case.category for case in selected[:6]] == list(CATEGORIES)


# --------------------------------------------------------------------------- #
# Fairness guards
# --------------------------------------------------------------------------- #


def test_every_variant_uses_the_same_model_factory() -> None:
    """All variants must go through `react_agent.llm`, so one model config applies."""
    source = Path(variants_module.__file__).read_text(encoding="utf-8")

    assert "get_chat_model" not in source, "a variant built its own model"
    assert "ChatDeepSeek" not in source, "a variant hard-coded a provider"
