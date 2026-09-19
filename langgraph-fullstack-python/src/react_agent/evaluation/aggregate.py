"""Aggregate evaluation results into tables, JSON reports and a review sheet.

Two rules shape every number here:

1. **N/A is not 0.** A metric that does not exist for a variant (citation
   validity without retrieval, revision rate without a Critic) is excluded from
   averages and printed as `N/A`.
2. **Only successful runs enter quality averages.** Failures are counted and
   reported separately, so a crashed run never improves a mean.
"""

import csv
import json
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from react_agent.evaluation.dataset import CATEGORIES
from react_agent.evaluation.judge import JudgeDimension
from react_agent.evaluation.runner import EvalResult
from react_agent.evaluation.variants import VARIANTS

ABLATION_COLUMNS = (
    "Variant",
    "ObsFidelity(judge)",
    "ObsViolationRate(det)",
    "TheoryDiff(judge)",
    "CitationValidity(det)",
    "CitationMetaViolation(det)",
    "CitationSupport(judge)",
    "OverinterpControl(judge)",
    "ClinicalBoundary(judge)",
    "DiagnosisRefusalRate(det)",
    "AvgLLMCalls",
    "AvgLatencyS",
    "RunsOK",
    "Failures",
)
"""Columns of the ablation table (`avg_latency_seconds` is truncated in CSV)."""

PAIRWISE_PAIRS: tuple[tuple[str, str], ...] = (
    ("single_agent", "multi_agent"),
    ("multi_agent", "multi_agent_rag"),
    ("multi_agent_rag", "full_system"),
)
"""Comparisons that isolate one change at a time: A/B, B/C, C/D."""


def _mean(values: Iterable[float | None]) -> float | None:
    """Return the mean of the defined values, or None when none are defined."""
    defined = [value for value in values if value is not None]
    if not defined:
        return None
    return fmean(defined)


def _float_or_none(value: Any) -> float | None:
    """Return a numeric value as float, or None when it is not a number."""
    return (
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else None
    )


def _format(value: float | None, digits: int = 2) -> str:
    """Render a number for a table, using `N/A` for undefined values."""
    if value is None:
        return "N/A"
    if digits == 0:
        return f"{value:.0f}"
    return f"{value:.{digits}f}"


def judge_scores(
    results: list[EvalResult], dimension: JudgeDimension
) -> list[int | None]:
    """Return one judge dimension across results (judge errors become None)."""
    scores: list[int | None] = []
    for result in results:
        payload = result.judge or {}
        value = payload.get(dimension)
        scores.append(int(value) if isinstance(value, int) else None)
    return scores


def variant_metrics(results: list[EvalResult], variant: str) -> dict[str, Any]:
    """Aggregate every reported metric for one variant.

    Args:
        results: All results of the run (every variant).
        variant: Variant name to aggregate.

    Returns:
        A JSON-ready dict with means, rates, counts and N/A bookkeeping.
    """
    runs = [item for item in results if item.variant == variant]
    ok = [item for item in runs if item.status == "ok"]
    failed = [item for item in runs if item.status != "ok"]
    deterministic = [item.deterministic for item in ok]

    citation_validity = [item.get("citation_id_validity") for item in deterministic]
    cited_runs = [
        item for item in ok if (item.deterministic.get("cited_total") or 0) > 0
    ]

    diagnosis_runs = [item for item in ok if item.category == "clinical_boundary"]
    diagnosis_pass = [
        item.deterministic.get("diagnosis_boundary_pass") for item in diagnosis_runs
    ]

    retrieval_latency = [item.deterministic.get("retrieval_latency") for item in ok]

    rag_runs = [
        item for item in ok if item.deterministic.get("retrieval_latency") is not None
    ]

    critic_issues = [
        item.critique.get("issue_count") for item in ok if item.critique is not None
    ]
    categories: Counter[str] = Counter()
    for item in ok:
        if item.critique is None:
            continue
        categories.update(item.critique.get("categories") or [])

    judge_grounding = judge_scores(ok, "theory_grounding")
    citation_support = [
        score
        for item, score in zip(ok, judge_grounding, strict=True)
        if (item.deterministic.get("cited_total") or 0) > 0
    ]

    return {
        "variant": variant,
        "runs_total": len(runs),
        "runs_ok": len(ok),
        "runs_failed": len(failed),
        "observation_fidelity_judge": _mean(judge_scores(ok, "observation_fidelity")),
        "observation_fidelity_ok_rate": _mean(
            [
                1.0 if item.get("observation_fidelity_ok") else 0.0
                for item in deterministic
            ]
        ),
        "unprovided_fact_violation_rate": _mean(
            [
                1.0 if (item.get("unprovided_fact_violation") or 0) > 0 else 0.0
                for item in deterministic
            ]
        ),
        "theory_differentiation_judge": _mean(
            judge_scores(ok, "theory_differentiation")
        ),
        "theory_grounding_judge": _mean(judge_grounding),
        "overinterpretation_control_judge": _mean(
            judge_scores(ok, "overinterpretation_control")
        ),
        "clinical_boundary_judge": _mean(judge_scores(ok, "clinical_boundary")),
        "answer_usefulness_judge": _mean(judge_scores(ok, "answer_usefulness")),
        "citation_id_validity": _mean(citation_validity),
        "citation_runs_with_citations": len(cited_runs),
        "citation_metadata_violation_rate": _mean(
            [
                1.0 if item.get("citation_metadata_ok") is False else 0.0
                for item in deterministic
            ]
        ),
        "citation_support_judge": _mean(citation_support),
        "diagnosis_refusal_rate": _mean(
            [1.0 if value else 0.0 for value in diagnosis_pass]
        ),
        "diagnosis_runs": len(diagnosis_runs),
        "schema_valid_rate": _mean(
            [1.0 if item.get("schema_valid") else 0.0 for item in deterministic]
        ),
        "avg_llm_calls": _mean([float(item.llm_calls) for item in ok]),
        "avg_latency_seconds": _mean([item.latency_seconds for item in ok]),
        "avg_retrieval_latency": _mean(retrieval_latency),
        "avg_critic_latency": _mean(
            [item.deterministic.get("critic_latency") for item in ok]
        ),
        "structured_output_retries": sum(
            int(item.deterministic.get("structured_output_retries") or 0) for item in ok
        ),
        "revision_rate": _mean(
            [
                1.0 if item.get("revision_triggered") else 0.0
                for item in deterministic
                if item.get("revision_triggered") is not None
            ]
        ),
        "safe_fallback_rate": _mean(
            [
                1.0 if item.get("safe_fallback_triggered") else 0.0
                for item in deterministic
                if item.get("safe_fallback_triggered") is not None
            ]
        ),
        "critic_pass_rate": (
            _mean(
                [
                    1.0 if (item.critique or {}).get("verdict") == "pass" else 0.0
                    for item in ok
                    if item.critique is not None
                ]
            )
        ),
        "avg_critic_issues": _mean([_float_or_none(value) for value in critic_issues]),
        "critic_issue_categories": dict(sorted(categories.items())),
        "retrieval_runs": len(rag_runs),
        "retrieval_hit_rate": _mean(
            [
                1.0 if item.get("retrieval_available") else 0.0
                for item in deterministic
                if item.get("retrieval_available") is not None
            ]
        ),
        "avg_evidence_count": _mean(
            [
                _float_or_none(item.get("evidence_count_total"))
                for item in deterministic
                if item.get("evidence_count_total") is not None
            ]
        ),
        "avg_retrieval_calls": _mean(
            [
                _float_or_none(item.get("retrieval_calls"))
                for item in deterministic
                if item.get("retrieval_calls") is not None
            ]
        ),
        "token_usage_available": (
            all(item.token_usage_available for item in ok) if ok else False
        ),
        "avg_total_tokens": _mean(
            [_float_or_none(item.tokens.get("total_tokens")) for item in ok]
        ),
    }


def pairwise_stats(
    pairwise_records: list[dict[str, Any]], variant: str
) -> dict[str, int]:
    """Count pairwise wins, losses and ties for one variant.

    Args:
        pairwise_records: Records written by the pairwise stage, each with
            `pair` (`"a__vs__b"`) and `winner` (`"first"`, `"second"` or `"tie"`).
        variant: Variant to count for.

    Returns:
        `wins`, `losses`, `ties` and `comparisons` for that variant.
    """
    wins = losses = ties = comparisons = 0
    for record in pairwise_records:
        pair = tuple(str(record.get("pair", "")).split("__vs__"))
        if len(pair) != 2 or variant not in pair:
            continue
        comparisons += 1
        winner = record.get("winner")
        if winner == "tie":
            ties += 1
        elif winner == "first":
            if pair[0] == variant:
                wins += 1
            else:
                losses += 1
        elif winner == "second":
            if pair[1] == variant:
                wins += 1
            else:
                losses += 1
    return {"wins": wins, "losses": losses, "ties": ties, "comparisons": comparisons}


def ablation_row(metrics: dict[str, Any]) -> list[str]:
    """Render one ablation table row from aggregated metrics."""
    return [
        str(metrics["variant"]),
        _format(metrics["observation_fidelity_judge"]),
        _format(metrics["unprovided_fact_violation_rate"]),
        _format(metrics["theory_differentiation_judge"]),
        _format(metrics["citation_id_validity"]),
        _format(metrics["citation_metadata_violation_rate"]),
        _format(metrics["citation_support_judge"]),
        _format(metrics["overinterpretation_control_judge"]),
        _format(metrics["clinical_boundary_judge"]),
        _format(metrics["diagnosis_refusal_rate"]),
        _format(metrics["avg_llm_calls"]),
        _format(metrics["avg_latency_seconds"], digits=1),
        str(metrics["runs_ok"]),
        str(metrics["runs_failed"]),
    ]


def build_report(
    results: list[EvalResult],
    run_id: str,
    pairwise_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the full aggregated report of one evaluation run.

    Args:
        results: Every result of the run, in execution order.
        run_id: Evaluation run identifier.
        pairwise_records: Records of the pairwise stage, when it ran.

    Returns:
        A JSON-ready report with per-variant metrics, the ablation table and the
        category coverage.
    """
    records = pairwise_records or []
    per_variant = {name: variant_metrics(results, name) for name in VARIANTS}
    for name in VARIANTS:
        per_variant[name]["pairwise"] = pairwise_stats(records, name)
    rows = [ablation_row(per_variant[name]) for name in VARIANTS]

    by_category: dict[str, dict[str, int]] = {}
    for category in CATEGORIES:
        ok = [
            item
            for item in results
            if item.category == category and item.status == "ok"
        ]
        by_category[category] = {
            "runs_ok": len(ok),
            "runs_total": len([item for item in results if item.category == category]),
        }

    failed = [
        {
            "variant": item.variant,
            "case_id": item.case_id,
            "error_type": item.error_type,
            "error_message": item.error_message,
            "attempts": item.attempts,
        }
        for item in results
        if item.status != "ok"
    ]

    return {
        "run_id": run_id,
        "runs_total": len(results),
        "runs_ok": len([item for item in results if item.status == "ok"]),
        "runs_failed": len(failed),
        "per_variant": per_variant,
        "ablation_table": {"columns": list(ABLATION_COLUMNS), "rows": rows},
        "pairwise_pairs": [f"{a}__vs__{b}" for a, b in PAIRWISE_PAIRS],
        "by_category": by_category,
        "failures": failed,
    }


def write_ablation_markdown(report: dict[str, Any], path: Path) -> None:
    """Write the ablation table as a markdown file next to the JSON report."""
    columns = report["ablation_table"]["columns"]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in report["ablation_table"]["rows"]:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append(
        "N/A = the metric does not exist for that variant (no retrieval, no Critic). "
        "It is never averaged in as 0."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_human_review(results: list[EvalResult], path: Path) -> None:
    """Write a CSV sheet for manual review.

    Args:
        results: Every result of the run.
        path: Output CSV path.
    """
    columns = [
        "case_id",
        "category",
        "variant",
        "status",
        "input",
        "final_response",
        "judge_scores",
        "judge_summary",
        "llm_calls",
        "latency_seconds",
        "revision_count",
        "citation_total",
        "unprovided_fact_violation",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for item in results:
            judge = item.judge or {}
            writer.writerow(
                {
                    "case_id": item.case_id,
                    "category": item.category,
                    "variant": item.variant,
                    "status": item.status,
                    "input": item.input_text,
                    "final_response": item.answer,
                    "judge_scores": json.dumps(
                        {
                            key: judge.get(key)
                            for key in (
                                "observation_fidelity",
                                "theory_grounding",
                                "theory_differentiation",
                                "overinterpretation_control",
                                "clinical_boundary",
                                "answer_usefulness",
                            )
                        },
                        ensure_ascii=False,
                    ),
                    "judge_summary": str(judge.get("reasoning_summary", ""))[:400],
                    "llm_calls": item.llm_calls,
                    "latency_seconds": f"{item.latency_seconds:.1f}",
                    "revision_count": item.deterministic.get("revision_count"),
                    "citation_total": item.deterministic.get("cited_total"),
                    "unprovided_fact_violation": item.deterministic.get(
                        "unprovided_fact_violation"
                    ),
                }
            )


def write_summary_csv(report: dict[str, Any], path: Path) -> None:
    """Write the ablation table as CSV for later plotting."""
    columns = report["ablation_table"]["columns"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(report["ablation_table"]["rows"])
