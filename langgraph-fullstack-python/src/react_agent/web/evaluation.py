"""Evaluation page: read a small, versioned summary of the Phase 8 run.

The page never re-runs the evaluation and never calls a model. It reads
`docs/evaluation_summary.json`, a trimmed artifact produced by
`scripts/export_evaluation_summary.py` from the stored Phase 8 reports, so a
deployment does not depend on the git-ignored `data/evals/` folder existing.

Only aggregate numbers are stored there: no user input, no model output, no API
information. When the file is missing the page says so instead of inventing
numbers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from fasthtml.common import (
    H1,
    A,
    Div,
    P,
    Table,
    Tbody,
    Td,
    Th,
    Thead,
    Tr,
)

from react_agent.web.nodes import FT, children

DEFAULT_SUMMARY_PATH = Path("docs/evaluation_summary.json")
"""Where the export script writes the artifact, relative to the project root."""

DISCLAIMER = (
    "这些指标评估的是 Agent 的工程行为：Grounding、Citation Integrity、"
    "Theory Differentiation、Safety Boundary、Latency。"
    "它们不代表心理诊断有效性，也不代表精神分析理论的医学有效性或临床治疗效果。"
)
"""Explicit limits of the numbers, shown on the page."""

CORPUS_NOTE = (
    "Phase 8 的 RAG 机制评测使用项目自编测试语料（TEST FIXTURE），"
    "不是 Freud / Klein / Lacan 原著数据库。"
)


@dataclass(frozen=True)
class MetricRow:
    """One metric row of the comparison table.

    Attributes:
        key: Metric identifier from the report.
        label: Reader-facing name.
        values: One value per variant, in variant order; None means N/A.
        unit: `score`, `seconds`, `count` or `rate`.
    """

    key: str
    label: str
    values: tuple[float | None, ...]
    unit: str


@dataclass(frozen=True)
class VariantSummary:
    """Aggregated numbers of one evaluation variant.

    Attributes:
        key: Variant identifier (`single_agent`, ...).
        label: Reader-facing name.
        runs: Number of stored runs.
        metrics: Mapping from metric key to value.
    """

    key: str
    label: str
    runs: int
    metrics: Mapping[str, float | None]


@dataclass(frozen=True)
class EvaluationSummary:
    """The whole artifact.

    Attributes:
        schema_version: Artifact version.
        generated_at: ISO timestamp of the export.
        source: Report file the numbers were read from.
        dataset_cases: Number of dataset cases behind the run.
        rescored: Whether the source report was recomputed with the current
            metric code.
        variants: Per-variant aggregates.
        notes: Free-form caveats carried by the artifact.
    """

    schema_version: int
    generated_at: str
    source: str
    dataset_cases: int
    rescored: bool
    variants: tuple[VariantSummary, ...]
    notes: tuple[str, ...] = ()


VARIANT_LABELS: Mapping[str, str] = {
    "single_agent": "Single Agent",
    "multi_agent": "Multi Agent",
    "multi_agent_rag": "Multi Agent + RAG",
    "full_system": "Full System (Critic)",
}
"""Display names of the four Phase 8 variants."""

METRIC_SPECS: tuple[tuple[str, str, str], ...] = (
    ("avg_llm_calls", "平均 LLM 调用次数", "count"),
    ("avg_latency_seconds", "平均延迟（秒）", "seconds"),
    ("avg_total_tokens", "平均 token 用量", "count"),
    ("theory_differentiation_judge", "Theory Differentiation（1–5）", "score"),
    ("observation_fidelity_judge", "Observation Fidelity（1–5）", "score"),
    ("overinterpretation_control_judge", "Overinterpretation Control（1–5）", "score"),
    ("clinical_boundary_judge", "Clinical Boundary（1–5）", "score"),
    ("citation_id_validity", "Citation Validity（0–1）", "rate"),
    ("citation_support_judge", "Citation Support（1–5）", "score"),
    ("diagnosis_refusal_rate", "临床拒绝率（0–1）", "rate"),
    ("critic_pass_rate", "Critic 一次通过率（0–1）", "rate"),
    ("revision_rate", "Revision 触发率（0–1）", "rate"),
    ("safe_fallback_rate", "Safe Fallback 率（0–1）", "rate"),
)
"""Metrics shown on the page, in display order."""

NA_TEXT = "N/A"
"""Shown when a metric does not apply to a variant (never `0`)."""


def format_value(value: float | None, unit: str) -> str:
    """Format one metric value.

    Args:
        value: Numeric value or None.
        unit: `score`, `seconds`, `count` or `rate`.

    Returns:
        Display text.
    """
    if value is None:
        return NA_TEXT
    if unit == "seconds":
        return f"{value:.1f}s"
    if unit == "count":
        return (
            f"{value:,.0f}" if value >= 100 else f"{value:.2f}".rstrip("0").rstrip(".")
        )
    return f"{value:.2f}"


def parse_summary(payload: Any) -> EvaluationSummary | None:
    """Validate a summary payload.

    Args:
        payload: Parsed JSON value.

    Returns:
        The summary, or None when the payload is not usable.
    """
    if not isinstance(payload, Mapping):
        return None
    raw_variants = payload.get("variants")
    if not isinstance(raw_variants, Sequence) or isinstance(raw_variants, (str, bytes)):
        return None
    variants: list[VariantSummary] = []
    for entry in raw_variants:
        if not isinstance(entry, Mapping):
            return None
        key = entry.get("key")
        if not isinstance(key, str) or not key:
            return None
        metrics_raw = entry.get("metrics")
        metrics: dict[str, float | None] = {}
        if isinstance(metrics_raw, Mapping):
            for name, value in metrics_raw.items():
                metrics[str(name)] = (
                    float(value)
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                    else None
                )
        variants.append(
            VariantSummary(
                key=key,
                label=str(entry.get("label") or VARIANT_LABELS.get(key, key)),
                runs=int(entry.get("runs") or 0),
                metrics=metrics,
            )
        )
    if not variants:
        return None
    notes_raw = payload.get("notes")
    notes = (
        tuple(str(item) for item in notes_raw)
        if isinstance(notes_raw, Sequence) and not isinstance(notes_raw, (str, bytes))
        else ()
    )
    return EvaluationSummary(
        schema_version=int(payload.get("schema_version") or 1),
        generated_at=str(payload.get("generated_at") or ""),
        source=str(payload.get("source") or ""),
        dataset_cases=int(payload.get("dataset_cases") or 0),
        rescored=bool(payload.get("rescored")),
        variants=tuple(variants),
        notes=notes,
    )


def load_evaluation_summary(path: Path | None = None) -> EvaluationSummary | None:
    """Load the evaluation summary artifact.

    Args:
        path: Optional explicit path; the default artifact is used otherwise.

    Returns:
        The summary, or None when the artifact is missing or unreadable.
    """
    target = path or DEFAULT_SUMMARY_PATH
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return parse_summary(payload)


def metric_rows(summary: EvaluationSummary) -> list[MetricRow]:
    """Build the comparison rows of the summary.

    Args:
        summary: Loaded summary.

    Returns:
        One row per metric, in display order.
    """
    rows: list[MetricRow] = []
    for key, label, unit in METRIC_SPECS:
        values = tuple(variant.metrics.get(key) for variant in summary.variants)
        if all(value is None for value in values):
            continue
        rows.append(MetricRow(key=key, label=label, values=values, unit=unit))
    return rows


def comparison_table(summary: EvaluationSummary) -> FT:
    """Render the metric comparison table.

    Args:
        summary: Loaded summary.

    Returns:
        A table node.
    """
    header = Tr(
        Th("指标", scope="col"),
        *[Th(variant.label, scope="col") for variant in summary.variants],
    )
    rows = [
        Tr(
            Th(row.label, scope="row"),
            *[Td(format_value(value, row.unit)) for value in row.values],
        )
        for row in metric_rows(summary)
    ]
    run_row = Tr(
        Th("已完成 runs", scope="row"),
        *[Td(str(variant.runs)) for variant in summary.variants],
    )
    return Table(
        Thead(header),
        Tbody(run_row, *rows),
        cls="eval-table",
    )


def evaluation_page(summary: EvaluationSummary | None) -> FT:
    """Render the System / Evaluation page.

    Args:
        summary: Loaded summary, or None when the artifact is missing.

    Returns:
        The page body.
    """
    if summary is None:
        body: list[FT] = [
            P(
                "没有找到评估摘要 (docs/evaluation_summary.json)。"
                "运行 scripts/export_evaluation_summary.py 可以从 Phase 8 的结果生成它。",
                cls="ev-empty",
            )
        ]
    else:
        source_note = f"数据来源：{summary.source}" + (
            "（已用当前指标代码重算）" if summary.rescored else "（原始报告）"
        )
        body = [
            P(source_note, cls="panel-hint"),
            P(
                f"导出于 {summary.generated_at} · 数据集 {summary.dataset_cases} 条用例",
                cls="panel-hint",
            ),
            comparison_table(summary),
            P("N/A 表示该指标对该变体不适用，并不等于 0。", cls="panel-hint"),
        ]
    return Div(
        children(
            Div(
                children(
                    H1("System / Evaluation"),
                    P(
                        "Phase 8 的 ablation 结果：四个变体的质量、延迟与调用成本对比。",
                        cls="panel-hint",
                    ),
                ),
                cls="chat-head",
            ),
            Div(
                children(
                    *body,
                    P(DISCLAIMER, cls="eval-disclaimer"),
                    P(CORPUS_NOTE, cls="eval-note"),
                    A("返回对话", href="/", cls="btn-new"),
                ),
                cls="eval-body",
            ),
        ),
        cls="col col-eval",
    )
