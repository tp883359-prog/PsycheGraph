r"""Export a small, versioned evaluation summary for the web page.

The Phase 8 artifacts live in `data/evals/`, which is git-ignored runtime data:
a deployment cannot rely on it existing. This script writes the trimmed,
aggregate-only file the System / Evaluation page reads instead:

    uv run python scripts/export_evaluation_summary.py
    uv run python scripts/export_evaluation_summary.py --run-dir data/evals/full \\
        --out docs/evaluation_summary.json

What goes in: per-variant averages for the metrics the page shows (calls,
latency, tokens, judge scores, citation validity, refusal rate, critic rates).
What never goes in: user input, model output, prompts, API information - only
numbers from the stored report and a couple of fixed notes.
"""

# Progress output is the intended deliverable of this CLI.
# ruff: noqa: T201

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from react_agent.web.evaluation import METRIC_SPECS, VARIANT_LABELS

DEFAULT_RUN_DIR = Path("data/evals/full")
DEFAULT_OUT = Path("docs/evaluation_summary.json")
DEFAULT_DATASET = Path("evals/dataset.jsonl")

REPORT_CANDIDATES = ("report_rescored.json", "report.json")
"""Preferred report file: the one recomputed with the current metric code."""

VARIANT_ORDER = ("single_agent", "multi_agent", "multi_agent_rag", "full_system")
"""Display order of the four Phase 8 variants."""

NOTES = (
    "聚合指标，来自 Phase 8 的完整评测运行（60 条用例 × 4 个变体）。",
    "RAG 相关指标使用项目自编测试语料（TEST FIXTURE）。",
    "N/A 表示该指标对该变体不适用，并不等于 0。",
)
"""Fixed caveats carried by the artifact."""


def pick_report(run_dir: Path) -> tuple[Path, dict[str, Any]] | None:
    """Pick the newest usable report file of a run directory.

    Args:
        run_dir: Directory holding the Phase 8 artifacts.

    Returns:
        The path and parsed payload, or None when nothing usable exists.
    """
    for name in REPORT_CANDIDATES:
        candidate = run_dir / name
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("per_variant"), dict):
            return candidate, payload
    return None


def number_or_none(value: Any) -> float | None:
    """Return a float for numeric values and None otherwise.

    Args:
        value: Raw value from the report.

    Returns:
        The number, or None. Strings are never copied into the artifact.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return round(float(value), 4)
    return None


def variant_summary(key: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Build one variant entry of the artifact.

    Args:
        key: Variant key.
        payload: The variant's `per_variant` record.

    Returns:
        A trimmed entry with only whitelisted numbers.
    """
    metrics: dict[str, float | None] = {}
    for metric_key, _label, _unit in METRIC_SPECS:
        metrics[metric_key] = number_or_none(payload.get(metric_key))
    runs = number_or_none(payload.get("runs_ok"))
    return {
        "key": key,
        "label": VARIANT_LABELS.get(key, key),
        "runs": int(runs or 0),
        "metrics": metrics,
    }


def dataset_size(dataset_path: Path) -> int:
    """Count the cases of the dataset file.

    Args:
        dataset_path: Path to `evals/dataset.jsonl`.

    Returns:
        Number of non-empty lines, or 0 when the file is missing.
    """
    try:
        text = dataset_path.read_text(encoding="utf-8")
    except OSError:
        return 0
    return sum(1 for line in text.splitlines() if line.strip())


def build_summary(
    report_path: Path,
    report: Mapping[str, Any],
    *,
    run_dir: Path,
    dataset_cases: int,
) -> dict[str, Any]:
    """Build the artifact payload.

    Args:
        report_path: Report file the numbers came from.
        report: Parsed report.
        run_dir: Run directory the report belongs to.
        dataset_cases: Number of dataset cases used by the run.

    Returns:
        The artifact payload.
    """
    per_variant = report.get("per_variant") or {}
    variants: list[dict[str, Any]] = []
    for key in VARIANT_ORDER:
        payload = per_variant.get(key)
        if isinstance(payload, Mapping):
            variants.append(variant_summary(key, payload))
    for key, payload in per_variant.items():
        if key not in VARIANT_ORDER and isinstance(payload, Mapping):
            variants.append(variant_summary(str(key), payload))
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": f"{run_dir.as_posix()}/{report_path.name}",
        "rescored": report_path.name.endswith("_rescored.json"),
        "dataset_cases": dataset_cases,
        "run_id": str(report.get("run_id") or run_dir.name),
        "runs_total": number_or_none(report.get("runs_total")),
        "runs_failed": number_or_none(report.get("runs_failed")),
        "variants": variants,
        "notes": list(NOTES),
    }


def main() -> int:
    """Write the evaluation summary artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()

    picked = pick_report(args.run_dir)
    if picked is None:
        print(
            f"no usable report in {args.run_dir} "
            f"(expected one of: {', '.join(REPORT_CANDIDATES)})"
        )
        print("run `uv run python scripts/run_evaluation.py --full` first")
        return 1
    report_path, report = picked
    summary = build_summary(
        report_path,
        report,
        run_dir=args.run_dir,
        dataset_cases=dataset_size(args.dataset),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out}")
    print(f"source: {summary['source']} (rescored={summary['rescored']})")
    print(f"dataset cases: {summary['dataset_cases']}")
    for variant in summary["variants"]:
        metrics = variant["metrics"]
        print(
            f"  {variant['label']:<22} runs={variant['runs']:<3} "
            f"calls={metrics.get('avg_llm_calls')} "
            f"latency={metrics.get('avg_latency_seconds')} "
            f"tokens={metrics.get('avg_total_tokens')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
