"""Evaluation experiment orchestration: resume, concurrency, artefacts.

Layout of one run (`data/evals/<run_id>/`):

    raw_results.jsonl   one JSON object per (variant, case); appended as soon as
                        the run finishes, so a crash never loses earlier work
    pairwise.jsonl      one JSON object per (pair, case) comparison
    report.json         aggregated metrics, ablation table, failure list
    summary.json        the ablation table plus the headline numbers
    summary.csv         the same table as CSV (for plotting)
    human_review.csv    case / variant / answer / judge scores for manual review

Resume is per (variant, case): a successful record is skipped unless `--rerun`
is given. Failed records are retried on the next start, because the most common
cause is a transient provider error.
"""

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from react_agent.evaluation.aggregate import (
    PAIRWISE_PAIRS,
    build_report,
    write_ablation_markdown,
    write_human_review,
    write_summary_csv,
)
from react_agent.evaluation.dataset import (
    EvalCase,
    interleave,
    load_dataset,
    smoke_subset,
)
from react_agent.evaluation.judge import compare_answers
from react_agent.evaluation.runner import EvalResult, run_case
from react_agent.evaluation.usage import install_usage_recorder
from react_agent.evaluation.variants import VARIANTS, VariantName


def _log(message: str) -> None:
    """Print one progress line (the runner's output is the deliverable)."""
    print(message, flush=True)  # noqa: T201 - progress output of the evaluation CLI


@dataclass
class ExperimentConfig:
    """Everything a run needs.

    Attributes:
        run_id: Folder name under `output_root`.
        dataset_path: JSONL dataset.
        output_root: Root folder for run artefacts (git-ignored `data/evals`).
        variants: Variants to run.
        judge: Whether to score answers with the judge.
        pairwise: Whether to run the pairwise comparisons.
        concurrency: How many (variant, case) runs may be in flight at once.
        rerun: Ignore existing results and rerun everything.
        per_category: Keep only the first N cases of every category (smoke run).
        limit: Keep only the first N cases of the selection.
        max_attempts: Attempts per (variant, case) before recording a failure.
    """

    run_id: str
    dataset_path: Path
    output_root: Path
    variants: tuple[VariantName, ...] = VARIANTS
    judge: bool = True
    pairwise: bool = False
    concurrency: int = 1
    rerun: bool = False
    per_category: int | None = None
    limit: int | None = None
    categories: tuple[str, ...] | None = None
    interleave: bool = False
    max_attempts: int = 2
    judge_retry_attempts: int = 1
    extra: dict[str, Any] = field(default_factory=dict)


def result_key(result: EvalResult) -> str:
    """Return the resume key of one result."""
    return f"{result.variant}::{result.case_id}"


def load_results(path: Path) -> dict[str, EvalResult]:
    """Load previously saved results, keeping one per resume key.

    Args:
        path: `raw_results.jsonl` of an earlier run.

    Returns:
        Mapping of `variant::case_id` to the last stored record. Records written
        by a crashed process are ignored (they are not valid JSON).
    """
    stored: dict[str, EvalResult] = {}
    if not path.is_file():
        return stored
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        record = EvalResult.model_validate(payload)
        stored[result_key(record)] = record
    return stored


def load_pairwise(path: Path) -> set[str]:
    """Return the `case_id::pair` keys already compared."""
    done: set[str] = set()
    if not path.is_file():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        done.add(f"{payload.get('case_id')}::{payload.get('pair')}")
    return done


def select_cases(
    dataset_path: Path,
    *,
    per_category: int | None = None,
    limit: int | None = None,
    categories: tuple[str, ...] | None = None,
    interleave_category_order: bool = False,
) -> list[EvalCase]:
    """Load the dataset and apply the category / smoke / limit / order selection.

    Args:
        dataset_path: JSONL dataset.
        per_category: Keep the first N cases of every category.
        limit: Keep only the first N cases afterwards.
        categories: Keep only these categories (debugging a single family).
        interleave_category_order: Round robin over categories, so a partial run
            stays balanced instead of covering only the first families.

    Returns:
        The selected cases.
    """
    cases = load_dataset(dataset_path)
    if categories:
        wanted = set(categories)
        cases = [case for case in cases if case.category in wanted]
    if per_category:
        cases = smoke_subset(cases, per_category)
    if interleave_category_order:
        cases = interleave(cases)
    if limit:
        cases = cases[:limit]
    return cases


async def _run_and_store(
    case: EvalCase,
    variant: VariantName,
    config: ExperimentConfig,
    path: Path,
    lock: asyncio.Lock,
    counters: dict[str, int],
) -> EvalResult:
    """Run one pair and append its record immediately."""
    result = await run_case(
        case,
        variant,
        run_id=config.run_id,
        judge=config.judge,
        max_attempts=config.max_attempts,
    )
    async with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(result.model_dump_json() + "\n")
        counters["done"] += 1
        status = "ok" if result.status == "ok" else f"FAILED({result.error_type})"
        _log(
            f"[{counters['done']}/{counters['total']}] {variant:<15} {case.case_id:<22} "
            f"{status:<22} calls={result.llm_calls} latency={result.latency_seconds:.1f}s"
        )
    return result


async def _pairwise_stage(
    cases: list[EvalCase],
    results: dict[str, EvalResult],
    config: ExperimentConfig,
    path: Path,
) -> list[dict[str, Any]]:
    """Compare the configured pairs for every case that succeeded.

    Args:
        cases: Cases of this run.
        results: Stored results keyed by `variant::case_id`.
        config: Experiment configuration.
        path: `pairwise.jsonl` path.

    Returns:
        The pairwise records (new and previously stored).
    """
    done = load_pairwise(path)
    records: list[dict[str, Any]] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    for case in cases:
        for first, second in PAIRWISE_PAIRS:
            if first not in config.variants or second not in config.variants:
                continue
            pair_label = f"{first}__vs__{second}"
            key = f"{case.case_id}::{pair_label}"
            if key in done:
                continue
            left = results.get(f"{first}::{case.case_id}")
            right = results.get(f"{second}::{case.case_id}")
            if (
                left is None
                or right is None
                or left.status != "ok"
                or right.status != "ok"
            ):
                continue
            try:
                verdict = await compare_answers(
                    case, left.answer, right.answer, pair_label=pair_label
                )
            except Exception as error:  # noqa: BLE001 - recorded like any failure
                verdict = {
                    "pair": pair_label,
                    "winner": "error",
                    "main_basis": "answer_usefulness",
                    "rationale": f"{type(error).__name__}",
                    "display_order": "unknown",
                }
            record = {
                "case_id": case.case_id,
                "category": case.category,
                **verdict,
            }
            records.append(record)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            _log(
                f"  pairwise {pair_label:<40} {case.case_id:<22} "
                f"winner={record['winner']}"
            )
    return records


async def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """Run an evaluation experiment and write its artefacts.

    Args:
        config: Experiment configuration.

    Returns:
        The aggregated report (also written to `report.json`).
    """
    install_usage_recorder()
    run_dir = config.output_root / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / "raw_results.jsonl"
    pairwise_path = run_dir / "pairwise.jsonl"

    cases = select_cases(
        config.dataset_path,
        per_category=config.per_category,
        limit=config.limit,
        categories=config.categories,
        interleave_category_order=config.interleave,
    )
    stored = {} if config.rerun else load_results(raw_path)
    if config.rerun and raw_path.is_file():
        raw_path.unlink()

    pending: list[tuple[EvalCase, VariantName]] = [
        (case, variant)
        for case in cases
        for variant in config.variants
        if f"{variant}::{case.case_id}" not in stored
        or stored[f"{variant}::{case.case_id}"].status != "ok"
    ]
    _log(
        f"run={config.run_id} cases={len(cases)} variants={len(config.variants)} "
        f"pending={len(pending)} resumed={len(stored)}"
    )

    lock = asyncio.Lock()
    counters = {"done": 0, "total": len(pending)}
    semaphore = asyncio.Semaphore(max(1, config.concurrency))

    async def worker(case: EvalCase, variant: VariantName) -> EvalResult:
        async with semaphore:
            return await _run_and_store(case, variant, config, raw_path, lock, counters)

    if pending:
        results = await asyncio.gather(
            *(worker(case, variant) for case, variant in pending)
        )
        for result in results:
            stored[result_key(result)] = result

    pairwise_records: list[dict[str, Any]] = []
    if config.pairwise and config.judge:
        pairwise_records = await _pairwise_stage(cases, stored, config, pairwise_path)

    ordered = [stored[key] for key in sorted(stored)]
    report = build_report(ordered, config.run_id, pairwise_records)
    report["config"] = {
        "dataset": str(config.dataset_path),
        "cases": len(cases),
        "variants": list(config.variants),
        "judge": config.judge,
        "pairwise": config.pairwise,
        "concurrency": config.concurrency,
        "per_category": config.per_category,
        "limit": config.limit,
        "categories": list(config.categories) if config.categories else None,
        "interleave": config.interleave,
    }

    (run_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "run_id": config.run_id,
                "cases": [
                    {
                        "case_id": item.case_id,
                        "category": item.category,
                        "variant": item.variant,
                        "status": item.status,
                        "llm_calls": item.llm_calls,
                        "latency_seconds": item.latency_seconds,
                        "tokens": item.tokens,
                        "deterministic": item.deterministic,
                        "judge": item.judge,
                    }
                    for item in ordered
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": report["run_id"],
                "config": report["config"],
                "runs_total": report["runs_total"],
                "runs_ok": report["runs_ok"],
                "runs_failed": report["runs_failed"],
                "ablation_table": report["ablation_table"],
                "per_variant": report["per_variant"],
                "by_category": report["by_category"],
                "failures": report["failures"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_summary_csv(report, run_dir / "summary.csv")
    write_ablation_markdown(report, run_dir / "ablation.md")
    write_human_review(ordered, run_dir / "human_review.csv")
    _log(f"artefacts written to {run_dir}")
    return report
