"""Run the Phase 8 evaluation (variants A/B/C/D) against the real model.

Examples (from the project root):

    # smoke run: first two cases of every category, all variants, with the judge
    uv run --env-file .env python scripts/run_evaluation.py \
        --all --per-category 2 --judge --pairwise --concurrency 2

    # one variant only, five cases
    uv run --env-file .env python scripts/run_evaluation.py \
        --variant full_system --limit 5 --judge

    # full dataset, resume-friendly; add --rerun to ignore earlier results
    uv run --env-file .env python scripts/run_evaluation.py --all --judge --pairwise

Artefacts land in `data/evals/<run-id>/` (git-ignored): `raw_results.jsonl`,
`pairwise.jsonl`, `report.json`, `summary.json`, `summary.csv`,
`ablation.md`, `human_review.csv`.

The script never prints credentials and never writes them to the artefacts.
"""

# Progress output is the intended deliverable of this CLI.
# ruff: noqa: T201

import argparse
import asyncio
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from react_agent.evaluation.aggregate import ABLATION_COLUMNS
from react_agent.evaluation.dataset import CATEGORIES, category_counts
from react_agent.evaluation.experiment import (
    ExperimentConfig,
    run_experiment,
    select_cases,
)
from react_agent.evaluation.variants import VARIANTS, VariantName

DEFAULT_DATASET = Path("evals/dataset.jsonl")
DEFAULT_OUTPUT = Path("data/evals")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Optional argument list (used by tests).

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        action="append",
        choices=list(VARIANTS),
        help="variant to run; repeatable (default: all four)",
    )
    parser.add_argument("--all", action="store_true", help="run every variant")
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET, help="JSONL dataset path"
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUTPUT, help="output root folder"
    )
    parser.add_argument("--run-id", default=None, help="run folder name")
    parser.add_argument(
        "--limit", type=int, default=None, help="run only the first N selected cases"
    )
    parser.add_argument(
        "--per-category",
        type=int,
        default=None,
        help="keep only the first N cases of every category (smoke run)",
    )
    parser.add_argument(
        "--category",
        action="append",
        choices=list(CATEGORIES),
        help="run only these categories; repeatable",
    )
    parser.add_argument(
        "--judge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="score answers with the LLM judge (enabled by default)",
    )
    parser.add_argument(
        "--pairwise",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="also run the pairwise comparisons (A/B, B/C, C/D)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="how many (variant, case) runs may be in flight (default 1)",
    )
    parser.add_argument(
        "--interleave",
        action="store_true",
        help="round robin over categories, so a partial run stays balanced",
    )
    parser.add_argument(
        "--rerun", action="store_true", help="ignore existing results and rerun"
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="attempts per (variant, case) before recording a failure",
    )
    parser.add_argument(
        "--list-dataset",
        action="store_true",
        help="print dataset statistics and exit without calling the model",
    )
    return parser.parse_args(argv)


def _print_dataset_overview(dataset: Path) -> None:
    """Print dataset size and category distribution."""
    cases = select_cases(dataset)
    counts = category_counts(cases)
    print(f"dataset: {dataset} cases={len(cases)}")
    for category in CATEGORIES:
        print(f"  {category:<24} {counts[category]}")


def main(argv: list[str] | None = None) -> int:
    """Run the requested evaluation.

    Args:
        argv: Optional argument list (used by tests).

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    logging.disable(logging.CRITICAL)
    load_dotenv()
    os.environ.setdefault("LANGSMITH_TRACING", "false")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if args.list_dataset:
        _print_dataset_overview(args.dataset)
        return 0

    if not os.environ.get("DEEPSEEK_API_KEY", "").strip():
        print("Missing model credential; configure it locally first.")
        return 2

    variants: tuple[VariantName, ...] = (
        tuple(args.variant) if args.variant else VARIANTS
    )
    run_id = args.run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config = ExperimentConfig(
        run_id=run_id,
        dataset_path=args.dataset,
        output_root=args.out,
        variants=variants,
        judge=args.judge,
        pairwise=args.pairwise,
        concurrency=max(1, args.concurrency),
        rerun=args.rerun,
        per_category=args.per_category,
        limit=args.limit,
        categories=tuple(args.category) if args.category else None,
        interleave=args.interleave,
        max_attempts=max(1, args.max_attempts),
    )

    started = time.perf_counter()
    report = asyncio.run(run_experiment(config))
    elapsed = time.perf_counter() - started

    print("\n" + "=" * 100)
    print(" | ".join(ABLATION_COLUMNS))
    print("=" * 100)
    for row in report["ablation_table"]["rows"]:
        print(" | ".join(row))
    print("=" * 100)
    print(
        f"runs ok={report['runs_ok']} failed={report['runs_failed']} "
        f"wall_clock={elapsed / 60:.1f} min"
    )
    for item in report["failures"][:10]:
        print(
            f"  failed: {item['variant']} {item['case_id']} "
            f"{item['error_type']}: {item['error_message'][:120]}"
        )
    print(
        "\nN/A means the metric does not exist for that variant. "
        "These numbers are engineering observations, not medical evidence."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
