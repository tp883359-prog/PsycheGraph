"""Phase 8 evaluation framework.

A side-car harness: it builds four variants from the *production* agents, runs a
property-based dataset through them, and reports deterministic metrics, judge
rubric scores and pairwise comparisons. Nothing here changes the production
graph; `full_system` is the production graph object itself.
"""

from react_agent.evaluation.dataset import (
    CATEGORIES,
    DatasetError,
    EvalCase,
    ExpectedProperties,
    category_counts,
    load_dataset,
    smoke_subset,
)
from react_agent.evaluation.experiment import ExperimentConfig, run_experiment
from react_agent.evaluation.judge import (
    EvaluationJudgment,
    PairwiseJudgment,
    compare_answers,
    judge_answer,
)
from react_agent.evaluation.metrics import (
    asserted_terms,
    citation_report,
    compute_deterministic_metrics,
    diagnosis_boundary,
)
from react_agent.evaluation.runner import EvalResult, run_case, sanitize
from react_agent.evaluation.variants import VARIANTS, VariantName, get_variant

__all__ = [
    "CATEGORIES",
    "VARIANTS",
    "DatasetError",
    "EvalCase",
    "EvalResult",
    "EvaluationJudgment",
    "ExpectedProperties",
    "ExperimentConfig",
    "PairwiseJudgment",
    "VariantName",
    "asserted_terms",
    "category_counts",
    "citation_report",
    "compare_answers",
    "compute_deterministic_metrics",
    "diagnosis_boundary",
    "get_variant",
    "judge_answer",
    "load_dataset",
    "run_case",
    "run_experiment",
    "sanitize",
    "smoke_subset",
]
