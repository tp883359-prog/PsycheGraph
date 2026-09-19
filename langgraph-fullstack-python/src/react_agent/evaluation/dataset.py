"""Evaluation dataset contract for Phase 8.

The evaluation is **property-based**: an open psychoanalytic reading has no single
correct answer, so a case never stores a reference answer. It stores the
properties an acceptable answer must have, split into

    deterministic properties - checkable in code (refusals, forbidden facts,
                               citation ids, schema validity)
    judged properties        - the LLM judge scores the rest with a rubric

Privacy: every case is project-authored synthetic text. No real user material is
stored here, and the loader refuses obvious personal-data markers.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

EvalCategory = Literal[
    "short_ambiguous",
    "contextual_dream",
    "relationship_narrative",
    "literary_analysis",
    "clinical_boundary",
    "adversarial_grounding",
]
"""The six case families of the dataset.

- `short_ambiguous`: a few words only; tests over-interpretation control.
- `contextual_dream`: a dream with scene, people, feeling and associations;
  tests whether theory is actually applied to material.
- `relationship_narrative`: conflicts, distance, a sentence that keeps coming
  back; tests whether the three schools stay distinguishable.
- `literary_analysis`: a work or a character; tests theory application.
- `clinical_boundary`: the user asks for a diagnosis; tests the clinical limit.
- `adversarial_grounding`: little material plus an invitation to over-infer,
  invent citations or prove an illness; tests grounding and safety.
"""

CATEGORIES: tuple[EvalCategory, ...] = (
    "short_ambiguous",
    "contextual_dream",
    "relationship_narrative",
    "literary_analysis",
    "clinical_boundary",
    "adversarial_grounding",
)
"""Category order used by reports and by `--per-category` smoke sampling."""

PRIVACY_MARKERS = ("@", "身份证", "手机号", "电话：", "wxid", "真实姓名")
"""Markers rejected by the loader: the dataset must not hold real user material.

A plain email-like token is enough to catch accidental copy-paste of a real
conversation; the dataset itself uses no personal identifiers at all.
"""


class ExpectedProperties(BaseModel):
    """What an acceptable answer must (and must not) contain.

    Attributes:
        must_refuse_diagnosis: The answer must decline a clinical judgement and
            must not sneak one in after the disclaimer.
        must_not_assume: Facts the material does not contain. Their appearance in
            `observations` or in the user-facing answer is a deterministic
            violation.
        expected_perspectives: Schools the answer is expected to distinguish.
            Used by the judged `theory_differentiation` metric and by reports.
        requires_citation: The answer is expected to rely on retrieved passages;
            a run that cites nothing is reported as citation N/A rather than 0.
        notes: Short human note explaining what the case probes.
    """

    must_refuse_diagnosis: bool = False
    must_not_assume: list[str] = Field(default_factory=list)
    expected_perspectives: list[Literal["freudian", "object_relations", "lacanian"]] = (
        Field(default_factory=list)
    )
    requires_citation: bool = False
    notes: str = ""


class EvalCase(BaseModel):
    """One evaluation case.

    Attributes:
        case_id: Stable identifier, e.g. `short-001`.
        category: One of the six families.
        input: The user message sent to a variant.
        expected_properties: Property-based expectations (never a reference answer).
    """

    case_id: str
    category: EvalCategory
    input: str
    expected_properties: ExpectedProperties

    @field_validator("case_id", "input")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()


class DatasetError(ValueError):
    """Raised when the dataset file is malformed or inconsistent."""


def _validate_case_rules(case: EvalCase, line_number: int) -> None:
    """Check the per-category requirements of one case.

    Args:
        case: The parsed case.
        line_number: 1-based line in the JSONL file, used in error messages.

    Raises:
        DatasetError: If the case does not satisfy its category's contract.
    """
    where = f"{case.case_id} (line {line_number})"
    if (
        case.category == "clinical_boundary"
        and not case.expected_properties.must_refuse_diagnosis
    ):
        raise DatasetError(
            f"{where}: clinical_boundary cases must set must_refuse_diagnosis"
        )
    if case.category in {"short_ambiguous", "adversarial_grounding"}:
        if not case.expected_properties.must_not_assume:
            raise DatasetError(
                f"{where}: {case.category} cases must list forbidden assumptions"
            )
    if (
        case.category == "literary_analysis"
        and not case.expected_properties.expected_perspectives
    ):
        raise DatasetError(
            f"{where}: literary_analysis cases must name expected perspectives"
        )
    for term in case.expected_properties.must_not_assume:
        if not str(term).strip():
            raise DatasetError(f"{where}: must_not_assume contains a blank term")


def parse_case(payload: dict[str, object], line_number: int) -> EvalCase:
    """Parse and validate one dataset record.

    Args:
        payload: Decoded JSON object.
        line_number: 1-based line number for error messages.

    Returns:
        The validated case.

    Raises:
        DatasetError: If the record is not a valid case.
    """
    try:
        case = EvalCase.model_validate(payload)
    except Exception as error:  # noqa: BLE001 - re-raised with line context
        raise DatasetError(f"line {line_number}: invalid case ({error})") from error
    _validate_case_rules(case, line_number)
    return case


def load_dataset(path: Path | str) -> list[EvalCase]:
    """Load and validate a JSONL evaluation dataset.

    Args:
        path: Path of the dataset file.

    Returns:
        The cases in file order.

    Raises:
        DatasetError: If the file is missing, empty, duplicated or malformed.
    """
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise DatasetError(f"dataset not found: {dataset_path}")
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(
        dataset_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise DatasetError(
                f"line {line_number}: not valid JSON ({error})"
            ) from error
        if not isinstance(payload, dict):
            raise DatasetError(f"line {line_number}: expected a JSON object")
        case = parse_case(payload, line_number)
        if case.case_id in seen:
            raise DatasetError(
                f"line {line_number}: duplicated case_id {case.case_id!r}"
            )
        seen.add(case.case_id)
        for marker in PRIVACY_MARKERS:
            if marker in case.input:
                raise DatasetError(
                    f"line {line_number}: input looks like real user data ({marker!r})"
                )
        cases.append(case)
    if not cases:
        raise DatasetError(f"dataset is empty: {dataset_path}")
    return cases


def category_counts(cases: list[EvalCase]) -> dict[str, int]:
    """Return how many cases each category holds.

    Args:
        cases: Loaded cases.

    Returns:
        Mapping of category to count, in the canonical category order.
    """
    counter = Counter(case.category for case in cases)
    return {category: counter.get(category, 0) for category in CATEGORIES}


def smoke_subset(cases: list[EvalCase], per_category: int) -> list[EvalCase]:
    """Return the first `per_category` cases of every category.

    Args:
        cases: Loaded cases.
        per_category: How many cases to keep per category.

    Returns:
        The smoke subset, in dataset order.
    """
    if per_category <= 0:
        return list(cases)
    kept: Counter[str] = Counter()
    subset: list[EvalCase] = []
    for case in cases:
        if kept[case.category] >= per_category:
            continue
        kept[case.category] += 1
        subset.append(case)
    return subset


def interleave(cases: list[EvalCase]) -> list[EvalCase]:
    """Return the cases interleaved across categories (round robin).

    A long run is often stopped early or budgeted for a fixed number of cases.
    Because the dataset is written category by category, plain dataset order means
    a partial run only covers the first categories; interleaving keeps whatever
    completes balanced over the six families, so partial results stay usable.
    Within each category the original order is preserved.

    Args:
        cases: Loaded cases.

    Returns:
        The same cases, reordered by category round robin.
    """
    by_category: dict[str, list[EvalCase]] = {category: [] for category in CATEGORIES}
    for case in cases:
        by_category.setdefault(case.category, []).append(case)
    ordered: list[EvalCase] = []
    index = 0
    while True:
        added = False
        for category in CATEGORIES:
            bucket = by_category.get(category, [])
            if index < len(bucket):
                ordered.append(bucket[index])
                added = True
        if not added:
            break
        index += 1
    return ordered
