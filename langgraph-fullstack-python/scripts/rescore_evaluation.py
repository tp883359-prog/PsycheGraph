"""Recompute answer-derived deterministic metrics from stored results.

A metric bug should not force a re-run: the raw answers are on disk, so the
metrics that depend only on the answer can be recomputed without any model call.
This is exactly what happened in the first Phase 8 smoke run (see
`docs/EVALUATION.md` §15).

What can be rescored offline:

    diagnosis_boundary_*      depends on the answer only
    citation_metadata_*       depends on the answer and the (empty) page metadata
    page_claims / fabricated_pages
    unprovided_fact_*         answer part only; the observation part needs the
                              run itself, because observations are not stored for
                              records written before `observations` existed

Usage:

    # show what would change
    uv run python scripts/rescore_evaluation.py --run-dir data/evals/smoke

    # write raw_results_rescored.jsonl and regenerate the reports from it
    uv run python scripts/rescore_evaluation.py --run-dir data/evals/smoke --write
"""

# Progress output is the intended deliverable of this CLI.
# ruff: noqa: T201

import argparse
import json
from pathlib import Path

from react_agent.evaluation.aggregate import (
    build_report,
    write_ablation_markdown,
    write_human_review,
    write_summary_csv,
)
from react_agent.evaluation.dataset import load_dataset
from react_agent.evaluation.metrics import (
    asserted_terms,
    citation_metadata_report,
    diagnosis_boundary,
)
from react_agent.evaluation.runner import EvalResult

DEFAULT_DATASET = Path("evals/dataset.jsonl")


def rescore_record(
    record: EvalResult, case_terms: dict[str, tuple[list[str], list[str], str]]
) -> EvalResult:
    """Return a copy of one record with answer-derived metrics recomputed.

    Args:
        record: Stored result.
        case_terms: For each case, the declared forbidden terms, the terms that
            are still checked (the user did not write them) and the input text.

    Returns:
        The updated record.
    """
    if record.status != "ok":
        return record
    declared, forbidden, _ = case_terms.get(record.case_id, ([], [], ""))
    answer = record.answer
    deterministic = dict(record.deterministic)

    boundary = diagnosis_boundary(answer)
    deterministic["diagnosis_boundary"] = boundary
    deterministic["diagnosis_boundary_pass"] = (
        boundary["pass"] if record.category == "clinical_boundary" else None
    )

    metadata = citation_metadata_report(answer, [])
    deterministic["page_claims"] = metadata["page_claims"]
    deterministic["fabricated_pages"] = metadata["fabricated_pages"]
    deterministic["citation_metadata_ok"] = metadata["citation_metadata_ok"]

    # The observation part of this metric needs the run itself; only the answer
    # is available here, so the observation count is kept as it was measured.
    previous_answer_terms = list(
        deterministic.get("unprovided_fact_terms_answer") or []
    )
    answer_hits = asserted_terms(answer, forbidden)
    observation_terms = list(
        deterministic.get("unprovided_fact_terms_observations") or []
    )
    # Terms the user wrote themselves are excluded from the deterministic check
    # (an honest answer has to be able to quote and refuse them), so they must
    # not be counted any more - including on the observation side.
    excluded = set(declared) - set(forbidden)
    observation_terms = [term for term in observation_terms if term not in excluded]
    deterministic["unprovided_fact_terms_observations"] = observation_terms
    deterministic["unprovided_fact_terms_answer"] = answer_hits
    deterministic["unprovided_fact_violation"] = len(observation_terms) + len(
        answer_hits
    )
    deterministic["observation_fidelity_ok"] = not observation_terms and not answer_hits
    deterministic["forbidden_terms_checked"] = forbidden
    deterministic["forbidden_terms_excluded_because_user_wrote_them"] = [
        term for term in declared if term in set(declared) - set(forbidden)
    ]
    deterministic["rescored_from_stored_answer"] = True
    deterministic["previous_answer_terms"] = previous_answer_terms

    return record.model_copy(update={"deterministic": deterministic})


def main() -> int:
    """Rescore one run folder and optionally regenerate its reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="e.g. data/evals/smoke"
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--write", action="store_true", help="write the rescored results"
    )
    args = parser.parse_args()

    cases = {case.case_id: case for case in load_dataset(args.dataset)}
    case_terms = {
        case_id: (
            list(case.expected_properties.must_not_assume),
            [
                term
                for term in case.expected_properties.must_not_assume
                if term not in case.input
            ],
            case.input,
        )
        for case_id, case in cases.items()
    }

    raw_path = args.run_dir / "raw_results.jsonl"
    records = [
        EvalResult.model_validate(json.loads(line))
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rescored = [rescore_record(record, case_terms) for record in records]

    changed = [
        (
            before.case_id,
            before.variant,
            before.deterministic.get("diagnosis_boundary_pass"),
            after.deterministic.get("diagnosis_boundary_pass"),
            before.deterministic.get("unprovided_fact_violation"),
            after.deterministic.get("unprovided_fact_violation"),
        )
        for before, after in zip(records, rescored, strict=True)
        if before.deterministic.get("diagnosis_boundary_pass")
        != after.deterministic.get("diagnosis_boundary_pass")
        or before.deterministic.get("unprovided_fact_violation")
        != after.deterministic.get("unprovided_fact_violation")
    ]
    print(f"records: {len(records)}, changed: {len(changed)}")
    for case_id, variant, old_pass, new_pass, old_violation, new_violation in changed:
        print(
            f"  {variant:<16} {case_id:<12} diagnosis {old_pass} -> {new_pass} "
            f"| violations {old_violation} -> {new_violation}"
        )

    if not args.write:
        print("dry run: pass --write to store the rescored results")
        return 0

    out = args.run_dir / "raw_results_rescored.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for record in rescored:
            handle.write(record.model_dump_json() + "\n")
    report = build_report(rescored, f"{args.run_dir.name}-rescored")
    (args.run_dir / "report_rescored.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_summary_csv(report, args.run_dir / "summary_rescored.csv")
    write_ablation_markdown(report, args.run_dir / "ablation_rescored.md")
    write_human_review(rescored, args.run_dir / "human_review_rescored.csv")
    print(f"wrote {out} and the *_rescored reports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
