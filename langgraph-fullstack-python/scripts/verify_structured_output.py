"""Verify Phase 4 structured output against the real DeepSeek API.

HISTORICAL (Phase 4): this script verified the *single* psychoanalytic analyst
and its `PsychoanalyticAnalysis` schema. Phase 5 replaced that schema with the
multi-agent contracts (`SupervisorPlan` / `SchoolAnalysis` / `SynthesisResult`)
and Phase 6 added theory evidence, so the checks below no longer match the
current graph.

The current verification entry points are:
    scripts/verify_multi_agent.py   (multi-agent orchestration, real API)
    scripts/verify_rag.py           (retrieval + citations, real API)
    scripts/verify_single_agent.py  (kept as the Phase 3/4 baseline record)

Running this file only prints this notice; the Phase 4 behaviour it describes is
preserved in docs/SINGLE_AGENT_BASELINE.md and docs/STRUCTURED_OUTPUT.md.
"""

# Script output is the intended deliverable of this manual-check CLI.
# ruff: noqa: T201

import sys

RETIRED_NOTICE = (
    "verify_structured_output.py is retired: the single-agent "
    "PsychoanalyticAnalysis schema was replaced in Phase 5.\n"
    "Use scripts/verify_multi_agent.py or scripts/verify_rag.py instead."
)


def main() -> int:
    """Print the retirement notice.

    Returns:
        Exit code 1, because this check is no longer runnable.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(RETIRED_NOTICE)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
