"""Run one (variant, case) pair and record everything that can be measured.

Every result is a single JSON object. Failures are first-class: a provider error
is stored with `status="failed"`, its type, its (sanitised) message and the
number of attempts, so a long run never loses the cases that already succeeded
and never hides the ones that did not.
"""

import asyncio
import os
import time
from datetime import UTC, datetime
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from react_agent.evaluation.dataset import EvalCase
from react_agent.evaluation.judge import judge_answer
from react_agent.evaluation.metrics import (
    answer_of,
    compute_deterministic_metrics,
    observations_of,
)
from react_agent.evaluation.usage import RECORDER, CallRecorder, install_usage_recorder
from react_agent.evaluation.variants import VariantName, get_variant, initial_state

SECRET_ENV_VARS = (
    "DEEPSEEK_API_KEY",
    "LANGCHAIN_API_KEY",
    "LANGSMITH_API_KEY",
    "OPENAI_API_KEY",
    "HF_TOKEN",
)
"""Environment variables whose values must never appear in a result file."""

RESULT_STATUS = ("ok", "failed")
"""Allowed statuses of one evaluation run."""


class EvalResult(BaseModel):
    """One (variant, case) evaluation record.

    Attributes:
        run_id: Evaluation run identifier (folder name under `data/evals/`).
        case_id: Case identifier.
        category: Case category.
        variant: Which variant produced the answer.
        status: `ok` or `failed`.
        error_type: Exception class name when the run failed.
        error_message: Sanitised error message when the run failed.
        attempts: How many times the case was attempted for this variant.
        answer: The user-facing answer.
        finalization_status: Variant-specific exit status (`passed`,
            `revised_and_passed`, `safe_fallback`, `baseline`, `unreviewed`).
        deterministic: Deterministic metrics (see `metrics.py`).
        critique: Critic verdict summary for the full system, else None.
        tokens: Real token counts when the provider reported them, else None.
        token_usage_available: Whether every call reported token usage.
        judge: The judge's rubric scores, when judging was enabled.
        judge_seconds: Time spent in the judge call.
        llm_calls: Model calls made by the system under evaluation.
        node_seconds: Wall clock time attributed to each graph node.
        latency_seconds: End-to-end time of the system under evaluation.
        started_at / finished_at: UTC timestamps.
    """

    run_id: str
    case_id: str
    category: str
    variant: VariantName
    status: str = Field(default="ok")
    error_type: str | None = None
    error_message: str | None = None
    attempts: int = 1
    input_text: str = ""
    answer: str = ""
    observations: list[str] = Field(default_factory=list)
    finalization_status: str | None = None
    deterministic: dict[str, Any] = Field(default_factory=dict)
    critique: dict[str, Any] | None = None
    tokens: dict[str, int | None] = Field(default_factory=dict)
    token_usage_available: bool = False
    judge: dict[str, Any] | None = None
    judge_seconds: float | None = None
    llm_calls: int = 0
    node_seconds: dict[str, float] = Field(default_factory=dict)
    latency_seconds: float = 0.0
    started_at: str = ""
    finished_at: str = ""


def sanitize(text: str) -> str:
    """Remove credentials from a string before it is written to disk.

    Args:
        text: Any text produced by the run (usually an error message).

    Returns:
        The text with every known credential value and `sk-`-style token
        replaced by `[redacted]`.
    """
    cleaned = text
    for name in SECRET_ENV_VARS:
        value = os.environ.get(name, "").strip()
        if len(value) >= 8:
            cleaned = cleaned.replace(value, "[redacted]")
    for prefix in ("sk-", "lsv2_", "hf_"):
        index = cleaned.find(prefix)
        while index != -1:
            end = index + len(prefix)
            while end < len(cleaned) and (
                cleaned[end].isalnum() or cleaned[end] in "-_"
            ):
                end += 1
            cleaned = cleaned[:index] + "[redacted]" + cleaned[end:]
            index = cleaned.find(prefix)
    return cleaned


def _now() -> str:
    """Return the current UTC time as an ISO string."""
    return datetime.now(UTC).isoformat(timespec="seconds")


async def _invoke_graph(
    graph: Any, case_input: str, config: RunnableConfig
) -> tuple[dict[str, Any], dict[str, float]]:
    """Run one graph, collecting the final state and per-node wall clock time.

    Args:
        graph: Compiled variant graph.
        case_input: The user's message.
        config: Runnable config (thread id is used for checkpointed graphs).

    Returns:
        The final state values and the time attributed to each node.
    """
    node_seconds: dict[str, float] = {}
    state: dict[str, Any] = {}
    last = time.perf_counter()
    async for mode, payload in graph.astream(
        initial_state(case_input), config=config, stream_mode=["updates", "values"]
    ):
        now = time.perf_counter()
        if mode == "updates":
            for node in payload:
                node_seconds[node] = node_seconds.get(node, 0.0) + (now - last)
            last = now
        else:
            state = payload
    return state, node_seconds


def _critique_summary(state: dict[str, Any]) -> dict[str, Any] | None:
    """Summarise the Critic verdict for reports, if a Critic ran."""
    critique = state.get("critique")
    if not isinstance(critique, dict):
        return None
    issues = [
        issue for issue in critique.get("issues") or [] if isinstance(issue, dict)
    ]
    return {
        "verdict": critique.get("verdict"),
        "issue_count": len(issues),
        "categories": sorted({str(issue.get("category")) for issue in issues}),
        "clinical_safety_ok": critique.get("clinical_safety_ok"),
        "evidence_grounding_ok": critique.get("evidence_grounding_ok"),
        "deterministic_issues": len(state.get("deterministic_issues") or []),
    }


async def run_case(
    case: EvalCase,
    variant_name: VariantName,
    *,
    run_id: str,
    config: RunnableConfig | None = None,
    judge: bool = True,
    max_attempts: int = 2,
) -> EvalResult:
    """Run one case with one variant, retrying transient failures.

    Args:
        case: Case to run.
        variant_name: Variant identifier.
        run_id: Evaluation run id.
        config: Optional runnable config forwarded to the graph and the judge.
        judge: Whether to score the answer with the judge.
        max_attempts: How many times to attempt the run before reporting failure.

    Returns:
        The result record; failures are returned, never raised.
    """
    install_usage_recorder()
    variant = get_variant(variant_name)
    started_at = _now()

    for attempt in range(1, max_attempts + 1):
        recorder = CallRecorder()
        token = RECORDER.set(recorder)
        started = time.perf_counter()
        try:
            graph = variant.build()
            run_config: RunnableConfig = config or {
                "configurable": {"thread_id": f"{run_id}-{variant_name}-{case.case_id}"}
            }
            state, node_seconds = await _invoke_graph(graph, case.input, run_config)
            latency = time.perf_counter() - started
        except Exception as error:  # noqa: BLE001 - failures are recorded, not raised
            latency = time.perf_counter() - started
            if attempt < max_attempts:
                await asyncio.sleep(2.0)
                continue
            return EvalResult(
                run_id=run_id,
                case_id=case.case_id,
                category=case.category,
                variant=variant_name,
                status="failed",
                error_type=type(error).__name__,
                error_message=sanitize(str(error))[:600],
                attempts=attempt,
                input_text=case.input,
                llm_calls=recorder.call_count,
                node_seconds={},
                latency_seconds=latency,
                started_at=started_at,
                finished_at=_now(),
            )
        finally:
            RECORDER.reset(token)

        deterministic = compute_deterministic_metrics(
            case,
            variant,
            state,
            schema_valid=recorder.parse_failures == 0,
            llm_calls=recorder.call_count,
            parse_failures=recorder.parse_failures,
            latency_seconds=latency,
            retrieval_latency=node_seconds.get("evidence"),
            critic_latency=node_seconds.get("critic"),
            revision_triggered=(
                int(state.get("revision_count", 0) or 0) > 0
                if variant.uses_critic
                else None
            ),
            safe_fallback_triggered=(
                state.get("finalization_status") == "safe_fallback"
                if variant.uses_critic
                else None
            ),
        )
        answer = answer_of(state)

        judge_payload: dict[str, Any] | None = None
        judge_seconds: float | None = None
        if judge:
            judge_started = time.perf_counter()
            try:
                judge_payload = await judge_answer(
                    case, answer, state=state, config=config
                )
            except Exception as error:  # noqa: BLE001 - a judge failure is recorded
                judge_payload = {
                    "error_type": type(error).__name__,
                    "error_message": sanitize(str(error))[:300],
                }
            judge_seconds = time.perf_counter() - judge_started

        return EvalResult(
            run_id=run_id,
            case_id=case.case_id,
            category=case.category,
            variant=variant_name,
            status="ok",
            attempts=attempt,
            input_text=case.input,
            answer=answer,
            observations=observations_of(variant, state),
            finalization_status=(
                str(state.get("finalization_status"))
                if state.get("finalization_status")
                else None
            ),
            deterministic=deterministic,
            critique=_critique_summary(state),
            tokens={
                "prompt_tokens": recorder.prompt_tokens,
                "completion_tokens": recorder.completion_tokens,
                "total_tokens": recorder.total_tokens,
            },
            token_usage_available=recorder.token_usage_available,
            judge=judge_payload,
            judge_seconds=judge_seconds,
            llm_calls=recorder.call_count,
            node_seconds=node_seconds,
            latency_seconds=latency,
            started_at=started_at,
            finished_at=_now(),
        )

    raise AssertionError("unreachable")  # pragma: no cover - loop always returns
