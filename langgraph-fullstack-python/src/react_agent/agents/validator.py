"""Deterministic validator node: code-only checks between draft and Critic.

The node performs no model call and no retrieval. It records its findings as
JSON in `deterministic_issues`, which the Critic then receives as part of its
input; the routing decision still belongs to the Critic, but an `error` here
prevents a `pass` verdict.
"""

import logging

from langchain_core.runnables import RunnableConfig

from react_agent.state import PsycheGraphState
from react_agent.validation import collect_deterministic_issues

logger = logging.getLogger(__name__)


async def deterministic_validator_node(
    state: PsycheGraphState,
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    """Check the draft with code and store the findings.

    Args:
        state: Current graph state, after the Synthesizer or the revision step.
        config: Optional runnable config (unused: no model call happens here).

    Returns:
        Partial state update with `deterministic_issues` as JSON dicts.
    """
    issues = collect_deterministic_issues(state)
    errors = [issue for issue in issues if issue.severity == "error"]
    logger.info(
        "deterministic validation: %d issue(s), %d error(s) %s",
        len(issues),
        len(errors),
        [issue.category for issue in errors],
    )
    return {"deterministic_issues": [issue.model_dump(mode="json") for issue in issues]}
