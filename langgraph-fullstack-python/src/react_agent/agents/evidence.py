"""Evidence node: local retrieval between the Supervisor and the specialists.

This node performs no model call. It turns the Supervisor's focus lines plus the
user's current message into three school-specific queries, retrieves the top-k
passages for each school from the local Chroma index, and stores them as plain
JSON evidence.

Keeping retrieval here (and not inside the specialists) means the three
specialists stay symmetric, retrieval happens once per run, and the DeepSeek call
count is unchanged at five per turn.

The retrieval itself is synchronous: loading BGE-M3 and running a Chroma query are
CPU-heavy, blocking calls (``HuggingFaceEmbeddings``, ``torch`` and the sqlite
backed vector store all block). The node therefore runs it in a worker thread via
``asyncio.to_thread`` so the event loop stays free: the dev server no longer needs
``--allow-blocking`` and concurrent requests are not stalled behind a retrieval.
The retrieval logic itself is untouched - same function, same result.
"""

import asyncio
import logging
import time

from langchain_core.runnables import RunnableConfig

from react_agent.agents.context import supervisor_plan
from react_agent.rag.retriever import RetrievalOutcome, retrieve_evidence
from react_agent.state import PsycheGraphState

logger = logging.getLogger(__name__)


async def evidence_node(
    state: PsycheGraphState,
    config: RunnableConfig | None = None,
) -> dict[str, object]:
    """Retrieve theory evidence for each school and store it in the state.

    Args:
        state: Current graph state; the conversation and the Supervisor plan are
            read.
        config: Optional runnable config (unused: no model call happens here).

    Returns:
        Partial state update with `evidence_by_school`. When no index or no
        embedding model is available the three lists are empty and the run
        continues without evidence.
    """
    plan = supervisor_plan(state)
    started = time.perf_counter()
    outcome: RetrievalOutcome = await asyncio.to_thread(
        retrieve_evidence, plan, list(state["messages"])
    )
    logger.info(
        "evidence retrieval: available=%s elapsed=%.3fs reason=%s counts=%s",
        outcome.available,
        outcome.elapsed_seconds,
        outcome.reason,
        {school: len(items) for school, items in outcome.evidence_by_school.items()},
    )
    elapsed = round(time.perf_counter() - started, 3)
    return {
        "evidence_by_school": outcome.evidence_by_school,
        "evidence_meta": {
            "available": outcome.available,
            "reason": outcome.reason,
            "elapsed_seconds": elapsed,
            "retrieval_seconds": outcome.elapsed_seconds,
            "counts": {
                school: len(items)
                for school, items in outcome.evidence_by_school.items()
            },
        },
    }
