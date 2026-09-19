"""Readiness information for the deployed app.

`/health` distinguishes "the process is up" from "retrieval is ready":

* `graph_loaded` - the graph module imports and exposes a compiled graph;
* `vectorstore_available` - a local index exists at the configured path;
* `embedding_loaded` - BGE-M3 has already been loaded in this process. It stays
  `False` until the first run, because the model is loaded lazily (2.2 GB, ~20 s
  on CPU). A `False` value is therefore **not** a failure.

The endpoint returns exactly these fields plus a status word: no environment
variables, no paths, no keys, no database URIs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from react_agent.rag.embeddings import get_embeddings
from react_agent.rag.settings import get_rag_settings
from react_agent.rag.vectorstore import index_exists

logger = logging.getLogger(__name__)

EMBEDDING_CACHE_HIT = 1
"""`lru_cache` reports one cached item once the model has been loaded."""


@dataclass(frozen=True)
class HealthReport:
    """Safe readiness payload.

    Attributes:
        status: `ok` when the process can serve requests.
        graph_loaded: Whether the agent graph is importable in this process.
        vectorstore_available: Whether a local index exists.
        embedding_loaded: Whether BGE-M3 is already in memory (lazy by design).
    """

    status: str
    graph_loaded: bool
    vectorstore_available: bool
    embedding_loaded: bool

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-ready payload.

        Returns:
            The four documented fields, nothing else.
        """
        return {
            "status": self.status,
            "graph_loaded": self.graph_loaded,
            "vectorstore_available": self.vectorstore_available,
            "embedding_loaded": self.embedding_loaded,
        }


def graph_is_loaded() -> bool:
    """Report whether the compiled graph can be imported.

    Returns:
        True when `react_agent.graph.graph` exists.
    """
    try:
        from react_agent.graph import graph
    except Exception as exc:  # noqa: BLE001 - readiness must never raise
        logger.warning("graph not importable: %s", type(exc).__name__)
        return False
    return graph is not None


def _safe(checker: Callable[[], bool], label: str) -> bool:
    """Run one readiness probe without letting it break the endpoint.

    Args:
        checker: Probe to run.
        label: Name used in the warning record.

    Returns:
        The probe result, or False when it raised.
    """
    try:
        return bool(checker())
    except Exception as exc:  # noqa: BLE001 - probes are best effort
        logger.warning("%s probe failed: %s", label, type(exc).__name__)
        return False


def embedding_is_loaded() -> bool:
    """Report whether the embedding model is already cached in this process.

    Returns:
        True when the model has been loaded (and thus the next retrieval is
        fast); False while it is still lazy.
    """
    try:
        cache_info = getattr(get_embeddings, "cache_info", None)
        if cache_info is None:  # pragma: no cover - defensive
            return False
        return bool(cache_info().currsize >= EMBEDDING_CACHE_HIT)
    except Exception:  # noqa: BLE001
        return False


def vectorstore_is_available() -> bool:
    """Report whether a local vector store exists.

    Returns:
        True when the configured index directory holds an index.
    """
    try:
        return bool(index_exists(get_rag_settings()))
    except Exception as exc:  # noqa: BLE001 - a broken index is not fatal
        logger.warning("vector store check failed: %s", type(exc).__name__)
        return False


def build_report(
    *,
    graph_loader: Callable[[], bool] = graph_is_loaded,
    index_checker: Callable[[], bool] = vectorstore_is_available,
    embedding_checker: Callable[[], bool] = embedding_is_loaded,
) -> HealthReport:
    """Build the readiness report.

    Checkers are isolated: a probe that raises is reported as False instead of
    taking the endpoint (and therefore the container healthcheck) down.

    Args:
        graph_loader: Graph check (injectable for tests).
        index_checker: Vector store check (injectable for tests).
        embedding_checker: Embedding cache check (injectable for tests).

    Returns:
        The report. `status` is `ok` when the graph is served and an index is
        present, `degraded` when retrieval will answer without evidence, and
        `unavailable` when the graph itself cannot be imported.
    """
    graph_loaded = _safe(graph_loader, "graph")
    vectorstore_available = _safe(index_checker, "vectorstore")
    embedding_loaded = _safe(embedding_checker, "embedding")
    if not graph_loaded:
        status = "unavailable"
    elif not vectorstore_available:
        status = "degraded"
    else:
        status = "ok"
    return HealthReport(
        status=status,
        graph_loaded=graph_loaded,
        vectorstore_available=vectorstore_available,
        embedding_loaded=embedding_loaded,
    )
