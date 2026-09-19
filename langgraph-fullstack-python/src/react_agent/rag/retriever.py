"""Per-school retrieval over the local theory index.

Retrieval is deliberately dumb and cheap: the query text is the Supervisor's
focus for that school plus the user's current message. There is no extra LLM
call for query rewriting, so adding RAG does not increase the model-call count.

Each school queries only its own material (plus shared `general` notes), so a
Freudian specialist never receives Lacanian literature and has to sort it out.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage

from react_agent.rag.embeddings import EmbeddingUnavailableError, get_embeddings
from react_agent.rag.settings import RagSettings, get_rag_settings
from react_agent.rag.vectorstore import index_exists, open_vectorstore
from react_agent.schemas import EvidenceItem
from react_agent.state import SPECIALIST_PERSPECTIVES

logger = logging.getLogger(__name__)

FOCUS_KEY_BY_SCHOOL: dict[str, str] = {
    "freudian": "freudian_focus",
    "object_relations": "object_relations_focus",
    "lacanian": "lacanian_focus",
}
"""Supervisor plan field that holds each school's analysis focus."""

MAX_QUERY_CHARS = 1200
"""Keep the query short: focus plus the current user message only."""


@dataclass
class RetrievalOutcome:
    """Result of one evidence-retrieval pass."""

    evidence_by_school: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    available: bool = False
    elapsed_seconds: float = 0.0
    reason: str | None = None
    queries: dict[str, str] = field(default_factory=dict)
    scores: dict[str, list[float]] = field(default_factory=dict)


def latest_user_text(messages: list[BaseMessage]) -> str:
    """Return the text of the most recent human message.

    Args:
        messages: Conversation messages.

    Returns:
        The latest user text, or an empty string when there is none.
    """
    for message in reversed(messages):
        if getattr(message, "type", "") == "human":
            content = message.content
            if isinstance(content, str):
                return content.strip()
            return str(content).strip()
    return ""


def build_school_query(school: str, plan: dict[str, Any], user_text: str) -> str:
    """Build the retrieval query for one school.

    Args:
        school: School name.
        plan: Supervisor plan as a JSON dict.
        user_text: Latest user message.

    Returns:
        A query combining the school's focus with the user's own words. When the
        plan carries no focus, the user text alone is used.
    """
    focus = str(plan.get(FOCUS_KEY_BY_SCHOOL.get(school, ""), "") or "").strip()
    parts = [part for part in (focus, user_text) if part]
    query = "\n".join(parts).strip()
    return query[:MAX_QUERY_CHARS]


def document_to_evidence(
    document: Document, school: str, score: float | None
) -> EvidenceItem:
    """Convert a retrieved Chroma document into an `EvidenceItem`.

    Args:
        document: Retrieved chunk with the knowledge metadata contract.
        school: School the retrieval was performed for.
        score: Cosine similarity in `[-1, 1]`, or None when unknown.

    Returns:
        A validated evidence item. Unknown metadata stays `None`.
    """
    metadata = dict(document.metadata or {})
    chunk_id = str(metadata.get("chunk_id") or metadata.get("source_id") or "unknown")
    raw_page = metadata.get("page")
    page = int(raw_page) if isinstance(raw_page, (int, float)) else None
    raw_year = metadata.get("year")
    year = int(raw_year) if isinstance(raw_year, (int, float)) else None
    return EvidenceItem(
        evidence_id=chunk_id,
        school=str(metadata.get("school") or school),  # type: ignore[arg-type]
        text=document.page_content.strip(),
        source_id=str(metadata.get("source_id") or "unknown"),
        title=metadata.get("title") or None,
        author=metadata.get("author") or None,
        work_title=metadata.get("work_title") or None,
        year=year,
        page=page,
        section=metadata.get("section") or None,
        source_path=metadata.get("source_path") or None,
        retrieval_score=round(float(score), 4) if score is not None else None,
    )


def retrieve_for_school(
    store: Any,
    school: str,
    query: str,
    top_k: int,
) -> list[EvidenceItem]:
    """Retrieve the top chunks for one school.

    Args:
        store: Open Chroma vector store.
        school: School whose material may be returned.
        query: Query text.
        top_k: Maximum number of chunks.

    Returns:
        Evidence items ordered by descending similarity.

    Raises:
        RuntimeError: If the underlying store query fails; callers decide whether
            to degrade or to surface the error.
    """
    if not query.strip() or top_k <= 0:
        return []
    results = store.similarity_search_with_score(
        query,
        k=top_k,
        filter={"school": {"$in": [school, "general"]}},
    )
    items: list[EvidenceItem] = []
    for document, distance in results:
        # Chroma is configured with cosine space, so 1 - distance is cosine
        # similarity.
        items.append(document_to_evidence(document, school, 1.0 - float(distance)))
    return items


def retrieve_evidence(
    plan: dict[str, Any],
    messages: list[BaseMessage],
    *,
    store: Any | None = None,
    embeddings: Embeddings | None = None,
    settings: RagSettings | None = None,
) -> RetrievalOutcome:
    """Run the three school-specific retrievals for one turn.

    Args:
        plan: Supervisor plan for the current run.
        messages: Conversation messages.
        store: Optional already-open vector store (used by tests and the CLI).
        embeddings: Optional embedding model; a cached one is used otherwise.
        settings: Optional resolved settings.

    Returns:
        A retrieval outcome. When no index or no embedding model is available the
        outcome reports `available=False` and three empty lists, so the graph can
        continue without evidence instead of failing.

    Note:
        The embedding model is only constructed after the index directory has been
        confirmed to exist, because building `HuggingFaceEmbeddings` loads the
        model weights immediately.
    """
    resolved = settings or get_rag_settings()
    outcome = RetrievalOutcome(
        evidence_by_school={school: [] for school in SPECIALIST_PERSPECTIVES}
    )
    if not resolved.rag_enabled:
        outcome.reason = "disabled"
        return outcome

    user_text = latest_user_text(messages)
    started = time.perf_counter()
    # Check the index before touching the embedding model: constructing
    # HuggingFaceEmbeddings loads (and on first use downloads) BGE-M3, which
    # must not happen just because an installation has no index yet.
    if store is None and not index_exists(resolved):
        outcome.reason = "no-index"
        outcome.elapsed_seconds = round(time.perf_counter() - started, 3)
        return outcome
    try:
        active_store = store
        if active_store is None:
            active_store = open_vectorstore(embeddings or get_embeddings(), resolved)
    except EmbeddingUnavailableError as exc:
        outcome.reason = f"embedding-unavailable: {exc}"
        outcome.elapsed_seconds = round(time.perf_counter() - started, 3)
        return outcome
    if active_store is None:
        outcome.reason = "no-index"
        outcome.elapsed_seconds = round(time.perf_counter() - started, 3)
        return outcome

    outcome.available = True
    for school in SPECIALIST_PERSPECTIVES:
        query = build_school_query(school, plan, user_text)
        outcome.queries[school] = query
        try:
            items = retrieve_for_school(active_store, school, query, resolved.top_k)
        except Exception as exc:  # noqa: BLE001 - degrade instead of failing a run
            logger.warning("retrieval failed for %s: %s", school, type(exc).__name__)
            outcome.reason = f"retrieval-error: {type(exc).__name__}"
            items = []
        outcome.evidence_by_school[school] = [
            item.model_dump(mode="json") for item in items
        ]
        outcome.scores[school] = [
            float(item.retrieval_score)
            for item in items
            if item.retrieval_score is not None
        ]
    outcome.elapsed_seconds = round(time.perf_counter() - started, 3)
    return outcome


def evidence_items_from_state(state: Any, school: str) -> list[EvidenceItem]:
    """Return one school's evidence as validated models.

    Args:
        state: Graph state carrying `evidence_by_school` as JSON.
        school: School name.

    Returns:
        Evidence items for that school; empty when nothing was retrieved.
    """
    raw = state.get("evidence_by_school") or {}
    entries = raw.get(school) or []
    items: list[EvidenceItem] = []
    for entry in entries:
        if isinstance(entry, dict):
            items.append(EvidenceItem.model_validate(entry))
    return items
