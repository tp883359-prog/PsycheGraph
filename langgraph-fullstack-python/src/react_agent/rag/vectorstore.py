"""Chroma-backed local vector store.

The store is a *local* Chroma database persisted under `data/vectorstore`
(git-ignored). Indexing happens offline in `scripts/index_knowledge.py`; serving
only opens the existing collection, so a normal request never re-embeds the
knowledge base.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from react_agent.rag.settings import RagSettings, get_rag_settings

logger = logging.getLogger(__name__)

# Cosine space keeps distances comparable across embedding models and lets us
# report a similarity in [0, 1] instead of an unbounded distance.
COLLECTION_METADATA: dict[str, object] = {"hnsw:space": "cosine"}


def build_vectorstore(
    embeddings: Embeddings,
    settings: RagSettings | None = None,
    *,
    collection_name: str | None = None,
) -> Chroma:
    """Open (or create) the persisted Chroma collection.

    Args:
        embeddings: Embedding model used for indexing and querying.
        settings: Optional resolved settings.
        collection_name: Optional override for the collection name.

    Returns:
        A Chroma wrapper bound to the local persistence directory.
    """
    resolved = settings or get_rag_settings()
    return Chroma(
        collection_name=collection_name or resolved.collection_name,
        embedding_function=embeddings,
        persist_directory=resolved.persist_directory,
        collection_metadata=COLLECTION_METADATA,
    )


def index_exists(settings: RagSettings | None = None) -> bool:
    """Report whether a persistence directory is present at all.

    Args:
        settings: Optional resolved settings.

    Returns:
        True when the vector store directory exists on disk.
    """
    resolved = settings or get_rag_settings()
    return Path(resolved.vectorstore_path).is_dir()


def open_vectorstore(
    embeddings: Embeddings | None = None,
    settings: RagSettings | None = None,
) -> Chroma | None:
    """Open the existing index, or return None when there is nothing to read.

    Nothing is created here: when the directory is missing or the collection is
    empty the caller falls back to evidence-free analysis instead of crashing.

    Args:
        embeddings: Embedding model; required to query the collection.
        settings: Optional resolved settings.

    Returns:
        The Chroma wrapper, or None when no usable index exists.
    """
    resolved = settings or get_rag_settings()
    if not index_exists(resolved) or embeddings is None:
        return None
    try:
        store = build_vectorstore(embeddings, resolved)
        if collection_count(store) == 0:
            logger.info("vector store present but empty; skipping retrieval")
            return None
        return store
    except Exception as exc:  # noqa: BLE001 - a broken index must not break a run
        logger.warning("could not open vector store: %s", type(exc).__name__)
        return None


def collection_count(vectorstore: Chroma) -> int:
    """Return the number of stored chunks.

    Args:
        vectorstore: Open Chroma wrapper.

    Returns:
        Chunk count, or 0 when the count cannot be read.
    """
    try:
        collection = getattr(vectorstore, "_collection", None)
        if collection is None:
            return 0
        return int(collection.count())
    except Exception:  # noqa: BLE001 - count is informational only
        return 0


def reset_collection(
    embeddings: Embeddings, settings: RagSettings | None = None
) -> None:
    """Delete the collection contents before a rebuild.

    Args:
        embeddings: Embedding model used to open the store.
        settings: Optional resolved settings.
    """
    resolved = settings or get_rag_settings()
    try:
        store = build_vectorstore(embeddings, resolved)
        store.delete_collection()
    except Exception as exc:  # noqa: BLE001 - rebuild may start from nothing
        logger.info("nothing to reset in vector store: %s", type(exc).__name__)


def store_summary(vectorstore: Chroma | None) -> dict[str, object]:
    """Return a small description of the store for reports.

    Args:
        vectorstore: Open store or None.

    Returns:
        A dict with the collection name and chunk count.
    """
    if vectorstore is None:
        return {"collection": None, "chunks": 0}
    return {
        "collection": getattr(vectorstore, "_collection_name", None),
        "chunks": collection_count(vectorstore),
    }
