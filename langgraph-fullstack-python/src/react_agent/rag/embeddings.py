"""Local embedding model (BGE-M3) used by the knowledge base.

The model runs entirely on this machine: no DeepSeek call and no cloud service
is involved in embedding. The heavy dependency (`langchain-huggingface` and its
`torch` backend) is imported lazily so that importing the graph never requires
the embedding stack, and so that tests without the model can still run the
retrieval mechanics with a substitute embedder.
"""

from functools import lru_cache
from typing import Any

from langchain_core.embeddings import Embeddings

from react_agent.rag.settings import RagSettings, get_rag_settings


class EmbeddingUnavailableError(RuntimeError):
    """Raised when the local embedding model cannot be loaded."""


def build_embeddings(settings: RagSettings | None = None) -> Embeddings:
    """Create the local embedding model described by the settings.

    Args:
        settings: Optional settings; the environment-derived ones are used when
            omitted.

    Returns:
        A LangChain `Embeddings` instance with normalised vectors.

    Raises:
        EmbeddingUnavailableError: If the embedding dependency is missing or the
            model cannot be loaded.
    """
    resolved = settings or get_rag_settings()
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise EmbeddingUnavailableError(
            "langchain-huggingface is not installed; run "
            "`uv add langchain-huggingface sentence-transformers`"
        ) from exc

    try:
        return HuggingFaceEmbeddings(
            model_name=resolved.embedding_model,
            model_kwargs={"device": resolved.embedding_device},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc:  # pragma: no cover - depends on the environment
        raise EmbeddingUnavailableError(
            f"could not load embedding model {resolved.embedding_model!r}: "
            f"{type(exc).__name__}"
        ) from exc


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Return the process-wide embedding model.

    Loading BGE-M3 takes seconds, so the instance is cached per process; the
    first call is the only one that pays the model-initialisation cost.

    Returns:
        The cached embedding model.

    Raises:
        EmbeddingUnavailableError: If the model cannot be loaded.
    """
    return build_embeddings()


def reset_embeddings_cache() -> None:
    """Clear the cached embedding model so tests can inject their own."""
    get_embeddings.cache_clear()


def describe_embedding_backend(embeddings: Embeddings) -> dict[str, Any]:
    """Return a small description of an embedder for logs and reports.

    Args:
        embeddings: Any LangChain embedder.

    Returns:
        A dict with the class name and, when present, the model name.
    """
    model_name = getattr(embeddings, "model_name", None)
    return {"class": type(embeddings).__name__, "model": model_name}
