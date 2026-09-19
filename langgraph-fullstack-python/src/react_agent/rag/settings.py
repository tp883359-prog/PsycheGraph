"""Environment-driven configuration for the local knowledge base.

Everything the RAG layer needs is configured through environment variables with
sensible local defaults, so no cloud service and no extra database is required:

    EMBEDDING_MODEL      default BAAI/bge-m3 (runs locally, CPU by default)
    EMBEDDING_DEVICE     default cpu; set to cuda only if the user asks for it
    RAG_VECTORSTORE_PATH default data/vectorstore (git-ignored)
    RAG_COLLECTION       default psychegraph_theory
    RAG_TOP_K            default 5 evidence chunks per school
    RAG_CHUNK_SIZE       default 1000 characters
    RAG_CHUNK_OVERLAP    default 150 characters
    RAG_ENABLED          default true; set to false to skip retrieval entirely
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_EMBEDDING_DEVICE = "cpu"
DEFAULT_VECTORSTORE_PATH = "data/vectorstore"
DEFAULT_COLLECTION = "psychegraph_theory"
DEFAULT_TOP_K = 5
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150


def project_root() -> Path:
    """Return the repository root that contains `pyproject.toml`.

    The path is derived from this module's file name only. `Path.resolve()` and
    `Path.cwd()` are deliberately avoided: they perform a blocking `os.getcwd()`
    call, which the LangGraph dev server rejects (blockbuster) and which would
    abort the `evidence` node.

    Returns:
        The nearest ancestor of this module holding a `pyproject.toml`, or the
        directory of this module when none is found.
    """
    module_path = Path(os.path.abspath(__file__))
    for candidate in module_path.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return module_path.parent


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


@dataclass(frozen=True)
class RagSettings:
    """Resolved settings for embedding, indexing and retrieval."""

    embedding_model: str
    embedding_device: str
    vectorstore_path: Path
    collection_name: str
    top_k: int
    chunk_size: int
    chunk_overlap: int
    rag_enabled: bool = True

    @property
    def persist_directory(self) -> str:
        """Return the vector store directory as a string for Chroma."""
        return str(self.vectorstore_path)


@lru_cache(maxsize=1)
def get_rag_settings() -> RagSettings:
    """Return the RAG settings resolved from the environment.

    Returns:
        Cached settings; the environment is read once per process.
    """
    raw_path = os.environ.get("RAG_VECTORSTORE_PATH", "").strip()
    path = Path(raw_path) if raw_path else project_root() / DEFAULT_VECTORSTORE_PATH
    if not path.is_absolute():
        # Joining is enough here; resolving would call os.getcwd(), which the
        # dev server's blocking-call detector forbids.
        path = project_root() / path
    return RagSettings(
        embedding_model=(
            os.environ.get("EMBEDDING_MODEL", "").strip() or DEFAULT_EMBEDDING_MODEL
        ),
        embedding_device=(
            os.environ.get("EMBEDDING_DEVICE", "").strip() or DEFAULT_EMBEDDING_DEVICE
        ),
        vectorstore_path=path,
        collection_name=(
            os.environ.get("RAG_COLLECTION", "").strip() or DEFAULT_COLLECTION
        ),
        top_k=_env_int("RAG_TOP_K", DEFAULT_TOP_K),
        chunk_size=_env_int("RAG_CHUNK_SIZE", DEFAULT_CHUNK_SIZE),
        chunk_overlap=_env_int("RAG_CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP),
        rag_enabled=os.environ.get("RAG_ENABLED", "true").strip().lower()
        not in {"false", "0", "no"},
    )


def reset_settings_cache() -> None:
    """Clear the settings cache so tests can change the environment."""
    get_rag_settings.cache_clear()
