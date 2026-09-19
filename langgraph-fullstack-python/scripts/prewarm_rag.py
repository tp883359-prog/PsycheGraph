"""Prewarm the retrieval stack (vector store + BGE-M3) before serving traffic.

BGE-M3 is ~2.2 GB and needs ~20 s to load on CPU. The server deliberately keeps
loading it lazily - a container must start fast and must not download weights on
boot - so a deployment runs this command once after the image starts (or as an
init job) to pay that cost outside a user request.

    uv run python scripts/prewarm_rag.py

It prints only non-sensitive facts: whether an index was found, whether the
model loaded, how long it took, the model name and the device. No API key, no
user data, no document text is ever printed.
"""

# Progress output is the intended deliverable of this CLI.
# ruff: noqa: T201

from __future__ import annotations

import argparse
import time

from react_agent.logging_config import configure_logging
from react_agent.rag.embeddings import (
    EmbeddingUnavailableError,
    describe_embedding_backend,
    get_embeddings,
)
from react_agent.rag.settings import get_rag_settings
from react_agent.rag.vectorstore import collection_count, index_exists, open_vectorstore


def main() -> int:
    """Load the index and the embedding model once and report the result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe",
        default="warmup probe",
        help="Short text used for one real embedding + similarity search.",
    )
    args = parser.parse_args()
    configure_logging()

    settings = get_rag_settings()
    print(f"embedding model   : {settings.embedding_model}")
    print(f"embedding device  : {settings.embedding_device}")
    print(f"rag enabled       : {settings.rag_enabled}")

    started = time.perf_counter()
    if not index_exists(settings):
        print("vectorstore       : missing (retrieval will degrade gracefully)")
        print(f"index path set    : {settings.vectorstore_path.name}")
        return 0
    print("vectorstore       : found")

    try:
        embeddings = get_embeddings()
    except EmbeddingUnavailableError as exc:
        print(f"embedding_loaded  : False ({exc})")
        return 1
    load_seconds = time.perf_counter() - started
    info = describe_embedding_backend(embeddings)
    print("embedding_loaded  : True")
    print(f"loaded            : {info.get('class')} / {info.get('model')}")
    print(f"elapsed           : {load_seconds:.1f}s")

    query_started = time.perf_counter()
    store = open_vectorstore(embeddings, settings)
    if store is None:
        print("vectorstore       : could not be opened")
        return 1
    vector = embeddings.embed_query(args.probe[:120])
    print(f"embedding dims    : {len(vector)}")
    results = store.similarity_search_with_score(args.probe, k=1)
    print(f"probe search      : {len(results)} chunk(s) returned")
    chunks = collection_count(store)
    print(f"indexed chunks    : {chunks}")
    print(
        f"query elapsed     : {time.perf_counter() - query_started:.3f}s (model cached)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
