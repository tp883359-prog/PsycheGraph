"""Build the local theory index for PsycheGraph.

Indexing is an offline step and deliberately separate from serving: the web
server never re-embeds the knowledge base, it only reads the persisted Chroma
collection.

Usage (from the project root):

    uv run python scripts/index_knowledge.py             # add/refresh chunks
    uv run python scripts/index_knowledge.py --rebuild   # wipe, then re-index
    uv run python scripts/index_knowledge.py --knowledge-dir tests/fixtures/knowledge

Re-running is idempotent: chunk ids are stable (`<source_id>_<index:06d>`), and
Chroma upserts by id, so unchanged material is simply overwritten.
"""

# Script output is the intended deliverable of this manual-check CLI.
# ruff: noqa: T201

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from react_agent.rag.embeddings import (  # noqa: E402
    EmbeddingUnavailableError,
    build_embeddings,
    describe_embedding_backend,
)
from react_agent.rag.ingestion import IngestionReport, build_chunks  # noqa: E402
from react_agent.rag.settings import get_rag_settings  # noqa: E402
from react_agent.rag.vectorstore import (  # noqa: E402
    build_vectorstore,
    collection_count,
    reset_collection,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Index the PsycheGraph knowledge base."
    )
    parser.add_argument(
        "--knowledge-dir",
        default=str(PROJECT_ROOT / "knowledge"),
        help="Directory holding the theory material (default: knowledge/).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete the existing collection before indexing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and chunk only; do not embed or write anything.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the indexing pipeline.

    Returns:
        Process exit code: 0 on success, 1 when the embedding model is missing.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    settings = get_rag_settings()
    knowledge_root = Path(args.knowledge_dir).resolve()

    print(f"knowledge dir : {knowledge_root}")
    print(f"vectorstore   : {settings.vectorstore_path}")
    print(f"collection    : {settings.collection_name}")
    print(
        f"chunking      : size={settings.chunk_size} overlap={settings.chunk_overlap}"
    )
    print(f"top_k         : {settings.top_k}")

    report = IngestionReport()
    chunk_started = time.perf_counter()
    chunks = build_chunks(knowledge_root, settings, report)
    chunk_seconds = time.perf_counter() - chunk_started

    print(f"\nfiles read    : {report.files}")
    print(f"documents     : {report.documents}")
    print(f"chunks        : {len(chunks)}")
    print(f"chunking time : {chunk_seconds:.2f}s")
    print(f"per school    : {dict(sorted(report.per_school.items()))}")
    if report.skipped:
        print(f"skipped       : {len(report.skipped)} file(s)")
    if not chunks:
        print("\nNothing to index: the knowledge directory has no supported files.")
        return 0

    sample = chunks[0]
    print(f"first chunk_id: {sample.metadata['chunk_id']}")

    if args.dry_run:
        print("\n--dry-run: no embedding, no writes.")
        return 0

    try:
        model_started = time.perf_counter()
        embeddings = build_embeddings(settings)
        # Force a first encode so model-download time is reported separately from
        # steady-state retrieval.
        embeddings.embed_query("warmup")
        model_seconds = time.perf_counter() - model_started
    except EmbeddingUnavailableError as exc:
        print(f"\nEmbedding model unavailable: {exc}")
        print("Install it with: uv add langchain-huggingface sentence-transformers")
        return 1

    print(f"\nembedding     : {describe_embedding_backend(embeddings)}")
    print(f"model load    : {model_seconds:.2f}s (first use, includes any download)")

    if args.rebuild:
        reset_collection(embeddings, settings)
        print("collection    : reset")

    store = build_vectorstore(embeddings, settings)
    write_started = time.perf_counter()
    ids = [str(chunk.metadata["chunk_id"]) for chunk in chunks]
    store.add_documents(chunks, ids=ids)
    write_seconds = time.perf_counter() - write_started

    print(f"embed+write   : {write_seconds:.2f}s")
    print(f"collection    : {collection_count(store)} chunks")
    print("\nIndex ready. The graph reads this index without re-embedding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
