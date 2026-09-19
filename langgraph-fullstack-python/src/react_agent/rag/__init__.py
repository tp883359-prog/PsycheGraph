"""Local Retrieval-Augmented Generation support for PsycheGraph.

The package is split so that indexing and serving stay separate:

    ingestion   load knowledge files, apply the metadata contract, chunk them
    embeddings  BGE-M3 text embeddings, running locally on CPU by default
    vectorstore Chroma persistence, opened read-only during serving
    retriever   per-school dense retrieval producing `EvidenceItem`s
    citations   rendering of real metadata only, never invented references

Nothing here calls the DeepSeek API.
"""

from react_agent.rag.citations import format_citation, render_sources_section
from react_agent.rag.settings import RagSettings, get_rag_settings

__all__ = [
    "RagSettings",
    "format_citation",
    "get_rag_settings",
    "render_sources_section",
]
