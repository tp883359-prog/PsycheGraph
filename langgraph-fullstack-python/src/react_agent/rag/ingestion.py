"""Knowledge ingestion: file loading, metadata contract and chunking.

The pipeline is deliberately offline and explicit:

    knowledge/<school>/<file>.{md,txt,pdf}
      -> load      (one Document per section or page)
      -> chunk     (deterministic `chunk_id` per source and index)
      -> embed     (BGE-M3, local)
      -> Chroma    (persisted under data/vectorstore)

Metadata honesty is a hard rule: a `page` is only recorded when the loader can
read a real page number, a `section` only when the file has a real heading, and
`year`/`author` only when the file states them. Unknown values stay `None`
instead of being guessed.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from langchain_core.documents import Document

from react_agent.rag.settings import RagSettings, get_rag_settings

SCHOOLS: tuple[str, ...] = ("freudian", "object_relations", "lacanian")
"""Knowledge sub-directories that map to a school. Other names become `general`."""

SUPPORTED_SUFFIXES: tuple[str, ...] = (".md", ".txt", ".pdf")

FRONT_MATTER_DELIMITER = "---"
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class IngestionReport:
    """Summary of one indexing run, used by the CLI and by tests."""

    files: int = 0
    documents: int = 0
    chunks: int = 0
    skipped: list[str] = field(default_factory=list)
    per_school: dict[str, int] = field(default_factory=dict)
    per_source: dict[str, int] = field(default_factory=dict)

    def add_chunk(self, metadata: dict[str, Any]) -> None:
        """Record one produced chunk."""
        school = str(metadata.get("school", "general"))
        source = str(metadata.get("source_id", ""))
        self.chunks += 1
        self.per_school[school] = self.per_school.get(school, 0) + 1
        self.per_source[source] = self.per_source.get(source, 0) + 1


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Split an optional `---` delimited metadata header from the body.

    Only flat `key: value` lines are supported, which keeps the parser
    dependency-free and predictable.

    Args:
        text: Raw file content.

    Returns:
        The parsed metadata mapping and the remaining body text.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != FRONT_MATTER_DELIMITER:
        return {}, text
    meta: dict[str, str] = {}
    for index in range(1, len(lines)):
        line = lines[index].strip()
        if line == FRONT_MATTER_DELIMITER:
            body = "\n".join(lines[index + 1 :])
            return meta, body.lstrip("\n")
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip().lower()] = value.strip()
    # No closing delimiter: treat the whole file as body.
    return {}, text


def detect_language(text: str) -> str:
    """Return a coarse language tag for a text.

    Args:
        text: Any text.

    Returns:
        `"zh"` when CJK characters are present, otherwise `"en"`.
    """
    return "zh" if _CJK_RE.search(text) else "en"


def school_for_path(path: Path, knowledge_root: Path) -> str:
    """Return the school a knowledge file belongs to.

    Args:
        path: File inside the knowledge directory.
        knowledge_root: Root of the knowledge directory.

    Returns:
        One of `SCHOOLS`, or `"general"` for files outside a school folder.
    """
    try:
        relative = path.relative_to(knowledge_root)
    except ValueError:
        return "general"
    top = relative.parts[0].lower() if relative.parts else ""
    return top if top in SCHOOLS else "general"


def source_id_for_path(path: Path, knowledge_root: Path) -> str:
    """Build a stable, human-readable id for a knowledge file.

    Args:
        path: File inside the knowledge directory.
        knowledge_root: Root of the knowledge directory.

    Returns:
        A slug such as `freudian_freud_dreams` derived from the relative path.
    """
    try:
        relative = path.relative_to(knowledge_root)
    except ValueError:
        relative = Path(path.name)
    stem = relative.with_suffix("")
    parts = [part for part in stem.parts if part not in ("", ".")]
    slug = "_".join(parts)
    slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", slug).strip("_").lower()
    return slug or hashlib.sha1(str(relative).encode("utf-8")).hexdigest()[:12]


def _base_metadata(
    path: Path,
    knowledge_root: Path,
    front_matter: dict[str, str],
    body: str,
) -> dict[str, Any]:
    """Assemble the metadata contract for one knowledge file.

    Args:
        path: File being read.
        knowledge_root: Root of the knowledge directory.
        front_matter: Parsed front-matter values.
        body: File body used for language detection.

    Returns:
        Metadata shared by every chunk of the file. Unknown values are `None`.
    """
    relative = (
        path.relative_to(knowledge_root)
        if knowledge_root in path.parents or path.is_relative_to(knowledge_root)
        else Path(path.name)
    )
    year_raw = front_matter.get("year", "").strip()
    year: int | None = None
    if year_raw.isdigit():
        year = int(year_raw)
    school = front_matter.get("school", "").strip().lower()
    if school not in (*SCHOOLS, "general"):
        school = school_for_path(path, knowledge_root)
    return {
        "source_id": source_id_for_path(path, knowledge_root),
        "school": school,
        "title": front_matter.get("title") or path.stem,
        "author": front_matter.get("author") or None,
        "work_title": front_matter.get("work_title") or None,
        "year": year,
        "source_path": str(relative).replace("\\", "/"),
        "page": None,
        "section": None,
        "language": front_matter.get("language") or detect_language(body),
        "test_fixture": front_matter.get("test_fixture", "").strip().lower()
        in {"1", "true", "yes"},
    }


def _markdown_sections(body: str) -> list[tuple[str | None, str]]:
    """Split markdown into `(heading, text)` pairs.

    Args:
        body: Markdown body without front matter.

    Returns:
        One entry per heading block; text before the first heading uses `None`.
    """
    sections: list[tuple[str | None, list[str]]] = []
    current_heading: str | None = None
    current_lines: list[str] = []
    for line in body.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            if current_lines or current_heading is not None:
                sections.append((current_heading, current_lines))
            current_heading = match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)
    sections.append((current_heading, current_lines))
    result: list[tuple[str | None, str]] = []
    for heading, lines in sections:
        text = "\n".join(lines).strip()
        if text:
            result.append((heading, text))
    return result


def _pdf_documents(path: Path, metadata: dict[str, Any]) -> list[Document]:
    """Load a PDF into one Document per page.

    Args:
        path: PDF path.
        metadata: Base metadata for the file.

    Returns:
        Documents carrying real 1-based `page` numbers.

    Raises:
        ImportError: If `pypdf` is not installed.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("pypdf is required to index PDF files") from exc

    reader = PdfReader(str(path))
    documents: list[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        page_metadata = dict(metadata)
        page_metadata["page"] = page_number
        documents.append(Document(page_content=text, metadata=page_metadata))
    return documents


def load_knowledge_file(path: Path, knowledge_root: Path) -> list[Document]:
    """Load one knowledge file into section/page level documents.

    Args:
        path: File to load; unknown suffixes yield an empty list.
        knowledge_root: Root of the knowledge directory.

    Returns:
        Documents with the metadata contract applied.
    """
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        return []
    if suffix == ".pdf":
        front: dict[str, str] = {}
        base = _base_metadata(path, knowledge_root, front, path.stem)
        return _pdf_documents(path, base)

    text = path.read_text(encoding="utf-8", errors="replace")
    front_matter, body = parse_front_matter(text)
    base = _base_metadata(path, knowledge_root, front_matter, body)
    if suffix == ".md":
        sections = _markdown_sections(body)
        documents = []
        for heading, section_text in sections:
            metadata = dict(base)
            metadata["section"] = heading
            documents.append(Document(page_content=section_text, metadata=metadata))
        return documents
    body = body.strip()
    if not body:
        return []
    return [Document(page_content=body, metadata=dict(base))]


def is_indexable_name(path: Path) -> bool:
    """Report whether a file inside the knowledge tree should be indexed.

    Documentation and editor leftovers live next to the material, so they are
    filtered out: files named `README*`, files starting with `.` or `_`, and
    anything that is not `.md`/`.txt`/`.pdf`.

    Args:
        path: Candidate file.

    Returns:
        True when the file is meant to be retrieved as theory material.
    """
    name = path.name
    if name.startswith((".", "_")):
        return False
    if name.lower().startswith("readme"):
        return False
    return path.suffix.lower() in SUPPORTED_SUFFIXES


def iter_knowledge_files(root: Path) -> list[Path]:
    """Return every supported knowledge file, sorted for stable ids.

    Args:
        root: Knowledge directory.

    Returns:
        Sorted file paths; documentation files such as `README.md` are skipped.
    """
    if not root.is_dir():
        return []
    return sorted(
        path for path in root.rglob("*") if path.is_file() and is_indexable_name(path)
    )


def chunk_documents(
    documents: Iterable[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
    report: IngestionReport | None = None,
    chunk_index_start: dict[str, int] | None = None,
) -> list[Document]:
    """Split documents into retrieval-sized chunks with stable ids.

    Each source file keeps its own running chunk index, so ids look like
    `freud_dreams_000123` and stay identical across re-indexing runs.

    Args:
        documents: Section/page level documents.
        chunk_size: Maximum chunk length in characters.
        chunk_overlap: Overlap between neighbouring chunks.
        report: Optional report updated with per-school counters.
        chunk_index_start: Optional running index per `source_id`, used when a
            file is chunked in several calls.

    Returns:
        Chunked documents; metadata is copied to every chunk.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=False,
    )
    counters = chunk_index_start if chunk_index_start is not None else {}
    chunks: list[Document] = []
    for document in documents:
        source_id = str(document.metadata.get("source_id", "source"))
        for piece in splitter.split_text(document.page_content):
            text = piece.strip()
            if not text:
                continue
            index = counters.get(source_id, 0)
            counters[source_id] = index + 1
            metadata = dict(document.metadata)
            metadata["chunk_id"] = f"{source_id}_{index:06d}"
            chunks.append(Document(page_content=text, metadata=metadata))
            if report is not None:
                report.add_chunk(metadata)
    return chunks


def build_chunks(
    knowledge_root: Path,
    settings: RagSettings | None = None,
    report: IngestionReport | None = None,
) -> list[Document]:
    """Load and chunk an entire knowledge directory.

    Args:
        knowledge_root: Directory holding the theory material.
        settings: Optional settings for chunk sizes.
        report: Optional report updated with counters.

    Returns:
        All chunks, each carrying the metadata contract and a `chunk_id`.
    """
    resolved = settings or get_rag_settings()
    active_report = report if report is not None else IngestionReport()
    counters: dict[str, int] = {}
    all_chunks: list[Document] = []
    for path in iter_knowledge_files(knowledge_root):
        documents = load_knowledge_file(path, knowledge_root)
        if not documents:
            active_report.skipped.append(str(path))
            continue
        active_report.files += 1
        active_report.documents += len(documents)
        all_chunks.extend(
            chunk_documents(
                documents,
                chunk_size=resolved.chunk_size,
                chunk_overlap=resolved.chunk_overlap,
                report=active_report,
                chunk_index_start=counters,
            )
        )
    return all_chunks
