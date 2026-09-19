"""Ingestion tests: metadata honesty, page numbers and stable chunk ids.

The corpus under `tests/fixtures/knowledge` is project-authored test material
(marked `test_fixture: true`); it is not Freud / Lacan source text.
"""

from pathlib import Path

from react_agent.rag.ingestion import (
    IngestionReport,
    build_chunks,
    chunk_documents,
    detect_language,
    iter_knowledge_files,
    load_knowledge_file,
    parse_front_matter,
    school_for_path,
    source_id_for_path,
)
from tests.unit_tests.conftest import FIXTURE_KNOWLEDGE_ROOT, make_rag_settings
from tests.unit_tests.rag_helpers import write_minimal_pdf


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_front_matter_is_parsed_and_removed_from_the_body() -> None:
    meta, body = parse_front_matter(
        "---\ntitle: 梦的笔记\nauthor: 项目组\nyear: 1900\n---\n正文第一行\n"
    )

    assert meta == {"title": "梦的笔记", "author": "项目组", "year": "1900"}
    assert body == "正文第一行"


def test_front_matter_without_a_closing_delimiter_is_ignored() -> None:
    meta, body = parse_front_matter("---\ntitle: 没有结尾\n正文\n")

    assert meta == {}
    assert body.startswith("---")


def test_language_detection_is_coarse_but_deterministic() -> None:
    assert detect_language("我梦见水") == "zh"
    assert detect_language("I dreamt of water") == "en"


def test_school_comes_from_the_first_folder(tmp_path: Path) -> None:
    assert school_for_path(tmp_path / "freudian" / "a.md", tmp_path) == "freudian"
    assert school_for_path(tmp_path / "notes" / "a.md", tmp_path) == "general"
    assert school_for_path(tmp_path / "a.md", tmp_path) == "general"


def test_front_matter_can_override_the_school(tmp_path: Path) -> None:
    path = _write_text(
        tmp_path / "freudian" / "a.md", "---\nschool: lacanian\n---\n正文\n"
    )
    documents = load_knowledge_file(path, tmp_path)

    assert documents[0].metadata["school"] == "lacanian"


def test_unknown_metadata_stays_none_instead_of_being_guessed(tmp_path: Path) -> None:
    path = _write_text(
        tmp_path / "lacanian" / "note.md", "只有正文，没有任何元数据。\n"
    )
    metadata = load_knowledge_file(path, tmp_path)[0].metadata

    assert metadata["title"] == "note"
    assert metadata["author"] is None
    assert metadata["work_title"] is None
    assert metadata["year"] is None
    assert metadata["page"] is None
    assert metadata["section"] is None
    assert metadata["source_path"] == "lacanian/note.md"
    assert metadata["test_fixture"] is False


def test_real_front_matter_values_are_kept(tmp_path: Path) -> None:
    path = _write_text(
        tmp_path / "freudian" / "note.md",
        "---\ntitle: 笔记\nauthor: 某人\nwork_title: 某书\nyear: 1953\n---\n正文\n",
    )
    metadata = load_knowledge_file(path, tmp_path)[0].metadata

    assert metadata["author"] == "某人"
    assert metadata["work_title"] == "某书"
    assert metadata["year"] == 1953


def test_markdown_headings_become_sections(tmp_path: Path) -> None:
    path = _write_text(
        tmp_path / "object_relations" / "note.md",
        "---\ntitle: 笔记\n---\n# 第一部分\n内容一\n## 第二节\n内容二\n",
    )
    documents = load_knowledge_file(path, tmp_path)

    assert [document.metadata["section"] for document in documents] == [
        "第一部分",
        "第二节",
    ]
    assert documents[0].page_content == "内容一"
    assert documents[1].page_content == "内容二"
    assert all(document.metadata["page"] is None for document in documents)


def test_txt_is_one_document_without_section(tmp_path: Path) -> None:
    path = _write_text(tmp_path / "freudian" / "note.txt", "纯文本内容。\n")
    documents = load_knowledge_file(path, tmp_path)

    assert len(documents) == 1
    assert documents[0].metadata["section"] is None
    assert documents[0].page_content == "纯文本内容。"


def test_pdf_pages_carry_real_page_numbers(tmp_path: Path) -> None:
    pdf_path = write_minimal_pdf(
        _ensure_dir(tmp_path / "freudian" / "note.pdf"),
        ["defense and repression on page one", "internal objects on page two"],
    )
    documents = load_knowledge_file(pdf_path, tmp_path)

    assert [document.metadata["page"] for document in documents] == [1, 2]
    assert "page one" in documents[0].page_content
    assert documents[0].metadata["title"] == "note"


def test_unsupported_suffixes_are_ignored(tmp_path: Path) -> None:
    _write_text(tmp_path / "freudian" / "skip.docx", "irrelevant")

    assert load_knowledge_file(tmp_path / "freudian" / "skip.docx", tmp_path) == []
    assert iter_knowledge_files(tmp_path) == []


def test_documentation_and_hidden_files_are_not_indexed(tmp_path: Path) -> None:
    _write_text(tmp_path / "README.md", "这个目录用来放理论材料。\n")
    _write_text(tmp_path / "freudian" / "README.md", "学派说明。\n")
    _write_text(tmp_path / "freudian" / "_draft.md", "草稿。\n")
    _write_text(tmp_path / "freudian" / ".hidden.md", "隐藏文件。\n")
    keep = _write_text(tmp_path / "freudian" / "note.md", "真正的材料。\n")

    assert iter_knowledge_files(tmp_path) == [keep]

    report = IngestionReport()
    chunks = build_chunks(tmp_path, make_rag_settings(), report)

    assert [chunk.metadata["source_id"] for chunk in chunks] == ["freudian_note"]
    assert report.files == 1


def test_chunk_ids_are_stable_and_per_source(tmp_path: Path) -> None:
    path = _write_text(
        tmp_path / "freudian" / "long.md",
        "---\ntitle: 长文\n---\n" + "水" * 900 + "\n",
    )
    documents = load_knowledge_file(path, tmp_path)
    first = chunk_documents(documents, chunk_size=300, chunk_overlap=50)
    second = chunk_documents(documents, chunk_size=300, chunk_overlap=50)

    assert len(first) > 1
    assert [chunk.metadata["chunk_id"] for chunk in first] == [
        chunk.metadata["chunk_id"] for chunk in second
    ]
    assert [chunk.metadata["chunk_id"] for chunk in first] == [
        f"freudian_long_{index:06d}" for index in range(len(first))
    ]


def test_chunk_counter_continues_across_calls_with_the_same_state() -> None:
    from langchain_core.documents import Document

    document_a = Document(
        page_content="甲" * 400, metadata={"source_id": "s", "school": "freudian"}
    )
    document_b = Document(
        page_content="乙" * 400, metadata={"source_id": "s", "school": "freudian"}
    )
    counters: dict[str, int] = {}
    first = chunk_documents(
        [document_a], chunk_size=200, chunk_overlap=0, chunk_index_start=counters
    )
    second = chunk_documents(
        [document_b], chunk_size=200, chunk_overlap=0, chunk_index_start=counters
    )

    assert first[0].metadata["chunk_id"] == "s_000000"
    assert second[0].metadata["chunk_id"] == "s_000002"


def test_build_chunks_copies_the_contract_to_every_chunk(tmp_path: Path) -> None:
    _write_text(
        tmp_path / "lacanian" / "note.md",
        "---\ntitle: 能指笔记\nauthor: 项目组\ntest_fixture: true\n---\n# 能指\n"
        + "能指链" * 200
        + "\n",
    )
    report = IngestionReport()
    chunks = build_chunks(tmp_path, make_rag_settings(chunk_size=200), report)

    required = {
        "source_id",
        "school",
        "title",
        "author",
        "work_title",
        "year",
        "source_path",
        "page",
        "section",
        "language",
        "test_fixture",
        "chunk_id",
    }
    assert chunks
    assert report.files == 1
    assert report.chunks == len(chunks)
    assert report.per_school == {"lacanian": len(chunks)}
    for chunk in chunks:
        assert required <= set(chunk.metadata)
        assert chunk.metadata["school"] == "lacanian"
        assert chunk.metadata["test_fixture"] is True
        assert chunk.metadata["author"] == "项目组"


def test_source_id_is_a_clean_slug_or_a_hash(tmp_path: Path) -> None:
    source_id = source_id_for_path(tmp_path / "freudian" / "!!!.md", tmp_path)

    assert source_id
    assert "!" not in source_id


def test_fixture_corpus_is_marked_and_covers_three_schools() -> None:
    chunks = build_chunks(FIXTURE_KNOWLEDGE_ROOT, make_rag_settings())

    assert {chunk.metadata["school"] for chunk in chunks} == {
        "freudian",
        "object_relations",
        "lacanian",
    }
    assert all(chunk.metadata["test_fixture"] is True for chunk in chunks)
    assert all(chunk.metadata["author"] for chunk in chunks)
    assert all(chunk.metadata["page"] is None for chunk in chunks)


def test_ingestion_report_counts_documents_per_school(tmp_path: Path) -> None:
    _write_text(tmp_path / "freudian" / "a.md", "# 一\n内容\n# 二\n内容\n")
    _write_text(tmp_path / "lacanian" / "b.txt", "内容\n")
    report = IngestionReport()

    build_chunks(tmp_path, make_rag_settings(), report)

    assert report.files == 2
    assert report.documents == 3
    assert report.per_school == {"freudian": 2, "lacanian": 1}
    assert report.per_source == {"freudian_a": 2, "lacanian_b": 1}


def _ensure_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
