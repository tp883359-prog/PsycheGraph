"""Retrieval and citation tests.

These use a real Chroma index built in `tmp_path` from
`tests/fixtures/knowledge` with a deterministic hashing embedder, so they check
the *mechanics* (filters, top_k, metadata, degradation) without downloading
BGE-M3. Semantic quality is checked separately by the real-model verification
script.
"""

import os
from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from react_agent.rag.citations import (
    build_label_map,
    describe_source,
    format_citation,
    render_sources_section,
)
from react_agent.rag.retriever import (
    build_school_query,
    document_to_evidence,
    evidence_items_from_state,
    retrieve_evidence,
    retrieve_for_school,
)
from react_agent.rag.settings import (
    get_rag_settings,
    project_root,
    reset_settings_cache,
)
from react_agent.rag.vectorstore import (
    build_vectorstore,
    collection_count,
    open_vectorstore,
)
from react_agent.schemas import EvidenceItem, TheoryInterpretation
from react_agent.state import SPECIALIST_PERSPECTIVES
from tests.unit_tests.conftest import RagIndex, make_plan, make_rag_settings
from tests.unit_tests.rag_helpers import HashingEmbeddings


def _query_for(school: str) -> str:
    """Return a keyword query aimed at one school's fixture material."""
    return {
        "freudian": "防御机制 压抑 梦的工作 愿望",
        "object_relations": "内在客体 分裂 投射 抱持",
        "lacanian": "能指 能指链 欲望 缺失",
    }[school]


def test_the_test_index_contains_every_chunk(rag_index: RagIndex) -> None:
    assert collection_count(rag_index.store) == len(rag_index.chunks)


def test_each_school_retrieves_only_its_own_material(rag_index: RagIndex) -> None:
    for school in SPECIALIST_PERSPECTIVES:
        items = retrieve_for_school(
            rag_index.store, school, _query_for(school), top_k=5
        )
        assert items, f"no evidence retrieved for {school}"
        for item in items:
            assert item.school in {school, "general"}, (
                f"{school} received {item.school} material"
            )
            assert item.source_id.startswith(school)


def test_a_freudian_query_does_not_return_lacanian_material(
    rag_index: RagIndex,
) -> None:
    items = retrieve_for_school(
        rag_index.store, "freudian", _query_for("lacanian"), top_k=5
    )

    assert items
    assert all(item.source_id.startswith("freudian") for item in items)


def test_top_k_limits_the_number_of_passages(rag_index: RagIndex) -> None:
    many = retrieve_for_school(
        rag_index.store, "freudian", _query_for("freudian"), top_k=5
    )
    few = retrieve_for_school(
        rag_index.store, "freudian", _query_for("freudian"), top_k=1
    )

    assert len(few) == 1
    assert len(many) >= len(few)


def test_similarity_scores_are_reported_as_cosine_similarity(
    rag_index: RagIndex,
) -> None:
    items = retrieve_for_school(
        rag_index.store, "lacanian", _query_for("lacanian"), top_k=3
    )

    scores = [item.retrieval_score for item in items]
    assert all(score is not None for score in scores)
    assert all(-1.0 <= float(score) <= 1.0 for score in scores)
    # The store is queried with cosine space, so the best hit must beat a
    # perfectly unrelated one.
    assert scores == sorted(scores, reverse=True)


def test_an_empty_query_returns_nothing(rag_index: RagIndex) -> None:
    assert retrieve_for_school(rag_index.store, "freudian", "   ", top_k=5) == []


def test_evidence_metadata_matches_the_index(rag_index: RagIndex) -> None:
    items = retrieve_for_school(
        rag_index.store, "object_relations", _query_for("object_relations"), top_k=3
    )
    chunks = {chunk.metadata["chunk_id"]: chunk for chunk in rag_index.chunks}

    for item in items:
        chunk = chunks[item.evidence_id]
        assert item.text == chunk.page_content
        assert item.school == chunk.metadata["school"]
        assert item.title == chunk.metadata["title"]
        assert item.section == chunk.metadata["section"]
        assert item.source_path == chunk.metadata["source_path"]
        # Fixture material has no real page numbers, and nothing may invent one.
        assert item.page is None


def test_page_metadata_survives_conversion() -> None:
    document = Document(
        page_content="defense mechanism passage",
        metadata={
            "chunk_id": "freudian_book_000002",
            "school": "freudian",
            "source_id": "freudian_book",
            "title": "某书",
            "author": "某作者",
            "work_title": "作品",
            "page": 12,
            "section": None,
            "source_path": "freudian/book.pdf",
        },
    )
    item = document_to_evidence(document, "freudian", 0.81234)

    assert item.page == 12
    assert item.retrieval_score == 0.8123
    assert item.evidence_id == "freudian_book_000002"


def test_school_query_combines_focus_and_user_text(rag_index: RagIndex) -> None:
    query = build_school_query(
        "lacanian", make_plan(), "我觉得这像一个循环，一直没有终点。"
    )

    assert make_plan()["lacanian_focus"] in query
    assert "一直没有终点" in query


def test_full_retrieval_reports_availability_and_counts(rag_index: RagIndex) -> None:
    outcome = retrieve_evidence(
        make_plan(), [HumanMessage(content="我梦见水，水是平静的湖水。")]
    )

    assert outcome.available is True
    assert outcome.reason is None
    assert set(outcome.evidence_by_school) == set(SPECIALIST_PERSPECTIVES)
    assert all(outcome.evidence_by_school[school] for school in SPECIALIST_PERSPECTIVES)
    assert outcome.elapsed_seconds >= 0.0
    assert set(outcome.scores) == set(SPECIALIST_PERSPECTIVES)


def test_missing_index_degrades_instead_of_raising(tmp_path: Path) -> None:
    settings = make_rag_settings(
        vectorstore_path=tmp_path / "does-not-exist", rag_enabled=True
    )
    outcome = retrieve_evidence(
        make_plan(),
        [HumanMessage(content="我梦见水。")],
        embeddings=HashingEmbeddings(),
        settings=settings,
    )

    assert outcome.available is False
    assert outcome.reason == "no-index"
    assert outcome.evidence_by_school == {
        school: [] for school in SPECIALIST_PERSPECTIVES
    }


def test_empty_collection_is_treated_as_no_index(tmp_path: Path) -> None:
    settings = make_rag_settings(vectorstore_path=tmp_path / "empty")
    settings.vectorstore_path.mkdir(parents=True)
    embeddings = HashingEmbeddings()
    build_vectorstore(embeddings, settings)

    assert open_vectorstore(embeddings, settings) is None


def test_disabled_rag_returns_early(tmp_path: Path) -> None:
    settings = make_rag_settings(
        vectorstore_path=tmp_path / "whatever", rag_enabled=False
    )
    outcome = retrieve_evidence(
        make_plan(), [HumanMessage(content="我梦见水。")], settings=settings
    )

    assert outcome.available is False
    assert outcome.reason == "disabled"
    assert outcome.elapsed_seconds == 0.0


def test_a_failing_store_is_reported_as_a_retrieval_error() -> None:
    class BrokenStore:
        def similarity_search_with_score(self, *_: Any, **__: Any) -> list[Any]:
            raise RuntimeError("index corrupted")

    outcome = retrieve_evidence(
        make_plan(),
        [HumanMessage(content="我梦见水。")],
        store=BrokenStore(),
        settings=make_rag_settings(),
    )

    assert outcome.available is True
    assert outcome.reason == "retrieval-error: RuntimeError"
    assert outcome.evidence_by_school == {
        school: [] for school in SPECIALIST_PERSPECTIVES
    }


def test_evidence_items_from_state_validates_the_json() -> None:
    payload = {
        "evidence_id": "freudian_a_000000",
        "school": "freudian",
        "text": "防御机制的说明。",
        "source_id": "freudian_a",
        "title": "笔记",
        "author": None,
        "work_title": None,
        "page": None,
        "section": "防御机制",
        "source_path": "freudian/a.md",
        "retrieval_score": 0.5,
    }
    state = {"evidence_by_school": {"freudian": [payload]}}

    items = evidence_items_from_state(state, "freudian")

    assert len(items) == 1
    assert isinstance(items[0], EvidenceItem)
    assert items[0].section == "防御机制"
    assert evidence_items_from_state(state, "lacanian") == []
    assert evidence_items_from_state({}, "freudian") == []


def test_citations_use_only_real_metadata() -> None:
    with_page = EvidenceItem(
        evidence_id="a_000001",
        school="freudian",
        text="x",
        source_id="a",
        title="某书",
        author="某作者",
        work_title="某书",
        page=42,
        section="第一章",
    )
    with_section = EvidenceItem(
        evidence_id="b_000001",
        school="lacanian",
        text="x",
        source_id="b",
        title="笔记",
        section="能指",
    )
    bare = EvidenceItem(
        evidence_id="c_000001",
        school="lacanian",
        text="x",
        source_id="c",
        title=None,
    )

    assert describe_source(with_page) == "某作者, 《某书》, p. 42"
    assert describe_source(with_section) == "《笔记》, 章节「能指」"
    body = describe_source(bare)
    assert "p." not in body
    assert "章节" not in body
    assert "c_000001" in body


def test_citation_labels_are_sequential() -> None:
    items = [
        EvidenceItem(
            evidence_id=f"a_{index:06d}", school="freudian", text="x", source_id="a"
        )
        for index in range(3)
    ]

    labels = build_label_map(items)

    assert list(labels.values()) == ["[E1]", "[E2]", "[E3]"]
    assert format_citation(items[0], labels["a_000000"]).startswith("[E1] ")


def test_sources_section_lists_each_used_item_once() -> None:
    items = [
        EvidenceItem(
            evidence_id=f"a_{index:06d}",
            school="freudian",
            text="x",
            source_id="a",
            title=f"材料{index}",
        )
        for index in range(2)
    ]

    rendered = render_sources_section(items)

    assert rendered.splitlines()[0] == "理论依据（本地知识库）"
    assert rendered.count("[E1]") == 1
    assert "《材料0》" in rendered
    assert "《材料1》" in rendered


def test_settings_resolution_never_calls_the_working_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`os.getcwd()` is a blocking call: the LangGraph dev server rejects it.

    Resolving paths with `Path.resolve()` aborts the `evidence` node inside
    `langgraph dev` (blockbuster), so settings must be derivable from
    `__file__` alone.
    """
    calls: list[str] = []

    def forbidden_getcwd() -> str:
        calls.append("getcwd")
        raise AssertionError("os.getcwd() must not be called while resolving settings")

    monkeypatch.delenv("RAG_VECTORSTORE_PATH", raising=False)
    monkeypatch.setattr(os, "getcwd", forbidden_getcwd)
    reset_settings_cache()

    settings = get_rag_settings()

    assert calls == []
    assert settings.vectorstore_path.is_absolute()
    assert settings.vectorstore_path.name == "vectorstore"
    assert project_root().is_dir()
    assert render_sources_section([]) == ""


def test_theoretical_interpretation_defaults_to_no_citation() -> None:
    interpretation = TheoryInterpretation(
        perspective="freudian",
        claim="一种读解。",
        textual_basis=["我梦见水"],
        uncertainty="材料不足。",
    )

    assert interpretation.evidence_ids == []
