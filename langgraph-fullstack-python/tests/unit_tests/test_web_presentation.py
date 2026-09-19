"""Tests for the presentation layer: escaping, citations, panels, evaluation.

Everything here is pure rendering: no client, no network, no model. The tests
assert both what the UI shows and what it must never show (paths, draft text,
unlinked invented citations).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from react_agent.rag.citations import describe_source
from react_agent.schemas import EvidenceItem
from react_agent.web.citations import (
    build_sources_payload,
    citation_refs,
    describe_card,
    safe_source_label,
    sanitize_card,
)
from react_agent.web.components import (
    answer_fragment,
    chat_bubble,
    close_fragment,
    conversation_columns,
    error_fragment,
    evidence_body,
    evidence_card,
    message_bubble,
    split_sources_section,
)
from react_agent.web.evaluation import (
    DISCLAIMER,
    METRIC_SPECS,
    NA_TEXT,
    format_value,
    load_evaluation_summary,
    parse_summary,
)
from react_agent.web.events import (
    AnswerPayload,
    CitationRef,
    EventMetadata,
    TheoryTag,
    WorkflowEvent,
)
from react_agent.web.fragments import citations_for_state, debug_render
from react_agent.web.render import render_answer
from react_agent.web.workflow import workflow_panel
from tests.unit_tests.web_helpers import (
    WINDOWS_PATH,
    evidence_by_school,
    make_evidence_item,
)


def html(node: object) -> str:
    """Render a node to HTML.

    Args:
        node: FastHTML node.

    Returns:
        The HTML string.
    """
    return str(debug_render(node))


def make_answer_event(
    text: str = "这是一个回答。",
    citations: tuple[CitationRef, ...] = (),
) -> WorkflowEvent:
    """Build an `answer` event for rendering tests.

    Args:
        text: Answer prose.
        citations: Citation references.

    Returns:
        The event.
    """
    payload = AnswerPayload(
        text=text,
        status="passed",
        status_label="质量审核：通过",
        revision_count=0,
        theory_tags=(
            TheoryTag("freudian", "弗洛伊德", True),
            TheoryTag("object_relations", "客体关系", True),
            TheoryTag("lacanian", "拉康", False),
        ),
        citations=citations,
    )
    return WorkflowEvent(
        event_type="answer",
        node="finalize",
        status="completed",
        label="最终回答",
        message="完成",
        metadata=EventMetadata(answer=payload),
    )


# ------------------------------------------------------------------- rendering


def test_answer_text_is_escaped() -> None:
    """Model output must never become markup."""
    html_text = html(render_answer("<script>alert('x')</script> 与 <b>加粗</b>"))
    assert "<script>" not in html_text
    assert "&lt;script&gt;" in html_text
    assert "<b>" not in html_text


def test_user_message_is_escaped() -> None:
    """User input is escaped as well."""
    rendered = html(
        message_bubble({"type": "human", "content": "<img src=x onerror=alert(1)>"})
    )
    assert "<img" not in rendered
    assert "&lt;img" in rendered


def test_markdown_lite_is_rendered() -> None:
    """Paragraphs, bullets, headings and emphasis survive as safe markup."""
    rendered = html(render_answer("## 小结\n\n- 第一点\n- 第二点\n\n**重点**内容"))
    assert "answer-heading" in rendered
    assert rendered.count("<li>") == 2
    assert "<strong>重点</strong>" in rendered


def test_citation_labels_become_buttons_only_when_real() -> None:
    """An invented label must stay plain text without a link."""
    refs = (
        CitationRef(
            label="E1",
            evidence_id="freud_dreams_000001",
            school="freudian",
            description="Freud, 《梦的解析》",
        ),
    )
    rendered = html(render_answer("见 [E1]，另见 [E9]。", refs))
    assert 'data-evidence="freud_dreams_000001"' in rendered
    assert "[E9]" in rendered
    assert rendered.count("data-evidence=") == 1


def test_answer_fragment_shows_tags_and_quality() -> None:
    """Theory tags and the quality line accompany the answer."""
    rendered = html(answer_fragment(make_answer_event()))
    assert "弗洛伊德" in rendered
    assert "拉康（未完成）" in rendered
    assert "tag--missing" in rendered
    assert "质量审核：通过" in rendered


def test_answer_fragment_deletes_the_placeholder() -> None:
    """The waiting bubble disappears when the answer arrives."""
    rendered = html(answer_fragment(make_answer_event()))
    assert 'hx-swap-oob="delete:#assistant-placeholder"' in rendered


def test_error_fragment_is_generic() -> None:
    """Errors explain what happened without internals."""
    event = WorkflowEvent(
        event_type="error",
        label="运行失败",
        message="本次分析未完成，请重试。",
        metadata=EventMetadata(error_code="run_failed"),
    )
    rendered = html(error_fragment(event))
    assert "本次分析未完成，请重试。" in rendered
    assert "Traceback" not in rendered
    assert "RuntimeError" not in rendered


def test_close_fragment_restores_the_composer() -> None:
    """A finished run re-enables the input and clears the stream controls."""
    event = WorkflowEvent(
        event_type="close",
        label="结束",
        message="本次分析已结束",
        elapsed_seconds=42.5,
        metadata=EventMetadata(revision_count=1),
    )
    rendered = html(close_fragment(event, "thread-1"))
    assert "服务端用时 42.5s" in rendered
    assert "hx-swap-oob" in rendered
    assert "#stream-controls" in rendered
    assert 'name="msg"' in rendered


def test_split_sources_section_separates_prose_from_citations() -> None:
    """Stored answers keep prose and source block apart."""
    prose, block = split_sources_section(
        "正文内容。\n\n理论依据（本地知识库）\n[E1] 某来源"
    )
    assert prose == "正文内容。"
    assert block.startswith("理论依据")
    assert "[E1]" in block


def test_history_bubble_without_run_keeps_labels_as_text() -> None:
    """Older turns must not fake clickable citations."""
    stored = "旧回答。\n\n理论依据（本地知识库）\n[E1] 旧来源"
    rendered = html(message_bubble({"type": "ai", "content": stored}))
    assert "data-evidence=" not in rendered
    assert "[E1]" in rendered


# ------------------------------------------------------------------ citations


def test_source_path_never_reaches_the_card() -> None:
    """Local paths are dropped, a bare file name is kept."""
    card = sanitize_card(
        make_evidence_item("freud_dreams_000001", "freudian"), "freudian"
    )
    assert card is not None
    assert "source_path" not in {field for field in vars(card)}
    assert WINDOWS_PATH not in html(evidence_card(card))
    assert card.source_label == "freud_dreams"
    assert ":" not in card.source_label


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (r"C:\Users\someone\notes.md", "notes.md"),
        (r"C:\Users\someone", None),
        (r"C:\Users\someone\Documents", None),
        ("/home/someone/notes.md", "notes.md"),
        ("notes.md", "notes.md"),
        (None, None),
        ("", None),
        ("dev/null", "null"),
        ("https://example.com/a.md", None),
    ],
)
def test_safe_source_label(value: str | None, expected: str | None) -> None:
    """Source labels are reduced to a bare file name."""
    assert safe_source_label(value) == expected


def test_card_description_matches_product_citations() -> None:
    """The panel prints exactly the metadata the product would print."""
    raw = make_evidence_item(
        "freud_dreams_000001",
        "freudian",
        page=42,
        year=1900,
        author="Freud",
        work_title="梦的解析",
    )
    card = sanitize_card(raw, "freudian")
    assert card is not None
    item = EvidenceItem.model_validate(raw)
    assert describe_card(card) == describe_source(item)


def test_card_without_metadata_stays_honest() -> None:
    """A passage without author/page is shown as unlabelled, not invented."""
    raw = make_evidence_item(
        "generic_note_000009",
        "general",
        author=None,
        work_title=None,
        title=None,
        page=None,
        section=None,
    )
    card = sanitize_card(raw, "general")
    assert card is not None
    assert "未标注来源" in describe_card(card)


def test_evidence_body_hides_internal_fields() -> None:
    """The rendered panel never contains vector or path internals."""
    long_text = (
        "梦的工作把愿望改写成意象。这是一段足够长的段落，用来验证折叠展示。" * 12
    )
    raw = {
        "freudian": [
            make_evidence_item("freud_dreams_000001", "freudian", text=long_text)
        ],
    }
    payload = build_sources_payload(raw, {"available": True, "counts": {"freudian": 1}})
    rendered = html(evidence_body(payload))
    for forbidden in ("source_path", "retrieval_score", "C:\\", ".venv", "chroma"):
        assert forbidden not in rendered
    assert "弗洛伊德" in rendered
    assert "展开原文片段" in rendered


def test_evidence_body_reports_an_empty_index() -> None:
    """No material is stated plainly instead of showing an empty panel."""
    payload = build_sources_payload({}, {"available": False, "reason": "no-index"})
    rendered = html(evidence_body(payload))
    assert "本轮没有可用的本地文献" in rendered
    assert "no-index" in rendered


def test_citation_refs_follow_used_order_and_skip_unknown() -> None:
    """Labels follow the citation order of the finalizer, unknown ids drop."""
    payload = build_sources_payload(evidence_by_school(), {"available": True})
    refs = citation_refs(
        ["lacan_signifier_000003", "missing_id", "freud_dreams_000001"],
        payload.cards,
    )
    assert [ref.label for ref in refs] == ["E1", "E2"]
    assert refs[0].evidence_id == "lacan_signifier_000003"
    assert refs[1].evidence_id == "freud_dreams_000001"


def test_citations_for_state_matches_the_answer() -> None:
    """The page-load mapping comes from real state, not from the answer text."""
    refs = citations_for_state(
        {
            "final_result": {"used_evidence_ids": ["freud_dreams_000001"]},
            "evidence_by_school": evidence_by_school(),
            "evidence_meta": {"available": True},
        }
    )
    assert [ref.evidence_id for ref in refs] == ["freud_dreams_000001"]
    assert citations_for_state({"messages": []}) == ()


# --------------------------------------------------------------- workflow panel


def test_workflow_panel_lists_every_phase() -> None:
    """The checklist groups nodes and marks optional ones as not triggered."""
    rendered = html(workflow_panel())
    for label in (
        "分析任务",
        "检索理论材料",
        "弗洛伊德视角",
        "客体关系视角",
        "拉康视角",
        "综合三个理论视角",
        "确定性检查",
        "质量审核",
        "修订综合回答",
        "完成",
        "采用保守回答",
    ):
        assert label in rendered
    assert "并行执行" in rendered
    assert rendered.count("未触发") >= 2
    assert "节点状态，不含模型内部推理" in rendered


def test_workflow_panel_states_are_text_not_only_colour() -> None:
    """Every status carries text and a symbol."""
    rendered = html(workflow_panel())
    assert "等待" in rendered
    assert "○" in rendered


def test_sidebar_highlights_the_current_thread() -> None:
    """The conversation list keeps its regression contract."""
    columns = conversation_columns(
        threads=[
            {
                "thread_id": "a",
                "metadata": {"title": "关于水的梦"},
                "created_at": "2026-09-16T10:00:00",
            },
            {"thread_id": "b", "metadata": {}, "created_at": "2026-09-16T11:00:00"},
        ],
        current_thread_id="b",
        messages=[{"type": "human", "content": "我梦见水。"}],
    )
    rendered = html(columns)
    assert "关于水的梦" in rendered
    assert "新对话" in rendered
    assert 'class="current"' in rendered
    assert 'href="/conversations/a"' in rendered
    assert "我梦见水。" in rendered


def test_chat_bubble_renders_both_roles() -> None:
    """User and assistant bubbles stay distinguishable."""
    assert "bubble--user" in html(chat_bubble("hi", role="user"))
    assert "bubble--bot" in html(chat_bubble("hi", role="bot"))


# ---------------------------------------------------------------- evaluation


def make_summary_payload() -> dict[str, object]:
    """Build a minimal valid summary payload.

    Returns:
        The payload.
    """
    return {
        "schema_version": 1,
        "generated_at": "2026-09-16T00:00:00Z",
        "source": "data/evals/full/report_rescored.json",
        "dataset_cases": 60,
        "rescored": True,
        "variants": [
            {
                "key": "single_agent",
                "label": "Single Agent",
                "runs": 60,
                "metrics": {"avg_llm_calls": 1.0, "avg_latency_seconds": 12.3},
            },
            {
                "key": "full_system",
                "label": "Full System (Critic)",
                "runs": 60,
                "metrics": {"avg_llm_calls": 6.23, "avg_latency_seconds": 67.0},
            },
        ],
        "notes": ["test fixture corpus"],
    }


def test_parse_summary_rejects_junk() -> None:
    """Bad artifacts must not be rendered as numbers."""
    assert parse_summary(None) is None
    assert parse_summary({}) is None
    assert parse_summary({"variants": []}) is None
    assert parse_summary({"variants": [{"metrics": {}}]}) is None


def test_parse_summary_reads_the_artifact() -> None:
    """A valid payload becomes typed rows."""
    summary = parse_summary(make_summary_payload())
    assert summary is not None
    assert summary.dataset_cases == 60
    assert summary.rescored is True
    assert [variant.key for variant in summary.variants] == [
        "single_agent",
        "full_system",
    ]


def test_load_evaluation_summary_handles_a_missing_file(tmp_path: Path) -> None:
    """A missing artifact is reported, not guessed."""
    assert load_evaluation_summary(tmp_path / "nope.json") is None


def test_load_evaluation_summary_reads_a_written_file(tmp_path: Path) -> None:
    """The loader reads the versioned artifact."""
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(make_summary_payload()), encoding="utf-8")
    summary = load_evaluation_summary(path)
    assert summary is not None
    assert summary.source.endswith("report_rescored.json")


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        (None, "score", NA_TEXT),
        (4.567, "score", "4.57"),
        (12.34, "seconds", "12.3s"),
        (23137.95, "count", "23,138"),
        (1.0, "rate", "1.00"),
    ],
)
def test_format_value(value: float | None, unit: str, expected: str) -> None:
    """N/A is never printed as a number."""
    assert format_value(value, unit) == expected


def test_metric_specs_cover_the_required_metrics() -> None:
    """The page shows quality, latency and cost."""
    keys = {key for key, _label, _unit in METRIC_SPECS}
    assert {"avg_llm_calls", "avg_latency_seconds", "avg_total_tokens"} <= keys
    assert {
        "theory_differentiation_judge",
        "citation_id_validity",
        "clinical_boundary_judge",
        "overinterpretation_control_judge",
    } <= keys


def test_disclaimer_mentions_the_limits() -> None:
    """The page states what the numbers do not mean."""
    assert "不代表心理诊断有效性" in DISCLAIMER


def test_evaluation_page_renders_missing_artifact_note() -> None:
    """Without the artifact the page explains how to create it."""
    from react_agent.web.evaluation import evaluation_page

    rendered = html(evaluation_page(None))
    assert "export_evaluation_summary.py" in rendered
    assert "TEST FIXTURE" in rendered
