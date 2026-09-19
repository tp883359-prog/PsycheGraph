"""Safe rendering of the model's answer for the browser.

The answer prose is model output, so it is never inserted as raw HTML: every
piece of text becomes a FastHTML node (which escapes its children) and only the
elements built here - paragraphs, list items, emphasis and citation buttons -
are ever created.

Citation labels are turned into buttons **only** when the label maps to a real
evidence id of the current run. An invented `[E9]` stays plain text, so an
invalid citation can never produce a link.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from fasthtml.common import Br, Div, Li, P, Strong, Ul

from react_agent.web.events import CitationRef
from react_agent.web.nodes import FT

BOLD_RE = re.compile(r"\*\*(?P<text>[^*\n]{1,200})\*\*")
CITATION_RE = re.compile(r"\[(?P<label>[Ee](?P<number>\d{1,3}))\]")
INLINE_RE = re.compile(
    rf"(?P<bold>{BOLD_RE.pattern})|(?P<citation>{CITATION_RE.pattern})"
)
BULLET_RE = re.compile(r"^\s*[-*•]\s+(?P<text>.+)$")
ORDERED_RE = re.compile(r"^\s*\d+[.)]\s+(?P<text>.+)$")
HEADING_RE = re.compile(r"^\s*#{2,4}\s+(?P<text>.+)$")

EMPTY_ANSWER_TEXT = "本次没有生成可展示的回答。"
"""Shown when the graph produced an empty answer."""


def _citation_token(token: str) -> str | None:
    """Return the normalized label of a citation token.

    Args:
        token: Token matched by the inline pattern.

    Returns:
        The label (for example `E1`), or None when the token is not a citation.
    """
    match = CITATION_RE.fullmatch(token)
    if match is None:
        return None
    return f"E{match.group('number')}"


def _inline_nodes(
    text: str,
    citations: Mapping[str, CitationRef],
) -> list[FT]:
    """Split one line into text, emphasis and citation-button nodes.

    Args:
        text: Line content (never trusted).
        citations: Label (`E1`) to reference mapping of the current run.

    Returns:
        FastHTML nodes in reading order.
    """
    nodes: list[FT] = []
    position = 0
    for match in INLINE_RE.finditer(text):
        if match.start() > position:
            nodes.append(text[position : match.start()])
        token = match.group(0)
        label = _citation_token(token)
        if label is not None:
            reference = citations.get(label)
            if reference is None:
                nodes.append(token)
            else:
                nodes.append(
                    Div(
                        token,
                        cls="citation",
                        role="button",
                        tabindex="0",
                        title=f"查看依据 {token}",
                        data_citation=token,
                        data_evidence=reference.evidence_id,
                        aria_label=f"查看依据 {token}：{reference.description}",
                    )
                )
        else:
            bold_match = BOLD_RE.fullmatch(token)
            inner = bold_match.group("text") if bold_match else token
            nodes.append(Strong(inner))
        position = match.end()
    if position < len(text):
        nodes.append(text[position:])
    return nodes


def _paragraph_nodes(
    lines: Sequence[str],
    citations: Mapping[str, CitationRef],
) -> list[FT]:
    """Render a run of non-list lines as one paragraph.

    Args:
        lines: Paragraph lines.
        citations: Citation label mapping.

    Returns:
        A single paragraph node; single newlines become line breaks.
    """
    children: list[FT] = []
    for index, line in enumerate(lines):
        if index:
            children.append(Br())
        children.extend(_inline_nodes(line, citations))
    return [P(*children)] if children else []


def render_answer(
    text: str,
    citations: Sequence[CitationRef] = (),
) -> list[FT]:
    """Render the answer prose as safe FastHTML nodes.

    Supports the formatting the model actually produces: paragraphs, `-`/`*`
    bullet lists, numbered lists, `##` headings and `**emphasis**`, plus
    citation labels.

    Args:
        text: `final_result.final_response` from the graph.
        citations: Citation references of this run.

    Returns:
        A list of block-level FastHTML nodes.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return [P(EMPTY_ANSWER_TEXT, cls="answer-empty")]
    by_label: Mapping[str, CitationRef] = {ref.label: ref for ref in citations}
    blocks: list[FT] = []
    paragraph: list[str] = []
    items: list[str] = []
    mode = "none"

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            blocks.extend(_paragraph_nodes(paragraph, by_label))
            paragraph = []

    def flush_list() -> None:
        nonlocal items, mode
        if items:
            blocks.append(
                Ul(
                    *[Li(*_inline_nodes(item, by_label)) for item in items],
                    cls="answer-list",
                )
            )
            items = []
        mode = "none"

    for raw_line in cleaned.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            flush_list()
            continue
        heading = HEADING_RE.match(line)
        if heading is not None:
            flush_paragraph()
            flush_list()
            blocks.append(
                Div(
                    *_inline_nodes(heading.group("text"), by_label),
                    cls="answer-heading",
                )
            )
            continue
        bullet = BULLET_RE.match(line) or ORDERED_RE.match(line)
        if bullet is not None:
            flush_paragraph()
            mode = "list"
            items.append(bullet.group("text"))
            continue
        if mode == "list":
            flush_list()
        paragraph.append(stripped)
    flush_paragraph()
    flush_list()
    return blocks or [P(EMPTY_ANSWER_TEXT, cls="answer-empty")]


def render_plain_note(text: str, *, css_class: str = "note") -> FT:
    """Render a short system note without formatting.

    Args:
        text: Note text.
        css_class: CSS class for the wrapper.

    Returns:
        A paragraph node.
    """
    return P(text, cls=css_class)


def node_text(value: Any) -> str:
    """Return a display string for an arbitrary value.

    Args:
        value: Any value from a payload.

    Returns:
        A whitespace-collapsed string.
    """
    return re.sub(r"\s+", " ", str(value or "")).strip()
