"""Citation rendering for retrieved theory evidence.

The rule is simple and strict: a citation may only contain metadata that really
exists in the knowledge base. A page number is printed only when the loader read
a real page, otherwise the section is used, and when neither exists the work
title plus the internal evidence id is shown. Nothing is ever invented.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from react_agent.schemas import EvidenceItem

DEFAULT_HEADING = "理论依据（本地知识库）"


def evidence_label(index: int) -> str:
    """Return the short label used for the n-th evidence item.

    Args:
        index: Zero-based position in the evidence list.

    Returns:
        A label such as `[E1]`.
    """
    return f"[E{index + 1}]"


def build_label_map(items: Sequence[EvidenceItem]) -> dict[str, str]:
    """Map each evidence id to its short display label.

    Args:
        items: Evidence items in the order they were offered to the model.

    Returns:
        Mapping from `evidence_id` to `[E1]`-style label.
    """
    return {item.evidence_id: evidence_label(index) for index, item in enumerate(items)}


def describe_source(item: EvidenceItem) -> str:
    """Return the bibliographic part of a citation, without the label.

    Args:
        item: Evidence item produced by the retriever.

    Returns:
        A citation body such as `Freud, 《梦的解析》, p. 42`. Only real metadata
        is used; missing fields are skipped instead of guessed.
    """
    parts: list[str] = []
    if item.author:
        parts.append(item.author)
    title = item.work_title or item.title
    if title:
        parts.append(f"《{title}》" if not title.startswith("《") else title)
    if item.year is not None:
        parts.append(str(item.year))
    body = ", ".join(parts) if parts else f"未标注来源（{item.source_id}）"
    if item.page is not None:
        return f"{body}, p. {item.page}"
    if item.section:
        return f"{body}, 章节「{item.section}」"
    return f"{body}（evidence_id: {item.evidence_id}）"


def format_citation(item: EvidenceItem, label: str) -> str:
    """Render one full citation line.

    Args:
        item: Evidence item.
        label: Short label from :func:`build_label_map`.

    Returns:
        A line such as `[E1] Freud, 《梦的解析》, p. 42`.
    """
    return f"{label} {describe_source(item)}"


def render_sources_section(
    items: Iterable[EvidenceItem],
    *,
    heading: str = DEFAULT_HEADING,
) -> str:
    """Render the source list appended to the user-visible answer.

    Args:
        items: Evidence items actually used by the synthesis.
        heading: Section heading.

    Returns:
        The rendered block, or an empty string when there is nothing to cite.
    """
    ordered: list[EvidenceItem] = list(items)
    if not ordered:
        return ""
    labels = build_label_map(ordered)
    lines = [heading]
    for item in ordered:
        lines.append(format_citation(item, labels[item.evidence_id]))
    return "\n".join(lines)
