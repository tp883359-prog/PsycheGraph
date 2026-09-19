"""Evidence sanitization, citation labels and evidence cards.

Two jobs live here:

* Turn raw retrieval payloads (`evidence_by_school`) into `SourceCard`s that are
  safe to send to a browser: no `source_path`, no embeddings, no Chroma ids, no
  absolute Windows paths, truncated excerpts.
* Map the answer's citation labels (`[E1]`, `[E2]`, ...) onto real evidence ids,
  using the order the finalizer used (`final_result.used_evidence_ids`). The
  mapping never guesses a source from the answer text.

The bibliographic line mirrors `react_agent.rag.citations.describe_source`, so
the panel shows exactly the metadata the product would print.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from react_agent.web.events import (
    CitationRef,
    SourceCard,
    SourcesPayload,
    school_label,
)

EXCERPT_CHARS = 220
"""Length of the excerpt shown collapsed in an evidence card."""

DETAIL_CHARS = 900
"""Length of the passage shown when the reader expands a card."""

MAX_ID_CHARS = 80
"""Evidence ids are short slugs; anything longer is truncated."""

WHITESPACE_RE = re.compile(r"\s+")
PATH_SEPARATOR_RE = re.compile(r"[\\/]+")
DRIVE_RE = re.compile(r"^[A-Za-z]:$")
HOME_RE = re.compile(r"^(users|home|documents and settings)$", re.IGNORECASE)
URL_RE = re.compile(r"^[a-z]+://", re.IGNORECASE)

SAFE_SCHOOLS: tuple[str, ...] = ("freudian", "object_relations", "lacanian", "general")
"""Schools the evidence panel may group by."""


def clean_text(value: Any, limit: int) -> str:
    """Collapse whitespace and truncate a value for display.

    Args:
        value: Raw value from the retrieval payload.
        limit: Maximum number of characters to keep.

    Returns:
        A single-line string, truncated with an ellipsis when needed.
    """
    text = WHITESPACE_RE.sub(" ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "…"


def looks_like_local_path(value: str) -> bool:
    r"""Report whether a string looks like a local filesystem path.

    Args:
        value: Candidate string.

    Returns:
        True for Windows paths (`C:\...`), UNC paths, home directories and
        POSIX absolute paths.
    """
    text = value.strip()
    if not text:
        return False
    if URL_RE.match(text):
        return True
    parts = [part for part in PATH_SEPARATOR_RE.split(text) if part]
    if len(parts) > 1:
        return True
    if DRIVE_RE.match(text[:2]):
        return True
    if text.startswith("~"):
        return True
    return bool(HOME_RE.match(text))


def safe_source_label(value: Any) -> str | None:
    r"""Reduce a source reference to a short, safe label.

    Only the last segment survives, and only when it is harmless:

    * URLs are dropped entirely.
    * A bare directory name from an absolute or user path is dropped, because
      `C:\Users\someone` would leak the account name; a real file name inside
      such a path (`.../notes.md`) is kept.
    * Drive letters and home directories never appear in the result.

    Args:
        value: Raw `source_id` or `source_path` from the index.

    Returns:
        A short label such as `notes.md`, or None when nothing safe remains.
    """
    if value is None:
        return None
    raw = str(value).strip()
    if not raw or URL_RE.match(raw):
        return None
    parts = [part for part in PATH_SEPARATOR_RE.split(raw) if part]
    candidate = parts[-1] if parts else raw
    if DRIVE_RE.match(candidate[:2]) or looks_like_local_path(candidate):
        return None
    if ":" in candidate:
        return None
    absolute = DRIVE_RE.match(raw[:2]) is not None or raw.startswith(("/", "\\", "~"))
    user_path = any(HOME_RE.match(part) for part in parts)
    if (absolute or user_path) and "." not in candidate:
        return None
    return clean_text(candidate, MAX_ID_CHARS)


def safe_id(value: Any) -> str | None:
    """Return a short, safe evidence id.

    Args:
        value: Raw evidence id.

    Returns:
        The cleaned id, or None when it is missing or path-like.
    """
    if value is None:
        return None
    raw = str(value).strip()
    if not raw or looks_like_local_path(raw) or ":" in raw:
        return None
    return clean_text(raw, MAX_ID_CHARS) or None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip()
    return int(text) if text.isdigit() else None


def sanitize_card(raw: Any, school: str) -> SourceCard | None:
    """Convert one raw evidence dict into a sanitized card.

    Args:
        raw: Raw evidence entry (a JSON dict from the graph state).
        school: School the retrieval was performed for.

    Returns:
        A card ready for the browser, or None when the entry carries no usable
        id or text.
    """
    if not isinstance(raw, Mapping):
        return None
    evidence_id = safe_id(raw.get("evidence_id"))
    text = str(raw.get("text") or "").strip()
    if evidence_id is None or not text:
        return None
    resolved_school = str(raw.get("school") or school)
    if resolved_school not in SAFE_SCHOOLS:
        resolved_school = school if school in SAFE_SCHOOLS else "general"
    title_value = raw.get("work_title") or raw.get("title")
    title = clean_text(title_value, 120) if title_value else None
    author_value = raw.get("author")
    author = clean_text(author_value, 80) if author_value else None
    section_value = raw.get("section")
    section = clean_text(section_value, 80) if section_value else None
    source_label = safe_source_label(raw.get("source_id"))
    if source_label is None:
        source_label = safe_source_label(raw.get("source_path"))
    return SourceCard(
        evidence_id=evidence_id,
        school=resolved_school,
        label=school_label(resolved_school),
        title=title,
        author=author,
        year=_int_or_none(raw.get("year")),
        page=_int_or_none(raw.get("page")),
        section=section,
        excerpt=clean_text(text, EXCERPT_CHARS),
        detail=clean_text(text, DETAIL_CHARS),
        source_label=source_label,
    )


def describe_card(card: SourceCard) -> str:
    """Render the bibliographic line of a card.

    Mirrors `react_agent.rag.citations.describe_source`: only real metadata is
    printed, and a missing page or year is skipped rather than guessed.

    Args:
        card: Sanitized evidence card.

    Returns:
        A line such as `Freud, 《梦的解析》, p. 42`.
    """
    parts: list[str] = []
    if card.author:
        parts.append(card.author)
    if card.title:
        title = card.title
        parts.append(title if title.startswith("《") else f"《{title}》")
    if card.year is not None:
        parts.append(str(card.year))
    fallback_id = card.source_label or card.evidence_id
    body = ", ".join(parts) if parts else f"未标注来源（{fallback_id}）"
    if card.page is not None:
        return f"{body}, p. {card.page}"
    if card.section:
        return f"{body}, 章节「{card.section}」"
    return f"{body}（evidence_id: {card.evidence_id}）"


def build_sources_payload(
    evidence_by_school: Any,
    evidence_meta: Any = None,
) -> SourcesPayload:
    """Build the `sources` payload from a raw evidence update.

    Args:
        evidence_by_school: Raw `evidence_by_school` mapping.
        evidence_meta: Raw `evidence_meta` mapping from the same update.

    Returns:
        Sanitized cards (deduplicated by evidence id, school order preserved)
        plus the retrieval counts and availability flag.
    """
    grouped: dict[str, list[SourceCard]] = {school: [] for school in SAFE_SCHOOLS}
    seen: set[str] = set()
    if isinstance(evidence_by_school, Mapping):
        for school in SAFE_SCHOOLS:
            entries = evidence_by_school.get(school)
            if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
                continue
            for entry in entries:
                card = sanitize_card(entry, school)
                if card is None or card.evidence_id in seen:
                    continue
                seen.add(card.evidence_id)
                grouped[card.school].append(card)
    counts: dict[str, int] = {school: len(items) for school, items in grouped.items()}
    available = False
    reason: str | None = None
    if isinstance(evidence_meta, Mapping):
        available = bool(evidence_meta.get("available"))
        raw_reason = evidence_meta.get("reason")
        reason = clean_text(raw_reason, 60) if raw_reason else None
        raw_counts = evidence_meta.get("counts")
        if isinstance(raw_counts, Mapping):
            counts = {str(key): int(value or 0) for key, value in raw_counts.items()}
    cards = tuple(
        card for school in SAFE_SCHOOLS for card in grouped.get(school, []) if card
    )
    return SourcesPayload(
        cards=cards,
        counts=counts,
        available=available or bool(cards),
        reason=reason,
    )


def citation_refs(
    used_evidence_ids: Any,
    cards: Sequence[SourceCard],
) -> tuple[CitationRef, ...]:
    """Map the answer's citation labels to real evidence ids.

    Args:
        used_evidence_ids: `final_result.used_evidence_ids`, in citation order.
        cards: Sanitized cards of the current run.

    Returns:
        One reference per label that resolves to a card. Unknown ids are
        dropped: an invalid citation must not produce a link.
    """
    if not isinstance(used_evidence_ids, Sequence) or isinstance(
        used_evidence_ids, (str, bytes)
    ):
        return ()
    by_id = {card.evidence_id: card for card in cards}
    refs: list[CitationRef] = []
    seen: set[str] = set()
    for index, raw_id in enumerate(used_evidence_ids):
        evidence_id = safe_id(raw_id)
        if evidence_id is None or evidence_id in seen:
            continue
        card = by_id.get(evidence_id)
        if card is None:
            continue
        seen.add(evidence_id)
        refs.append(
            CitationRef(
                label=f"E{len(refs) + 1}",
                evidence_id=evidence_id,
                school=card.school,
                description=describe_card(card),
            )
        )
    return tuple(refs)


def cards_by_school(cards: Sequence[SourceCard]) -> dict[str, list[SourceCard]]:
    """Group cards by school, keeping only the schools that have material.

    Args:
        cards: Sanitized cards.

    Returns:
        Mapping from school name to its cards.
    """
    grouped: dict[str, list[SourceCard]] = {}
    for card in cards:
        grouped.setdefault(card.school, []).append(card)
    return grouped
