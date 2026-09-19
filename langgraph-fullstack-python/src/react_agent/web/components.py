"""FastHTML components for the PsycheGraph web UI.

The page has three columns - conversations, chat, workflow plus evidence - and
the components here render both the first paint and every fragment the SSE
stream swaps in. Fragments are server-rendered HTML on purpose: the browser
never builds meaning out of JSON, so sanitizing happens in one place (Python).

Nothing in this module reads LangGraph payloads directly; it only consumes the
sanitized `WorkflowEvent` values produced by `react_agent.web.streaming`.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from fasthtml.common import (
    H1,
    A,
    Button,
    Details,
    Div,
    Form,
    P,
    Span,
    Summary,
    Textarea,
    Title,
)

from react_agent.rag.citations import DEFAULT_HEADING
from react_agent.web.citations import cards_by_school
from react_agent.web.events import (
    SCHOOL_LABELS,
    AnswerPayload,
    CitationRef,
    SourceCard,
    SourcesPayload,
    WorkflowEvent,
    school_label,
)
from react_agent.web.nodes import FT, children
from react_agent.web.render import render_answer
from react_agent.web.workflow import workflow_panel

PLACEHOLDER_ID = "assistant-placeholder"
"""Id of the waiting bubble; it is deleted when the run produces its result."""

DELETE_PLACEHOLDER_OOB = "delete:#assistant-placeholder"
"""Out-of-band instruction that removes the waiting bubble."""

BRAND_SUB = "Multi-agent psychoanalytic theory explorer"
BRAND_NOTE = (
    "用于理论学习、文本解释与文学/叙事分析，不用于临床诊断、心理治疗或医学建议。"
)
EVIDENCE_NOTE = (
    "当前演示知识库使用项目自编测试材料验证 RAG 与引用机制，"
    "不是 Freud / Klein / Lacan 原著数据库。"
)
EVIDENCE_EMPTY = "发送一条消息后，这里会显示本轮检索到的理论材料。"
EVIDENCE_SEARCHING = "正在检索本地知识库…检索完成后会按学派列出证据。"
INPUT_HINT = (
    "Enter 发送 · Shift+Enter 换行。分析通常需要 45–80 秒，期间可以查看右侧流程与依据。"
)
INPUT_HINT_BUSY = "分析进行中，完成后可继续输入。"
RUN_FALLBACK_HINT = "可以修改后重新发送。系统不会自动重放本轮，以免重复记录对话。"
QUALITY_UNKNOWN = "质量审核：状态未知"


def split_sources_section(content: str) -> tuple[str, str]:
    """Split a stored answer into its prose and its source block.

    Args:
        content: `AIMessage` content as stored in the checkpoint.

    Returns:
        The prose and the rendered source block (the block is empty when the
        message had no citations).
    """
    marker = f"\n\n{DEFAULT_HEADING}"
    if marker in content:
        prose, _, rest = content.partition(marker)
        return prose, f"{DEFAULT_HEADING}{rest}"
    if content.startswith(DEFAULT_HEADING):
        return "", content
    return content, ""


def chat_bubble(content: FT, *, role: str, element_id: str | None = None) -> FT:
    """Wrap content in a chat bubble.

    Args:
        content: Bubble body.
        role: `user` or `bot`.
        element_id: Optional id for the bubble row (used by the placeholder).

    Returns:
        The bubble row.
    """
    is_user = role == "user"
    attributes: dict[str, str] = {}
    if element_id:
        attributes["id"] = element_id
    return Div(
        Div(
            children(
                Div("你" if is_user else "PsycheGraph", cls="role-label"),
                content,
            ),
            cls=f"bubble bubble--{'user' if is_user else 'bot'}",
        ),
        cls=f"bubble-row bubble-row--{'user' if is_user else 'bot'}",
        **attributes,
    )


def message_bubble(
    message: Mapping[str, Any], citations: Sequence[CitationRef] = ()
) -> FT:
    """Render one chat message from the checkpoint.

    Args:
        message: Message mapping with `type` and `content`.
        citations: Citation references to link, when the panel holds the
            matching run (only the newest answer gets them).

    Returns:
        A bubble with escaped text; citation labels without a matching run stay
        plain text.
    """
    content = str(message.get("content") or "")
    if str(message.get("type") or "") == "human":
        return chat_bubble(content, role="user")
    if citations:
        # The newest answer can link its citations, so the stored source block
        # is dropped: the citation block below carries the same labels.
        prose, _block = split_sources_section(content)
        return chat_bubble(
            Div(*render_answer(prose), citation_block(citations)), role="bot"
        )
    return chat_bubble(Div(*render_answer(content)), role="bot")


def chat_history(
    messages: Sequence[Mapping[str, Any]],
    citations: Sequence[CitationRef] = (),
) -> list[FT]:
    """Render the stored conversation.

    Args:
        messages: Messages from the thread state.
        citations: Citation references of the newest answer.

    Returns:
        One bubble per message; the newest answer is the only one whose
        citation labels can be turned into links.
    """
    if not messages:
        return []
    bubbles = [message_bubble(message) for message in messages[:-1]]
    bubbles.append(message_bubble(messages[-1], citations))
    return bubbles


def evidence_panel(payload: SourcesPayload | None = None) -> FT:
    """Render the evidence panel shell.

    Args:
        payload: Evidence of the newest run, when the page shows a thread that
            already has one; otherwise the body starts empty.

    Returns:
        The panel section.
    """
    return Div(
        children(
            Div(
                children(
                    Span("Theory Evidence", cls="panel-title"),
                    Span("本地知识库", cls="panel-hint"),
                ),
                cls="panel-head",
            ),
            P(EVIDENCE_NOTE, cls="ev-note"),
            Div(
                evidence_body(payload)
                if payload and payload.cards
                else P(EVIDENCE_EMPTY, cls="ev-empty"),
                id="evidence-body",
                cls="ev-body",
                aria_live="polite",
            ),
        ),
        id="evidence-panel",
        cls="panel panel--evidence",
        aria_label="理论依据",
    )


def evidence_card(card: SourceCard) -> FT:
    """Render one sanitized evidence card.

    Args:
        card: `SourceCard` from the sanitized payload.

    Returns:
        A card node with an expandable passage.
    """
    description_parts = [part for part in (card.author, card.title, card.year) if part]
    subtitle = " · ".join(str(part) for part in description_parts)
    meta_bits = [f"evidence_id: {card.evidence_id}"]
    if card.source_label:
        meta_bits.append(f"来源: {card.source_label}")
    if card.page is not None:
        meta_bits.append(f"p. {card.page}")
    if card.section:
        meta_bits.append(f"章节「{card.section}」")
    expandable = card.detail != card.excerpt
    return Div(
        children(
            Div(
                children(
                    Span(card.evidence_id, cls="ev-id"),
                    Span(subtitle, cls="ev-desc") if subtitle else None,
                ),
                cls="ev-card-head",
            ),
            P(card.excerpt, cls="ev-excerpt"),
            Details(
                Summary("展开原文片段", cls="ev-more-summary"),
                P(card.detail, cls="ev-excerpt"),
                cls="ev-more",
            )
            if expandable
            else None,
            Div(" · ".join(meta_bits), cls="ev-meta"),
        ),
        id=f"ev-{card.evidence_id}",
        cls="ev-card",
        data_evidence_card=card.evidence_id,
        role="article",
        aria_label=f"证据 {card.evidence_id}",
    )


def evidence_body(payload: SourcesPayload | None, *, oob: bool = False) -> FT:
    """Render the inner body of the evidence panel.

    Args:
        payload: Sanitized sources payload, or None for the empty state.
        oob: Attach `hx-swap-oob` so the fragment replaces the panel body when
            it arrives as part of another response.

    Returns:
        The body content (the panel swaps this in place).
    """
    attributes: dict[str, str] = {}
    if oob:
        attributes["hx_swap_oob"] = "innerHTML:#evidence-body"
    if payload is None:
        # A fresh run: retrieval has not happened yet, so no claim is made.
        return P(EVIDENCE_SEARCHING, cls="ev-empty", **attributes)
    if not payload.cards:
        reason = f"（{payload.reason}）" if payload.reason else ""
        return P(f"本轮没有可用的本地文献{reason}。", cls="ev-empty", **attributes)
    grouped = cards_by_school(payload.cards)
    sections: list[FT] = []
    for school, cards in grouped.items():
        sections.append(
            Div(
                children(
                    Div(school_label(school), cls="ev-school"),
                    *[evidence_card(card) for card in cards],
                )
            )
        )
    summary = Span(
        f"共 {len(payload.cards)} 条 · 按学派分组",
        cls="ev-meta",
    )
    return Div(summary, *sections, cls="ev-groups", **attributes)


def topbar(*, active: str, thread_id: str | None = None) -> FT:
    """Render the top bar with navigation and drawer toggles.

    Args:
        active: `chat` or `evaluation`.
        thread_id: Current thread, when there is one.

    Returns:
        The header node.
    """
    chat_href = f"/conversations/{thread_id}" if thread_id else "/"
    return Div(
        children(
            Div(
                children(
                    Span("PsycheGraph", cls="brand-name"),
                    Span(BRAND_SUB, cls="brand-sub"),
                ),
                cls="brand",
            ),
            Div(
                children(
                    Button(
                        "流程 / 依据",
                        type="button",
                        cls="drawer-toggle",
                        data_drawer_toggle="right-column",
                        aria_expanded="false",
                        aria_controls="right-column",
                    ),
                    Button(
                        "会话列表",
                        type="button",
                        cls="drawer-toggle",
                        data_drawer_toggle="col-sidebar",
                        aria_expanded="false",
                        aria_controls="col-sidebar",
                    ),
                ),
                cls="drawer-buttons",
            ),
            Div(
                children(
                    A("对话", href=chat_href, cls="active" if active == "chat" else ""),
                    A(
                        "System / Evaluation",
                        href="/evaluation",
                        cls="active" if active == "evaluation" else "",
                    ),
                ),
                cls="topnav",
            ),
        ),
        cls="topbar",
    )


def conversation_sidebar(
    threads: Sequence[Mapping[str, Any]],
    *,
    current_thread_id: str,
) -> FT:
    """Render the conversation list.

    Args:
        threads: Thread records from the LangGraph client.
        current_thread_id: Thread highlighted as current.

    Returns:
        The sidebar column.
    """
    items: list[FT] = []
    for thread in threads:
        thread_id = str(thread.get("thread_id") or "")
        metadata = thread.get("metadata") if isinstance(thread, Mapping) else None
        title = ""
        if isinstance(metadata, Mapping):
            raw_title = metadata.get("title")
            title = str(raw_title) if raw_title else ""
        if not title:
            title = "新对话"
        created = str(thread.get("created_at") or "")[:16].replace("T", " ")
        items.append(
            Div(
                A(
                    children(
                        Span(title, cls="thread-title"),
                        Span(created, cls="thread-meta"),
                    ),
                    href=f"/conversations/{thread_id}",
                    cls="current" if thread_id == current_thread_id else "",
                ),
                cls="thread-item",
            )
        )
    if not items:
        items.append(P("还没有对话。", cls="sidebar-empty"))
    return Div(
        children(
            Div(
                children(
                    Span("Conversations", cls="sidebar-title"),
                ),
                cls="sidebar-head",
            ),
            A("＋ 新对话", href="/new-thread", cls="btn-new"),
            Div(*items, cls="thread-list", role="list", aria_label="对话列表"),
        ),
        id="col-sidebar",
        cls="col col-sidebar",
    )


def composer(thread_id: str, *, disabled: bool = False, oob: bool = False) -> FT:
    """Render the message composer.

    Args:
        thread_id: Current thread.
        disabled: True while a run is streaming (blocks a second submit).
        oob: Attach `hx-swap-oob` so a fresh composer replaces the current one
            instead of being appended a second time.

    Returns:
        The input area.
    """
    attributes: dict[str, str] = {}
    if oob:
        attributes["hx_swap_oob"] = "outerHTML:#input-area"
    return Div(
        children(
            Form(
                children(
                    Textarea(
                        name="msg",
                        id="composer-text",
                        rows="2",
                        placeholder="写下你想分析的材料：梦境、叙事、文学文本或关系片段…",
                        aria_label="输入消息",
                        disabled=disabled,
                    ),
                    Button(
                        "发送",
                        type="submit",
                        id="send-button",
                        disabled=disabled,
                    ),
                ),
                id="composer-form",
                cls="input-form",
                hx_post=f"/conversations/{thread_id}/send-message",
                hx_target="#chatlist",
                hx_swap="beforeend",
                aria_label="发送消息",
            ),
            Div(INPUT_HINT_BUSY if disabled else INPUT_HINT, cls="input-hint"),
        ),
        id="input-area",
        cls="input-area",
        **attributes,
    )


def run_status_bar() -> FT:
    """Render the live run status bar.

    Returns:
        A status bar with a step line (server-updated) and a real elapsed timer.
    """
    return Div(
        children(
            Span("就绪", id="run-status-text"),
            Span("", id="run-elapsed", cls="elapsed"),
        ),
        id="run-status",
        cls="run-status",
        role="status",
        aria_live="polite",
    )


def sse_hub(thread_id: str) -> FT:
    """Render the hidden SSE command centre for one run.

    The element owns the EventSource; its children forward each named event to
    the element that displays it. Keeping every `sse-swap` inside the connecting
    element is what the htmx SSE extension requires.

    Args:
        thread_id: Thread whose stream to follow.

    Returns:
        A hidden container (inserted by the send response).
    """
    return Div(
        children(
            Div(
                sse_swap="workflow",
                hx_target="#workflow-log",
                hx_swap="beforeend",
                hidden=True,
            ),
            Div(
                sse_swap="sources",
                hx_target="#evidence-body",
                hx_swap="innerHTML",
                hidden=True,
            ),
            Div(
                sse_swap="answer",
                hx_target="#chatlist",
                hx_swap="beforeend",
                hidden=True,
            ),
            Div(
                sse_swap="error",
                hx_target="#chatlist",
                hx_swap="beforeend",
                hidden=True,
            ),
            Div(
                sse_swap="close",
                hx_target="#run-status",
                hx_swap="innerHTML",
                hidden=True,
            ),
        ),
        id="sse-hub",
        cls="sse-hub",
        sse_connect=f"/conversations/{thread_id}/stream",
        sse_close="close",
        hidden=True,
        aria_hidden="true",
    )


def assistant_placeholder() -> FT:
    """Render the waiting bubble appended when a message is sent.

    Returns:
        A placeholder bubble that is filled only by a real `answer` event.
    """
    return chat_bubble(
        Div(
            Div(
                children(
                    Span(cls="dot"),
                    Span(cls="dot"),
                    Span(cls="dot"),
                ),
                cls="dots",
                aria_hidden="true",
            ),
            Div("正在执行多智能体分析…", cls="bubble-note"),
            cls="assistant-placeholder",
        ),
        role="bot",
        element_id=PLACEHOLDER_ID,
    )


def theory_tags(payload: AnswerPayload) -> FT:
    """Render the theory perspective tags of an answer.

    Args:
        payload: Answer payload.

    Returns:
        A row of tags; a perspective that did not finish is marked as missing.
    """
    return Div(
        *[
            Span(
                tag.label if tag.completed else f"{tag.label}（未完成）",
                cls="tag tag--done" if tag.completed else "tag tag--missing",
            )
            for tag in payload.theory_tags
        ],
        cls="tag-row",
        aria_label="本次使用的理论视角",
    )


def citation_block(citations: Sequence[CitationRef]) -> FT | None:
    """Render the clickable citation list of an answer.

    Args:
        citations: Citation references of the run that produced the answer.

    Returns:
        The source list, or None when nothing is cited.
    """
    if not citations:
        return None
    lines: list[FT] = []
    for reference in citations:
        lines.append(
            Div(
                Span(
                    f"[{reference.label}]",
                    cls="citation",
                    role="button",
                    tabindex="0",
                    data_citation=reference.label,
                    data_evidence=reference.evidence_id,
                    aria_label=f"查看依据 {reference.label}",
                ),
                Span(reference.description, cls="source-line-text"),
                cls="source-line",
            )
        )
    return Div(
        children(
            Span(DEFAULT_HEADING, cls="source-heading"),
            *lines,
        ),
        cls="source-list",
    )


def citation_list(payload: AnswerPayload) -> FT | None:
    """Render the clickable citation list of an answer payload.

    Args:
        payload: Answer payload.

    Returns:
        The source list, or None when the answer cites nothing.
    """
    return citation_block(payload.citations)


def answer_fragment(event: WorkflowEvent) -> FT:
    """Render an `answer` event as an assistant bubble.

    Args:
        event: Answer event carrying the sanitized payload.

    Returns:
        A chat bubble with the prose, the citation list, the theory tags and the
        quality line.
    """
    payload = event.metadata.answer
    if payload is None:
        return chat_bubble(
            P("本次没有生成可展示的回答。", cls="answer-empty"), role="bot"
        )
    return chat_bubble(
        children(
            *render_answer(payload.text, payload.citations),
            citation_list(payload),
            Div(
                children(
                    theory_tags(payload),
                    Span(payload.status_label, cls="quality"),
                ),
                cls="answer-meta",
            ),
            Div(hx_swap_oob=DELETE_PLACEHOLDER_OOB, cls="oob-only"),
        ),
        role="bot",
    )


def error_fragment(event: WorkflowEvent) -> FT:
    """Render an `error` event as a system bubble.

    Args:
        event: Error event with a safe message.

    Returns:
        A bubble that explains the failure without any traceback.
    """
    return chat_bubble(
        children(
            P(event.message, cls="error-text"),
            P(RUN_FALLBACK_HINT, cls="input-hint"),
            Div(hx_swap_oob=DELETE_PLACEHOLDER_OOB, cls="oob-only"),
        ),
        role="bot",
    )


def progress_fragment(event: WorkflowEvent) -> FT:
    """Render the status-bar line for a workflow event.

    Args:
        event: Workflow event.

    Returns:
        An out-of-band replacement for `#run-status-text`.
    """
    text = event.message or event.label
    return Span(
        f"{event.label}：{text}",
        hx_swap_oob="innerHTML:#run-status-text",
    )


def close_fragment(
    event: WorkflowEvent,
    thread_id: str,
    *,
    remove_placeholder: bool = False,
) -> FT:
    """Render the `close` event: final status line and a usable composer.

    Args:
        event: Close event.
        thread_id: Current thread.
        remove_placeholder: Delete the waiting bubble when no answer or error
            ever arrived (for example when the stream was cut short).

    Returns:
        The new status bar content plus out-of-band replacements for the
        composer and the stream controls.
    """
    elapsed = event.elapsed_seconds
    return Div(
        children(
            Span("分析结束", id="run-status-text"),
            Span(f"服务端用时 {elapsed:.1f}s", cls="elapsed"),
            Div(hx_swap_oob=DELETE_PLACEHOLDER_OOB, cls="oob-only")
            if remove_placeholder
            else None,
            Span(
                cls="stream-controls-reset",
                hx_swap_oob="innerHTML:#stream-controls",
                aria_hidden="true",
            ),
        ),
        Div(composer(thread_id, disabled=False), hx_swap_oob="outerHTML:#input-area"),
    )


def stream_controls() -> FT:
    """Render the (empty) container the SSE hub is inserted into.

    Returns:
        A hidden container; the send response swaps the hub into it, and the
        close event empties it again, which disposes the EventSource.
    """
    return Span(
        "等待发送",
        id="stream-controls",
        cls="stream-controls",
        hidden=True,
        aria_hidden="true",
    )


def page(
    *, title: str, active: str, body: FT, thread_id: str | None = None
) -> tuple[FT, ...]:
    """Wrap content in the application shell.

    Args:
        title: Document title.
        active: Active navigation entry.
        body: Page body (the columns).
        thread_id: Current thread, when there is one.

    Returns:
        FastHTML page parts: title and the shell.
    """
    return (
        Title(title),
        Div(
            children(
                topbar(active=active, thread_id=thread_id),
                body,
            ),
            cls="app",
            hx_ext="sse",
        ),
    )


def conversation_columns(
    *,
    threads: Sequence[Mapping[str, Any]],
    current_thread_id: str,
    messages: Sequence[Mapping[str, Any]],
    citations: Sequence[CitationRef] = (),
    sources: SourcesPayload | None = None,
) -> FT:
    """Render the three columns of the chat page.

    Args:
        threads: Conversation list.
        current_thread_id: Thread being viewed.
        messages: Stored messages.
        citations: Citation references of the newest answer, when the evidence
            panel shows the same run.
        sources: Evidence of the newest run, when the thread has one.

    Returns:
        The columns container.
    """
    return Div(
        children(
            conversation_sidebar(threads, current_thread_id=current_thread_id),
            Div(
                children(
                    Div(
                        children(
                            H1("精神分析理论探索"),
                            P(BRAND_NOTE),
                        ),
                        cls="chat-head",
                    ),
                    Div(
                        *chat_history(messages, citations),
                        id="chatlist",
                        cls="chat-scroll",
                        aria_live="polite",
                        aria_label="对话内容",
                    ),
                    stream_controls(),
                    run_status_bar(),
                    composer(current_thread_id),
                ),
                cls="col col-chat",
            ),
            Div(
                children(
                    workflow_panel(),
                    evidence_panel(sources),
                ),
                id="right-column",
                cls="col col-right",
                aria_label="流程与依据",
            ),
        ),
        cls="columns",
    )


def source_school_order() -> tuple[str, ...]:
    """Return the order the evidence panel groups schools in.

    Returns:
        School names in display order.
    """
    return tuple(SCHOOL_LABELS.keys())
