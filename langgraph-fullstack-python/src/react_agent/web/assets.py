"""Static assets (CSS and JavaScript) for the PsycheGraph web UI.

The stylesheet and the small script are kept as Python strings and inlined into
the page, so the demo has no build step and no static file routes to configure.
The look is deliberately academic: text first, calm colours, no mystical
imagery, no brain pictures.

The script only does presentation work - elapsed timer, citation clicks,
drawers and the textarea's Enter behaviour. All state changes arrive from the
server as SSE-swapped HTML.
"""

from __future__ import annotations

CUSTOM_CSS = """
:root {
  --ink: #1b1f24;
  --ink-soft: #4b5563;
  --ink-faint: #6b7280;
  --paper: #fbfbfa;
  --paper-alt: #f4f4f2;
  --line: #e2e2df;
  --line-strong: #cfcfc9;
  --accent: #2f5d50;
  --accent-soft: #e8f0ec;
  --user-bubble: #eef2f7;
  --bot-bubble: #ffffff;
  --warn: #8a5a1a;
  --warn-soft: #fdf4e3;
  --fail: #8c2f2f;
  --fail-soft: #fbeceb;
  --radius: 12px;
  --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  --serif: "Iowan Old Style", "Source Han Serif SC", "Songti SC", Georgia, serif;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", Roboto, sans-serif;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; height: 100%; }
body {
  background: var(--paper);
  color: var(--ink);
  font-family: var(--sans);
  font-size: 15px;
  line-height: 1.6;
}
a { color: var(--accent); }
button, .citation { font: inherit; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

.app { display: flex; flex-direction: column; height: 100vh; }
.topbar {
  display: flex; align-items: center; gap: 16px;
  padding: 10px 20px; border-bottom: 1px solid var(--line);
  background: var(--paper); position: sticky; top: 0; z-index: 20;
}
.brand { display: flex; flex-direction: column; }
.brand-name { font-weight: 600; letter-spacing: .02em; }
.brand-sub { font-size: 12px; color: var(--ink-faint); }
.topnav { margin-left: auto; display: flex; gap: 8px; align-items: center; }
.topnav a, .topnav button {
  padding: 6px 12px; border-radius: 999px; border: 1px solid var(--line);
  background: var(--paper); color: var(--ink-soft); text-decoration: none;
  font-size: 13px; cursor: pointer;
}
.topnav a.active { border-color: var(--accent); color: var(--accent); background: var(--accent-soft); }
.columns { flex: 1; display: flex; min-height: 0; }

.col-sidebar {
  width: 260px; flex: 0 0 260px; border-right: 1px solid var(--line);
  background: var(--paper-alt); display: flex; flex-direction: column; min-height: 0;
}
.sidebar-head { padding: 14px 16px 8px; }
.sidebar-title { font-size: 12px; text-transform: uppercase; letter-spacing: .08em; color: var(--ink-faint); }
.btn-new {
  display: block; margin: 0 16px 12px; padding: 8px 12px; text-align: center;
  border: 1px solid var(--line-strong); border-radius: 999px; background: var(--paper);
  color: var(--ink); text-decoration: none; font-size: 13px;
}
.thread-list { list-style: none; margin: 0; padding: 0 8px 16px; overflow-y: auto; }
.thread-item a {
  display: block; padding: 8px 10px; border-radius: 8px; text-decoration: none;
  color: var(--ink); border-left: 3px solid transparent;
}
.thread-item a:hover { background: #ececea; }
.thread-item a.current { background: var(--accent-soft); border-left-color: var(--accent); }
.thread-title { font-size: 13px; display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.thread-meta { font-size: 11px; color: var(--ink-faint); }
.sidebar-empty { padding: 0 16px 16px; font-size: 12px; color: var(--ink-faint); }

.col-chat { flex: 1 1 auto; display: flex; flex-direction: column; min-width: 0; min-height: 0; }
.chat-head {
  padding: 10px 24px; border-bottom: 1px solid var(--line); background: var(--paper);
  display: flex; align-items: baseline; gap: 12px;
}
.chat-head h1 { font-size: 15px; margin: 0; font-weight: 600; }
.chat-head p { margin: 0; font-size: 12px; color: var(--ink-faint); }
.chat-scroll { flex: 1; overflow-y: auto; padding: 20px 24px 8px; min-height: 0; }
.bubble-row { display: flex; margin: 0 0 18px; }
.bubble-row--user { justify-content: flex-end; }
.bubble {
  max-width: min(760px, 88%); padding: 12px 16px; border-radius: var(--radius);
  border: 1px solid var(--line); box-shadow: 0 1px 2px rgba(16, 24, 40, .04);
}
.bubble--user { background: var(--user-bubble); border-top-right-radius: 4px; white-space: pre-wrap; }
.bubble--bot { background: var(--bot-bubble); border-top-left-radius: 4px; font-family: var(--serif); }
.bubble--bot p { margin: 0 0 10px; }
.answer-list { margin: 0 0 10px; padding-left: 22px; }
.answer-heading { font-weight: 600; margin: 12px 0 6px; }
.answer-empty, .note { color: var(--ink-faint); font-size: 13px; }
.role-label { font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: var(--ink-faint); margin-bottom: 6px; }

.citation {
  display: inline-block; padding: 1px 6px; margin: 0 1px; border-radius: 6px;
  border: 1px solid var(--line-strong); background: var(--paper-alt);
  color: var(--accent); font-size: 12px; cursor: pointer; font-family: var(--mono);
}
.citation:hover { background: var(--accent-soft); }
.citation:focus-visible { outline: 2px solid var(--accent); }

.answer-meta { margin-top: 12px; padding-top: 10px; border-top: 1px dashed var(--line); display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center; }
.tag-row { display: flex; gap: 6px; flex-wrap: wrap; }
.tag { font-size: 11px; padding: 2px 8px; border-radius: 999px; border: 1px solid var(--line-strong); color: var(--ink-soft); }
.tag--done { border-color: var(--accent); color: var(--accent); background: var(--accent-soft); }
.tag--missing { border-color: var(--fail); color: var(--fail); background: var(--fail-soft); }
.quality { font-size: 12px; color: var(--ink-soft); }
.source-list { margin: 10px 0 0; padding-top: 8px; border-top: 1px dashed var(--line); font-size: 13px; }
.source-line { display: flex; gap: 8px; align-items: baseline; margin: 2px 0; }
.source-school { font-size: 11px; color: var(--ink-faint); min-width: 52px; }

.run-status { padding: 6px 24px; font-size: 12px; color: var(--ink-soft); min-height: 28px; display: flex; gap: 10px; align-items: center; }
.run-status .elapsed { font-family: var(--mono); color: var(--ink-faint); }
.input-area { border-top: 1px solid var(--line); padding: 12px 24px 18px; background: var(--paper); }
.input-form { display: flex; gap: 10px; align-items: flex-end; }
.input-form textarea {
  flex: 1; resize: vertical; min-height: 52px; max-height: 200px; padding: 10px 12px;
  border: 1px solid var(--line-strong); border-radius: 10px; font: inherit; color: inherit;
  background: #fff;
}
.input-form textarea:disabled { background: var(--paper-alt); color: var(--ink-faint); }
.input-form button {
  padding: 10px 18px; border-radius: 10px; border: 1px solid var(--accent);
  background: var(--accent); color: #fff; cursor: pointer;
}
.input-form button:disabled { background: var(--paper-alt); border-color: var(--line-strong); color: var(--ink-faint); cursor: not-allowed; }
.input-hint { font-size: 11px; color: var(--ink-faint); margin-top: 6px; }
.input-hint code { font-family: var(--mono); }

.col-right {
  width: 380px; flex: 0 0 380px; border-left: 1px solid var(--line);
  background: var(--paper-alt); overflow-y: auto; padding: 12px; min-height: 0;
}
.panel {
  background: #fff; border: 1px solid var(--line); border-radius: var(--radius);
  margin-bottom: 12px; overflow: hidden;
}
.panel-head {
  display: flex; align-items: baseline; justify-content: space-between; gap: 8px;
  padding: 10px 12px; border-bottom: 1px solid var(--line);
}
.panel-head--sub { border-top: 1px solid var(--line); }
.panel-title { font-weight: 600; font-size: 13px; letter-spacing: .02em; }
.panel-subtitle { font-size: 12px; text-transform: uppercase; letter-spacing: .08em; color: var(--ink-faint); }
.panel-hint { font-size: 11px; color: var(--ink-faint); }

.wf-group { padding: 8px 12px 4px; }
.wf-phase-head { display: flex; gap: 8px; align-items: baseline; margin-bottom: 4px; }
.wf-phase-title { font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: var(--ink-faint); }
.wf-phase-note { font-size: 11px; color: var(--ink-faint); }
.wf-list { list-style: none; margin: 0; padding: 0; }
.wf-row { display: flex; gap: 8px; align-items: flex-start; padding: 5px 0; border-bottom: 1px dotted var(--line); }
.wf-row:last-child { border-bottom: none; }
.wf-symbol { width: 16px; text-align: center; color: var(--ink-faint); }
.wf-text { flex: 1; min-width: 0; }
.wf-label { font-size: 13px; }
.wf-message { font-size: 11px; color: var(--ink-faint); }
.wf-state { text-align: right; font-size: 11px; color: var(--ink-faint); white-space: nowrap; }
.wf-duration { font-family: var(--mono); margin-left: 6px; }
.wf-row--running .wf-symbol, .wf-row--running .wf-label { color: var(--accent); font-weight: 600; }
.wf-row--completed .wf-symbol { color: var(--accent); }
.wf-row--revising .wf-symbol, .wf-row--revising .wf-label { color: var(--warn); font-weight: 600; }
.wf-row--failed .wf-symbol, .wf-row--failed .wf-label, .wf-row--failed .wf-state { color: var(--fail); }
.wf-row--skipped { opacity: .55; }
.wf-log { max-height: 200px; overflow-y: auto; padding: 8px 12px 12px; font-size: 12px; }
.wf-log-line { display: flex; gap: 8px; align-items: baseline; padding: 2px 0; }
.wf-log-time { font-family: var(--mono); color: var(--ink-faint); min-width: 34px; }
.wf-log-label { color: var(--ink-soft); min-width: 96px; }
.wf-log-message { color: var(--ink); }
.wf-log-duration { font-family: var(--mono); color: var(--ink-faint); }
.wf-log-line--failed .wf-log-message { color: var(--fail); }

.ev-body { padding: 10px 12px 14px; }
.ev-note { font-size: 11px; color: var(--ink-faint); margin: 0 0 8px; }
.ev-school { font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: var(--ink-faint); margin: 10px 0 4px; }
.ev-card { border: 1px solid var(--line); border-radius: 10px; padding: 8px 10px; margin-bottom: 8px; background: var(--paper); }
.ev-card:target, .ev-card.flash { border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent-soft); }
.ev-card-head { display: flex; gap: 8px; align-items: baseline; }
.ev-id { font-family: var(--mono); font-size: 11px; color: var(--ink-faint); }
.ev-desc { font-size: 13px; }
.ev-excerpt { font-size: 12px; color: var(--ink-soft); margin: 6px 0 0; font-family: var(--serif); }
.ev-more { font-size: 12px; margin-top: 6px; }
.ev-more summary { cursor: pointer; color: var(--accent); }
.ev-meta { font-size: 11px; color: var(--ink-faint); margin-top: 4px; }
.ev-empty { font-size: 12px; color: var(--ink-faint); }

.assistant-placeholder .dots { display: inline-flex; gap: 4px; }
.assistant-placeholder .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); opacity: .4; animation: pulse 1.4s infinite; }
.assistant-placeholder .dot:nth-child(2) { animation-delay: .2s; }
.assistant-placeholder .dot:nth-child(3) { animation-delay: .4s; }
@keyframes pulse { 0%,100% { opacity: .25; } 50% { opacity: .9; } }
@media (prefers-reduced-motion: reduce) { .assistant-placeholder .dot { animation: none; } }

.error-card { border-color: var(--fail); background: var(--fail-soft); }
.error-card .retry { font-size: 12px; color: var(--fail); text-decoration: underline; }

.drawer-buttons { display: none; }
@media (max-width: 1180px) {
  .col-right {
    position: fixed; top: 53px; right: 0; bottom: 0; width: min(420px, 92vw);
    transform: translateX(102%); transition: transform .2s ease; z-index: 30;
    box-shadow: -8px 0 24px rgba(16, 24, 40, .12);
  }
  .col-right.open { transform: translateX(0); }
  .drawer-buttons { display: flex; gap: 8px; }
}
@media (max-width: 860px) {
  .col-sidebar { display: none; }
  .col-sidebar.open { display: flex; position: fixed; top: 53px; bottom: 0; left: 0; width: min(300px, 90vw); z-index: 30; }
  .bubble { max-width: 100%; }
  .chat-scroll { padding: 16px 14px 4px; }
  .input-area { padding: 10px 14px 16px; }
}
"""


APP_JS = """
(function () {
  "use strict";

  function byId(id) { return document.getElementById(id); }

  function setDrawer(name, open) {
    var el = byId(name);
    if (!el) return;
    el.classList.toggle("open", open);
    var button = document.querySelector('[data-drawer-toggle="' + name + '"]');
    if (button) button.setAttribute("aria-expanded", open ? "true" : "false");
  }

  function wireDrawerToggles() {
    document.querySelectorAll("[data-drawer-toggle]").forEach(function (button) {
      button.addEventListener("click", function () {
        var target = byId(button.getAttribute("data-drawer-toggle"));
        var open = target ? !target.classList.contains("open") : false;
        setDrawer(button.getAttribute("data-drawer-toggle"), open);
      });
    });
    document.addEventListener("keydown", function (event) {
      if (event.key !== "Escape") return;
      setDrawer("right-column", false);
      setDrawer("col-sidebar", false);
    });
  }

  // Enter sends, Shift+Enter makes a new line. Plain DOM on purpose: the
  // `hx-on-` attributes FastHTML emits map to htmx-internal event names.
  function wireComposer() {
    var form = byId("composer-form");
    var box = byId("composer-text");
    if (!form || !box) return;
    box.addEventListener("keydown", function (event) {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        if (!box.disabled && box.value.trim()) {
          if (form.requestSubmit) form.requestSubmit(); else form.submit();
        }
      }
    });
  }

  function flashCard(card) {
    if (!card) return;
    card.classList.add("flash");
    window.setTimeout(function () { card.classList.remove("flash"); }, 2000);
  }

  function focusEvidence(evidenceId) {
    var card = document.querySelector('[data-evidence-card="' + evidenceId + '"]');
    if (!card) return;
    setDrawer("right-column", true);
    card.scrollIntoView({ block: "center", behavior: "smooth" });
    var details = card.querySelector("details");
    if (details) details.open = true;
    flashCard(card);
  }

  function wireCitations() {
    document.addEventListener("click", function (event) {
      var target = event.target.closest("[data-evidence]");
      if (!target) return;
      event.preventDefault();
      focusEvidence(target.getAttribute("data-evidence"));
    });
    document.addEventListener("keydown", function (event) {
      if (event.key !== "Enter" && event.key !== " ") return;
      var target = event.target.closest("[data-evidence]");
      if (!target) return;
      event.preventDefault();
      focusEvidence(target.getAttribute("data-evidence"));
    });
  }

  // Real elapsed time of the current run: counted from the moment the browser
  // starts the SSE stream, never an estimate.
  var timer = null;
  var timerStart = 0;
  function startTimer() {
    stopTimer();
    timerStart = Date.now();
    var node = byId("run-elapsed");
    if (!node) return;
    node.textContent = "已用时 0s";
    timer = window.setInterval(function () {
      var seconds = Math.floor((Date.now() - timerStart) / 1000);
      node.textContent = "已用时 " + seconds + "s";
    }, 1000);
  }
  function stopTimer() {
    if (timer !== null) { window.clearInterval(timer); timer = null; }
  }
  function wireTimer() {
    document.addEventListener("htmx:sseOpen", function () {
      startTimer();
      // On a narrow window the workflow panel is a drawer; open it so the run is
      // visible while it happens, and close it again when the run ends so the
      // composer is reachable without an extra tap.
      setDrawer("right-column", true);
    });
    document.addEventListener("htmx:sseClose", function () {
      stopTimer();
      setDrawer("right-column", false);
    });
    document.addEventListener("htmx:sseError", function () {
      stopTimer();
      setDrawer("right-column", false);
    });
    document.addEventListener("htmx:beforeRequest", function (event) {
      var elt = event.detail && event.detail.elt;
      if (elt && elt.id === "composer-form") {
        var box = byId("composer-text");
        if (box) box.value = "";
      }
    });
  }

  function keepChatInView() {
    var list = byId("chatlist");
    if (!list || !window.MutationObserver) return;
    var observer = new MutationObserver(function () {
      list.scrollTop = list.scrollHeight;
    });
    observer.observe(list, { childList: true });
  }

  function init() {
    wireDrawerToggles();
    wireComposer();
    wireCitations();
    wireTimer();
    keepChatInView();
    // Limit notices answer with a real 413/429; htmx does not swap error
    // responses by default, so those two statuses are opted in here. Everything
    // else (a real server error) stays unswapped.
    document.addEventListener("htmx:beforeSwap", function (event) {
      var status = event.detail && event.detail.xhr ? event.detail.xhr.status : 0;
      if (status === 413 || status === 429) {
        event.detail.shouldSwap = true;
        event.detail.isError = false;
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
"""
