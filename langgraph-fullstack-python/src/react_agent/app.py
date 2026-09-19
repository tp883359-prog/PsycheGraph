"""PsycheGraph chat application entry point.

`langgraph.json` points the HTTP app at this module
(`./src/react_agent/app.py:app`), so this file keeps a stable importable `app`
while the implementation lives in `react_agent.web`:

    react_agent/web/app.py        FastHTML instance and headers
    react_agent/web/routes.py     pages, send-message, SSE stream
    react_agent/web/streaming.py  LangGraph stream -> sanitized UI events
    react_agent/web/components.py chat, panels, layout
    react_agent/web/workflow.py   workflow panel (node states)
    react_agent/web/citations.py  evidence sanitization and citation labels

Phase 9 replaced the Phase 1 template UI here; the graph
(`react_agent.graph:graph`) is untouched by the web layer.
"""

from __future__ import annotations

from react_agent.web.app import app

__all__ = ["app"]
