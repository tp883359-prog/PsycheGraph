"""Web layer of PsycheGraph (Phase 9).

The package turns a LangGraph run into a small, sanitized event stream and
renders it with FastHTML + HTMX + SSE. `react_agent.app` re-exports the
FastHTML instance so `langgraph.json` keeps working unchanged.
"""

from __future__ import annotations

__all__ = ["events", "streaming", "citations", "render", "workflow", "components"]
