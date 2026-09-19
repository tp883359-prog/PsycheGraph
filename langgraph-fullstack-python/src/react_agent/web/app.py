"""FastHTML application for PsycheGraph.

`langgraph.json` loads `src/react_agent/app.py:app`; that module re-exports the
`app` defined here, so the HTTP application path stays compatible while the
implementation lives in this package.

The app is a custom route layer next to the LangGraph API: it renders pages,
queues a message, and streams one run as sanitized SSE events. It never touches
graph state itself.

Front-end assets are served by this app: htmx and the SSE extension are vendored
in `react_agent/static` (see `scripts/fetch_static_assets.py`), so a page load
does not depend on a CDN and cannot break when a CDN publishes a new version.
"""

from __future__ import annotations

from fasthtml.common import (
    FastHTML,
    Meta,
    Script,
    Style,
)

from react_agent.logging_config import configure_logging
from react_agent.web.assets import APP_JS, CUSTOM_CSS
from react_agent.web.routes import register_routes

configure_logging()

HTMX_SCRIPT = Script(src="/static/htmx.min.js")
"""htmx, served by this app (vendored 2.0.7)."""

SSE_EXT_SCRIPT = Script(src="/static/htmx-ext-sse.js")
"""SSE extension, served by this app (vendored 2.2.1)."""

META_TAGS = (
    Meta(charset="utf-8"),
    Meta(name="viewport", content="width=device-width, initial-scale=1"),
)
"""Base head tags; FastHTML's defaults are off so nothing is loaded from a CDN."""

app = FastHTML(
    hdrs=(
        *META_TAGS,
        HTMX_SCRIPT,
        SSE_EXT_SCRIPT,
        Style(CUSTOM_CSS),
        Script(APP_JS),
    ),
    title="PsycheGraph",
    live=False,
    htmx=False,
    default_hdrs=False,
)
"""The FastHTML app that `langgraph.json` serves at the HTTP app path."""

register_routes(app)
