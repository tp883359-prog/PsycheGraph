"""Fetch (or refresh) the vendored front-end assets.

The demo UI must not break because a CDN published a new version, and it should
work without internet access, so htmx and the SSE extension live in
`src/react_agent/static/` instead of being loaded from a CDN.

They are pinned downloads, byte-identical to the published files except for a
short header naming the source and the licence. Run this script to refresh
them; bump the versions here and in `docs/WEB_ARCHITECTURE.md` together.

    uv run python scripts/fetch_static_assets.py            # verify only
    uv run python scripts/fetch_static_assets.py --write    # download
"""

# Progress output is the intended deliverable of this CLI.
# ruff: noqa: T201

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

ASSETS: dict[str, tuple[str, str]] = {
    "htmx.min.js": (
        "2.0.7",
        "https://cdn.jsdelivr.net/npm/htmx.org@2.0.7/dist/htmx.min.js",
    ),
    "htmx-ext-sse.js": (
        "2.2.1",
        "https://unpkg.com/htmx-ext-sse@2.2.1/sse.js",
    ),
}
"""Pinned files: name -> (version, download URL)."""

STATIC_DIR = Path("src/react_agent/static")

HEADER = (
    "/*!\n"
    " * {name} v{version} - vendored so the demo UI does not depend on a CDN at"
    " runtime.\n"
    " * Source: {url}\n"
    " * License: BSD 2-Clause (see the htmx project). Vendored, unmodified except"
    " for\n"
    " * this header; refresh it with scripts/fetch_static_assets.py.\n"
    " */\n"
)


def main() -> int:
    """Check or refresh the vendored assets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="download the files")
    args = parser.parse_args()

    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    for name, (version, url) in ASSETS.items():
        target = STATIC_DIR / name
        if args.write:
            with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
                payload = response.read().decode("utf-8")
            target.write_text(
                HEADER.format(name=name.removesuffix(".js"), version=version, url=url)
                + payload,
                encoding="utf-8",
            )
            print(f"wrote {target} ({target.stat().st_size} bytes, v{version})")
            continue
        if not target.is_file():
            print(f"missing {target}; run with --write")
            return 1
        head = target.read_text(encoding="utf-8")[:400]
        if version not in head:
            print(f"{target}: version {version} not in the header; run with --write")
            return 1
        print(f"{target}: v{version} present ({target.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
