"""Measure process memory and cold-start timing for the deployment docs.

Answers the two sizing questions a deployment has to answer:

* How much RSS does the server hold, before and after BGE-M3 is loaded?
* How long does the first request take (lazy model load) versus the next ones?

RSS is read with ctypes on Windows and `/proc/self/status` on Linux, so the
measurement needs no extra dependency and works in the container as well as on
this host. The numbers feed `docs/DEPLOYMENT_BENCHMARK.md`; they are properties
of one CPU-only machine, not universal constants.

Usage:
    uv run python scripts/bench_rss.py --skip-run
    uv run python scripts/bench_rss.py              # needs DEEPSEEK_API_KEY
"""

# Progress output is the intended deliverable of this CLI.
# ruff: noqa: T201

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = REPO_ROOT / "data" / "reports" / "phase10a_resources.json"


class _ProcessMemoryCounters(ctypes.Structure):  # pragma: no cover - Windows only
    """`PROCESS_MEMORY_COUNTERS` from the Win32 API."""

    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def rss_mb() -> float:  # pragma: no cover - thin OS wrapper
    """Return the resident set size of this process in megabytes.

    On Windows the counter comes from `K32GetProcessMemoryInfo`; on Linux from
    `/proc/self/status`. Both are part of the OS, so no extra dependency is
    needed inside a slim container.

    Returns:
        RSS in MB, or 0.0 when the platform probe is unavailable.
    """
    if sys.platform == "win32":
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            get_info = kernel32.K32GetProcessMemoryInfo
            get_info.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(_ProcessMemoryCounters),
                ctypes.c_uint32,
            ]
            get_info.restype = ctypes.c_int
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            handle = kernel32.GetCurrentProcess()
            if not get_info(handle, ctypes.byref(counters), counters.cb):
                print(f"RSS probe failed: {ctypes.get_last_error()}")
                return 0.0
            return counters.WorkingSetSize / (1024 * 1024)
        except Exception as exc:  # noqa: BLE001 - a missing probe is not fatal
            print(f"RSS probe unavailable: {type(exc).__name__}")
            return 0.0
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024
    return 0.0


def stage(name: str, start: float, *, previous: float) -> dict[str, Any]:
    """Build one measurement record.

    Args:
        name: Stage label.
        start: `time.perf_counter()` value when the stage began.
        previous: RSS measured before the stage (for the delta).

    Returns:
        The record.
    """
    memory = rss_mb()
    record = {
        "stage": name,
        "elapsed_s": round(time.perf_counter() - start, 2),
        "rss_mb": round(memory, 1),
        "delta_mb": round(memory - previous, 1),
    }
    print(
        f"{name:<26} {record['elapsed_s']:>7.2f}s  "
        f"{record['rss_mb']:>8.1f} MB  (+{record['delta_mb']:.1f})"
    )
    return record


def main() -> int:
    """Run the measurement ladder.

    Returns:
        Process exit code (1 when a run was requested but impossible).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="stop after retrieval (no DeepSeek call, no API key needed)",
    )
    parser.add_argument(
        "--question",
        default="我梦见自己站在一扇打不开的门前，这说明什么？",
        help="Question used for the full-graph stage.",
    )
    args = parser.parse_args()

    os.environ.setdefault("RAG_VECTORSTORE_PATH", "data/vectorstore")
    try:  # the dev server reads .env for us; a script has to do it itself
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env", override=False)
    except ImportError:  # pragma: no cover - python-dotenv is a dependency
        pass
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    baseline = rss_mb()
    records.append(stage("interpreter + settings", started, previous=baseline))

    from react_agent.logging_config import configure_logging

    configure_logging()

    previous = rss_mb()
    started = time.perf_counter()
    from react_agent.graph import graph  # noqa: F401 - import cost is the point

    records.append(stage("graph import", started, previous=previous))

    previous = rss_mb()
    started = time.perf_counter()
    from react_agent.rag.embeddings import (
        EmbeddingUnavailableError,
        get_embeddings,
    )
    from react_agent.rag.settings import get_rag_settings
    from react_agent.rag.vectorstore import index_exists, open_vectorstore

    settings = get_rag_settings()
    if not index_exists(settings):
        print("vectorstore: missing; set RAG_VECTORSTORE_PATH to an index")
    try:
        embeddings = get_embeddings()
    except EmbeddingUnavailableError as exc:
        print(f"embedding model unavailable: {exc}")
        embeddings = None
    records.append(stage("bge-m3 lazy load", started, previous=previous))

    if embeddings is not None:
        previous = rss_mb()
        started = time.perf_counter()
        store = open_vectorstore(embeddings, settings)
        if store is not None:
            store.similarity_search_with_score(args.question, k=3)
        records.append(stage("first retrieval (cold)", started, previous=previous))

        previous = rss_mb()
        started = time.perf_counter()
        if store is not None:
            store.similarity_search_with_score(args.question, k=3)
        records.append(stage("second retrieval (warm)", started, previous=previous))

    if args.skip_run:
        peak = max(record["rss_mb"] for record in records)
        print(f"\npeak RSS: {peak:.1f} MB (measurement stopped before the graph run)")
    else:
        if not os.environ.get("DEEPSEEK_API_KEY"):
            print("DEEPSEEK_API_KEY is not set; rerun with --skip-run")
            return 1
        import asyncio

        previous = rss_mb()
        started = time.perf_counter()
        asyncio.run(
            graph.ainvoke(
                {"messages": [{"role": "user", "content": args.question}]},
                {"configurable": {"thread_id": "bench-script"}},
            )
        )
        records.append(
            stage("full graph run (6-8 LLM calls)", started, previous=previous)
        )
        peak = max(record["rss_mb"] for record in records)
        print(f"\npeak RSS: {peak:.1f} MB")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(
            {
                "platform": sys.platform,
                "python": sys.version.split()[0],
                "vectorstore": os.environ.get("RAG_VECTORSTORE_PATH"),
                "records": records,
                "peak_rss_mb": max(record["rss_mb"] for record in records),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"written: {REPORT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
