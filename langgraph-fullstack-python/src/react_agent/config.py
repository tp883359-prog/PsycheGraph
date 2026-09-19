"""Central configuration for the Phase 7 critic loop.

Phase 7 added one deterministic validator and one LLM Critic after the
Synthesizer. The loop is bounded by a single constant so the graph can never
cycle: after `MAX_REVISION` revisions the run always terminates in
`safe_finalize`.
"""

MAX_REVISION = 1
"""How many times the synthesis may be revised after a `revise` verdict.

Normal turn: `revision_count` stays 0. A first `revise` triggers one
`revise_synthesis` round; a second `revise` verdict cannot loop again and is
routed to `safe_finalize` instead.
"""
