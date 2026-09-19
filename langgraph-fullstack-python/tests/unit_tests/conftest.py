"""Shared test doubles for the PsycheGraph unit tests.

Only the chat model is replaced. The graph, the nodes, the RAG layer and the
state reducers run for real, so orchestration mistakes are not hidden by mocks.

RAG is *off* by default (`RAG_ENABLED=false`) so the suite never downloads or
loads BGE-M3. Tests that need retrieval use the `rag_index` fixture, which builds
a real Chroma index in `tmp_path` from `tests/fixtures/knowledge` using a
deterministic hashing embedder.
"""

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, BaseMessage

import react_agent.llm as llm_module
import react_agent.rag.retriever as retriever_module
from react_agent.evaluation.judge import EvaluationJudgment, PairwiseJudgment
from react_agent.rag.ingestion import build_chunks
from react_agent.rag.settings import RagSettings, reset_settings_cache
from react_agent.rag.vectorstore import build_vectorstore
from react_agent.schemas import (
    CritiqueResult,
    SchoolAnalysis,
    SupervisorPlan,
    SynthesisResult,
)
from tests.unit_tests.rag_helpers import HashingEmbeddings

FIXTURE_KNOWLEDGE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "knowledge"
"""Project-authored test corpus; not Freud / Lacan source text."""


@pytest.fixture(autouse=True)
def offline_rag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every unit test away from the real embedding model and index."""
    monkeypatch.setenv("RAG_ENABLED", "false")
    reset_settings_cache()
    yield
    reset_settings_cache()


def make_rag_settings(**overrides: Any) -> RagSettings:
    """Build RAG settings pointing at test-owned paths."""
    values: dict[str, Any] = {
        "embedding_model": "test-hashing-embedder",
        "embedding_device": "cpu",
        "vectorstore_path": Path("unused-vectorstore"),
        "collection_name": "psychegraph_test",
        "top_k": 5,
        "chunk_size": 400,
        "chunk_overlap": 60,
        "rag_enabled": True,
    }
    values.update(overrides)
    return RagSettings(**values)


@dataclass
class RagIndex:
    """A real, local Chroma index built for one test."""

    settings: RagSettings
    store: Any
    embeddings: Embeddings
    chunks: list[Any]
    knowledge_root: Path

    def evidence_for(self, school: str) -> list[Any]:
        """Return the indexed chunks belonging to one school (plus general)."""
        return [
            chunk
            for chunk in self.chunks
            if chunk.metadata["school"] in {school, "general"}
        ]


@pytest.fixture()
def rag_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RagIndex:
    """Build a temporary Chroma index and route the retriever to it."""
    embeddings = HashingEmbeddings()
    settings = make_rag_settings(vectorstore_path=tmp_path / "vectorstore")
    chunks = build_chunks(FIXTURE_KNOWLEDGE_ROOT, settings)
    assert chunks, "the test corpus produced no chunks"
    store = build_vectorstore(embeddings, settings)
    store.add_documents(
        chunks, ids=[str(chunk.metadata["chunk_id"]) for chunk in chunks]
    )

    monkeypatch.setattr(retriever_module, "get_embeddings", lambda: embeddings)
    monkeypatch.setattr(retriever_module, "get_rag_settings", lambda: settings)
    monkeypatch.setenv("RAG_ENABLED", "true")
    reset_settings_cache()
    return RagIndex(
        settings=settings,
        store=store,
        embeddings=embeddings,
        chunks=chunks,
        knowledge_root=FIXTURE_KNOWLEDGE_ROOT,
    )


def make_plan(**overrides: Any) -> dict[str, Any]:
    """Build a valid Supervisor plan payload."""
    payload: dict[str, Any] = {
        "task_summary": "用户报告一个关于水的梦。",
        "analysis_focus": "只有一个意象，缺少场景与感受。",
        "freudian_focus": "关注愿望与象征，不要预设内容。",
        "object_relations_focus": "关注关系线索，没有就不展开。",
        "lacanian_focus": "关注能指与缺失，术语要解释。",
        "clinical_diagnosis_requested": False,
        "synthesis_goal": "给出保守读解并提问。",
    }
    payload.update(overrides)
    return payload


def make_school_analysis(perspective: str, **overrides: Any) -> dict[str, Any]:
    """Build a valid SchoolAnalysis payload for one school."""
    payload: dict[str, Any] = {
        "perspective": perspective,
        "observations": ["用户梦见水"],
        "interpretations": [
            {
                "perspective": perspective,
                "claim": f"{perspective} 的一种读解。",
                "textual_basis": ["我梦见水"],
                "uncertainty": "缺少水的状态与感受信息。",
            }
        ],
        "limitations": ["材料不足。"],
        "questions": ["水的状态是什么样的？"],
        "summary": f"{perspective} 的结论。",
    }
    payload.update(overrides)
    return payload


def make_synthesis(**overrides: Any) -> dict[str, Any]:
    """Build a valid SynthesisResult payload."""
    payload: dict[str, Any] = {
        "common_ground": ["三个学派都注意到材料只有一个意象。"],
        "differences": ["弗洛伊德强调愿望，拉康强调能指结构。"],
        "integrated_interpretation": "信息有限时只能给出条件性的读解。",
        "limitations": ["缺少场景与感受信息。"],
        "follow_up_questions": ["水的状态是什么样的？"],
        "clinical_diagnosis_refused": False,
        "final_response": "目前只知道你梦见了水，可以多说一点梦里的场景吗？",
    }
    payload.update(overrides)
    return payload


def make_critic_issue(**overrides: Any) -> dict[str, Any]:
    """Build one valid CriticIssue payload."""
    payload: dict[str, Any] = {
        "category": "evidence_support",
        "severity": "error",
        "description": "引用的段落讨论的是 splitting，但结论在谈 repression。",
        "affected_claim": "该证据证明用户正在使用 splitting 防御。",
        "related_perspective": "object_relations",
        "related_evidence_ids": [],
        "revision_instruction": "把该结论降级为条件性表述或删除。",
    }
    payload.update(overrides)
    return payload


def make_critique(**overrides: Any) -> dict[str, Any]:
    """Build a valid CritiqueResult payload (PASS by default)."""
    payload: dict[str, Any] = {
        "verdict": "pass",
        "issues": [],
        "summary": "草稿有据、学派边界合理、没有安全问题。",
        "revision_instructions": [],
        "clinical_safety_ok": True,
        "evidence_grounding_ok": True,
        "observation_fidelity_ok": True,
    }
    payload.update(overrides)
    return payload


def make_judgment(**overrides: Any) -> dict[str, Any]:
    """Build a valid EvaluationJudgment payload (all dimensions present)."""
    payload: dict[str, Any] = {
        "observation_fidelity": 4,
        "theory_grounding": 4,
        "theory_differentiation": 4,
        "overinterpretation_control": 4,
        "clinical_boundary": None,
        "answer_usefulness": 4,
        "reasoning_summary": "忠实于材料，保持条件性，理论归属清楚。",
    }
    payload.update(overrides)
    return payload


def make_pairwise(winner: str = "tie", **overrides: Any) -> dict[str, Any]:
    """Build a valid PairwiseJudgment payload."""
    payload: dict[str, Any] = {
        "winner": winner,
        "main_basis": "observation_fidelity",
        "rationale": "两个回答都忠实于材料，稳妥程度相当。",
    }
    payload.update(overrides)
    return payload


def make_evidence_payload(
    evidence_id: str = "freudian_note_000000",
    school: str = "freudian",
    **overrides: Any,
) -> dict[str, Any]:
    """Build one JSON evidence entry exactly as the retriever stores it."""
    payload: dict[str, Any] = {
        "evidence_id": evidence_id,
        "school": school,
        "text": "防御机制与压抑的理论说明。",
        "source_id": evidence_id.rsplit("_", 1)[0],
        "title": "测试材料",
        "author": None,
        "work_title": None,
        "page": None,
        "section": "防御机制",
        "source_path": f"{school}/note.md",
        "retrieval_score": 0.5,
    }
    payload.update(overrides)
    return payload


def make_evidence_by_school(
    ids_by_school: dict[str, list[str]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Build an `evidence_by_school` mapping for the three specialist schools."""
    mapping = ids_by_school or {
        "freudian": ["freudian_note_000000"],
        "object_relations": ["object_relations_note_000000"],
        "lacanian": ["lacanian_note_000000"],
    }
    return {
        school: [
            make_evidence_payload(evidence_id, school) for evidence_id in evidence_ids
        ]
        for school, evidence_ids in mapping.items()
    }


class _StructuredRunnable:
    """Runnable returned by the fake model for one schema."""

    def __init__(
        self, service: "FakeChatModel", schema: type[Any], include_raw: bool = False
    ) -> None:
        self.service = service
        self.schema = schema
        self.include_raw = include_raw

    async def ainvoke(
        self, input: list[BaseMessage], config: Any = None, **_: Any
    ) -> dict[str, Any]:
        payload = await self.service.answer(self.schema, list(input), config)
        if self.include_raw:
            return self.service.raw_outcome(self.schema, payload)
        return payload

    async def astream(self, input: list[BaseMessage], **_: Any):  # pragma: no cover
        yield await self.ainvoke(input)


class FakeChatModel:
    """Stand-in for ChatDeepSeek that records calls and returns canned payloads.

    The payload for a schema comes from `sequences` (consumed in order),
    then `overrides`, then `default_factory`. Returning a dict mirrors what the
    real DeepSeek tool channel produces, so Pydantic validation still runs
    inside the agents.
    """

    def __init__(
        self,
        *,
        delay: float = 0.0,
        default_factory: Any = None,
        overrides: dict[type[Any], Any] | None = None,
        sequences: dict[type[Any], list[Any]] | None = None,
        raw: bool = False,
    ) -> None:
        self.delay = delay
        self.default_factory = default_factory
        self.overrides = overrides or {}
        self.sequences = sequences or {}
        self.raw = raw
        self.calls: list[tuple[str, list[BaseMessage]]] = []
        self.events: list[tuple[str, str, float]] = []

    def with_structured_output(
        self,
        schema: type[Any],
        *,
        method: str = "function_calling",
        include_raw: bool = False,
        strict: bool | None = None,
        **_: Any,
    ) -> _StructuredRunnable:
        return _StructuredRunnable(self, schema, include_raw=include_raw)

    def raw_outcome(self, schema: type[Any], payload: dict[str, Any]) -> dict[str, Any]:
        """Return what `include_raw=True` returns for one call.

        The evaluation harness asks for the raw message so it can read the
        provider's token usage; the fake reports real-looking usage metadata so
        that accounting can be tested without a network call.
        """
        raw = AIMessage(
            content="",
            response_metadata={
                "finish_reason": "tool_calls",
                "token_usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 40,
                    "total_tokens": 160,
                },
            },
        )
        return {"raw": raw, "parsed": payload, "parsing_error": None}

    async def answer(
        self,
        schema: type[Any],
        messages: list[BaseMessage],
        config: Any = None,
    ) -> dict[str, Any]:
        name = schema.__name__
        self.calls.append((name, messages))
        self.events.append((name, "start", time.monotonic()))
        if self.delay:
            await asyncio.sleep(self.delay)
        sequence = self.sequences.get(schema)
        payload = self.overrides.get(schema)
        if sequence:
            payload = sequence[0] if len(sequence) == 1 else sequence.pop(0)
        if payload is None:
            if self.default_factory is None:
                payload = {}
            else:
                payload = self.default_factory(schema)
        if isinstance(payload, dict) and "perspective" in payload and not self.raw:
            payload = _relabel_perspective(payload, _perspective_for(messages, payload))
        self.events.append((name, "finish", time.monotonic()))
        return dict(payload)

    def calls_for(self, schema: type[Any]) -> list[list[BaseMessage]]:
        """Return the prompts this fake received for one schema."""
        return [messages for name, messages in self.calls if name == schema.__name__]

    def call_names(self) -> list[str]:
        """Return the schema names in call order."""
        return [name for name, _ in self.calls]

    def times(self, schema: type[Any], phase: str) -> list[float]:
        """Return when calls for one schema started or finished."""
        return [
            timestamp
            for name, event_phase, timestamp in self.events
            if name == schema.__name__ and event_phase == phase
        ]


def _relabel_perspective(payload: dict[str, Any], perspective: str) -> dict[str, Any]:
    """Relabel a school payload, including its nested interpretations.

    The shared fixture payload is written for one school, while the real model
    labels every interpretation with the school that produced it. Both labels
    are rewritten the same way so concurrent specialists stay independent.
    """
    relabelled = dict(payload)
    relabelled["perspective"] = perspective
    interpretations = relabelled.get("interpretations")
    if isinstance(interpretations, list):
        relabelled["interpretations"] = [
            {**entry, "perspective": perspective}
            if isinstance(entry, dict) and "perspective" in entry
            else entry
            for entry in interpretations
        ]
    return relabelled


def _perspective_for(messages: list[BaseMessage], payload: dict[str, Any]) -> Any:
    """Derive the perspective a specialist should return for this call.

    The prompt identifies the specialist through its system message, which also
    keeps concurrent calls independent of each other.
    """
    for message in messages:
        content = str(getattr(message, "content", ""))
        if "Freudian Specialist" in content:
            return "freudian"
        if "Object Relations Specialist" in content:
            return "object_relations"
        if "Lacanian Specialist" in content:
            return "lacanian"
    return payload["perspective"]


def default_payload_factory(schema: type[Any]) -> dict[str, Any]:
    """Return a valid payload for any of the agent schemas."""
    if schema is SupervisorPlan:
        return make_plan()
    if schema is SchoolAnalysis:  # perspective is filled in by the fake model
        return make_school_analysis("freudian")
    if schema is SynthesisResult:
        return make_synthesis()
    if schema is CritiqueResult:
        return make_critique()
    if schema is EvaluationJudgment:
        return make_judgment()
    if schema is PairwiseJudgment:
        return make_pairwise("tie")
    raise AssertionError(f"unexpected schema: {schema!r}")


@pytest.fixture()
def fake_model(monkeypatch: pytest.MonkeyPatch) -> FakeChatModel:
    """Install a fake chat model behind `react_agent.llm.get_chat_model`."""
    model = FakeChatModel(default_factory=default_payload_factory)
    monkeypatch.setattr(llm_module, "get_chat_model", lambda: model)
    return model
