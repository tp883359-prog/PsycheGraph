"""Per-run model-call accounting for the evaluation harness.

The harness must report real calls, real latency and - when the provider
actually returns them - real token counts. Nothing here estimates or invents
numbers: a value the provider did not send stays `None` (reported as `null`).

The recorder is installed once per process and routes usage into a
`ContextVar`, so concurrent evaluation tasks each account for their own calls
without touching production code.
"""

import contextvars
from dataclasses import dataclass, field
from typing import Any

import react_agent.llm as llm_module

RECORDER: contextvars.ContextVar["CallRecorder | None"] = contextvars.ContextVar(
    "psychegraph_eval_recorder", default=None
)
"""Recorder of the evaluation task currently running, if any."""

_PATCHED = False
"""Whether `react_agent.llm.get_chat_model` already routes into the recorder."""


@dataclass
class CallRecord:
    """One structured model call.

    Attributes:
        schema: Name of the Pydantic schema the call was bound to.
        ok: Whether the provider returned a message that parsed into that schema.
        parsing_error: Provider-side parsing error, when one was reported.
        finish_reason: `finish_reason` from the response metadata, when present.
        prompt_tokens: Prompt tokens, when the provider reported them.
        completion_tokens: Completion tokens, when reported.
        total_tokens: Total tokens, when reported.
        metadata_keys: Response-metadata keys, used to document what is available.
    """

    schema: str
    ok: bool
    parsing_error: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    metadata_keys: list[str] = field(default_factory=list)


@dataclass
class CallRecorder:
    """Collects the model calls of one evaluation run."""

    calls: list[CallRecord] = field(default_factory=list)

    def record(self, record: CallRecord) -> None:
        """Store one call record."""
        self.calls.append(record)

    @property
    def call_count(self) -> int:
        """Number of model calls made by this run."""
        return len(self.calls)

    @property
    def parse_failures(self) -> int:
        """Calls whose structured output did not parse (each triggers a retry)."""
        return sum(1 for call in self.calls if not call.ok)

    @property
    def token_usage_available(self) -> bool:
        """Whether every call reported token usage."""
        return bool(self.calls) and all(
            call.total_tokens is not None for call in self.calls
        )

    def _sum(self, attribute: str) -> int | None:
        values = [getattr(call, attribute) for call in self.calls]
        if not values or any(value is None for value in values):
            return None
        return int(sum(values))

    @property
    def prompt_tokens(self) -> int | None:
        """Total prompt tokens, or None when the provider did not report them."""
        return self._sum("prompt_tokens")

    @property
    def completion_tokens(self) -> int | None:
        """Total completion tokens, or None when not reported."""
        return self._sum("completion_tokens")

    @property
    def total_tokens(self) -> int | None:
        """Total tokens, or None when not reported."""
        return self._sum("total_tokens")

    def schema_names(self) -> list[str]:
        """Return the schema names in call order (audit trail)."""
        return [call.schema for call in self.calls]

    def metadata_keys(self) -> list[str]:
        """Return the union of response-metadata keys seen during the run."""
        keys: set[str] = set()
        for call in self.calls:
            keys.update(call.metadata_keys)
        return sorted(keys)


def _as_int(value: Any) -> int | None:
    """Return `value` as an int when it really is a finite number."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _extract_token_usage(raw: Any) -> dict[str, int | None]:
    """Read token usage out of a response, without guessing.

    Args:
        raw: The raw `AIMessage` the provider returned.

    Returns:
        A dict with `prompt_tokens`, `completion_tokens` and `total_tokens`;
        every value stays None when the provider did not report usage.
    """
    metadata = getattr(raw, "response_metadata", None)
    usage: dict[str, Any] = {}
    if isinstance(metadata, dict):
        candidate = metadata.get("token_usage") or metadata.get("usage")
        if isinstance(candidate, dict):
            usage = candidate
    return {
        "prompt_tokens": _as_int(usage.get("prompt_tokens")),
        "completion_tokens": _as_int(usage.get("completion_tokens")),
        "total_tokens": _as_int(usage.get("total_tokens")),
    }


def _finish_reason(raw: Any) -> str | None:
    """Return the provider's `finish_reason`, when it reported one."""
    metadata = getattr(raw, "response_metadata", None)
    if isinstance(metadata, dict) and metadata.get("finish_reason") is not None:
        return str(metadata["finish_reason"])
    return None


def _metadata_keys(raw: Any) -> list[str]:
    """Return the response-metadata keys of one response."""
    metadata = getattr(raw, "response_metadata", None)
    return sorted(str(key) for key in metadata) if isinstance(metadata, dict) else []


class _RecordingStructuredRunnable:
    """Wraps one structured-output runnable and records every call."""

    def __init__(
        self, runnable: Any, schema: str, recorder_getter: Any, include_raw: bool
    ) -> None:
        self._runnable = runnable
        self._schema = schema
        self._recorder_getter = recorder_getter
        self._include_raw = include_raw

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Call the model, record the usage and return what the caller expects."""
        outcome = await self._runnable.ainvoke(input, config=config, **kwargs)
        recorder = self._recorder_getter()
        if recorder is not None and isinstance(outcome, dict):
            raw = outcome.get("raw")
            usage = _extract_token_usage(raw)
            parsing_error = outcome.get("parsing_error")
            recorder.record(
                CallRecord(
                    schema=self._schema,
                    ok=outcome.get("parsed") is not None and parsing_error is None,
                    parsing_error=str(parsing_error) if parsing_error else None,
                    finish_reason=_finish_reason(raw),
                    metadata_keys=_metadata_keys(raw),
                    **usage,
                )
            )
        if self._include_raw:
            return outcome
        if isinstance(outcome, dict):
            return outcome.get("parsed")
        return outcome


class _RecordingChatModel:
    """Stand-in chat model that records structured calls into the context recorder."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def with_structured_output(
        self,
        schema: Any,
        *,
        method: str = "function_calling",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Bind the schema as usual, but ask for the raw message as well.

        `include_raw=True` is what makes token usage observable. The result is
        unwrapped again before it reaches the caller, so agent code is unchanged.
        """
        runnable = self._inner.with_structured_output(
            schema, method=method, include_raw=True, **kwargs
        )
        return _RecordingStructuredRunnable(
            runnable,
            getattr(schema, "__name__", str(schema)),
            lambda: RECORDER.get(),
            include_raw,
        )

    def __getattr__(self, name: str) -> Any:
        """Forward every other attribute to the real model."""
        return getattr(self._inner, name)


def install_usage_recorder() -> None:
    """Patch `react_agent.llm.get_chat_model` once per process.

    Must be called before a run. The patch is inert without an active recorder,
    so it changes no behaviour outside the evaluation harness.
    """
    global _PATCHED
    if _PATCHED:
        return
    # `getattr`/`setattr` keep the patch invisible to static typing: the model
    # factory is a re-export in `react_agent.llm`, and the harness replaces that
    # module attribute at runtime (the same trick the unit tests use).
    original = getattr(llm_module, "get_chat_model")

    def recording_factory() -> Any:
        return _RecordingChatModel(original())

    setattr(llm_module, "get_chat_model", recording_factory)
    _PATCHED = True
