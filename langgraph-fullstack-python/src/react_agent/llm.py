"""Shared model-call boundary for PsycheGraph agents.

Every agent goes through :func:`invoke_structured` so that the boundary between
the model and the graph state is identical everywhere:

    LLM -> Pydantic model (validation) -> model_dump(mode="json") -> State

LangGraph checkpoints only receive JSON-serialisable data. Custom Pydantic
instances are never written into the state, which keeps checkpoint
deserialisation free of custom types.
"""

import logging
from typing import Any, TypeVar, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel, ValidationError

from react_agent.models import get_chat_model

logger = logging.getLogger(__name__)

# DeepSeek answers through its tool-calling channel. `strict` is left at the
# default because DeepSeek's strict mode requires a beta endpoint and a schema
# where every property is required.
STRUCTURED_OUTPUT_METHOD = "function_calling"

SchemaT = TypeVar("SchemaT", bound=BaseModel)
"""A Pydantic model used as the validation boundary of one agent."""


def build_structured_runnable(
    schema: type[SchemaT],
) -> Runnable[list[BaseMessage], SchemaT]:
    """Return a chat model bound to `schema` through structured output.

    The model is created lazily so importing agent modules never requires
    credentials.

    Args:
        schema: Pydantic class describing the agent's output contract.

    Returns:
        A runnable that answers with a validated instance of `schema`.
    """
    model = cast("BaseChatModel", get_chat_model())
    structured = model.with_structured_output(schema, method=STRUCTURED_OUTPUT_METHOD)
    return cast("Runnable[list[BaseMessage], SchemaT]", structured)


def _describe_validation_error(error: ValidationError) -> str:
    """Summarise a schema violation without dumping model output.

    Args:
        error: Validation error raised while parsing the model answer.

    Returns:
        A short description such as `schema violation: limitations: missing`.
    """
    problems: list[str] = []
    for item in error.errors()[:4]:
        location = ".".join(str(part) for part in item.get("loc", ())) or "<root>"
        problems.append(f"{location}: {item.get('type', 'invalid')}")
    return "schema violation: " + ("; ".join(problems) or "unknown")


async def _diagnose_parsing_failure(
    schema: type[SchemaT],
    messages: list[BaseMessage],
    config: RunnableConfig | None,
) -> str:
    """Repeat one call with `include_raw` to explain a parsing failure.

    LangChain returns `None` instead of raising when a provider fails mid-turn
    (for example a tool call truncated by the output limit). This helper asks
    for the raw message once so the real reason can be surfaced.

    Args:
        schema: Schema of the failed call.
        messages: Prompt of the failed call.
        config: Runnable config of the failed call.

    Returns:
        A short human-readable reason, or a generic note if it stays unknown.
    """
    try:
        model = cast("BaseChatModel", get_chat_model())
        diagnostic = model.with_structured_output(
            schema,
            method=STRUCTURED_OUTPUT_METHOD,
            include_raw=True,
        )
        outcome = await diagnostic.ainvoke(messages, config=config)
    except Exception:  # noqa: BLE001 - diagnosis must never mask the real error
        return "the retry also failed before parsing"
    if not isinstance(outcome, dict):
        return "the retry returned an unexpected shape"
    error = outcome.get("parsing_error")
    raw_message = outcome.get("raw")
    finish_reason = None
    metadata = getattr(raw_message, "response_metadata", None)
    if isinstance(metadata, dict):
        finish_reason = metadata.get("finish_reason")
    return f"parsing_error={error!r}, finish_reason={finish_reason!r}"


async def invoke_structured(
    schema: type[SchemaT],
    messages: list[BaseMessage],
    config: RunnableConfig | None = None,
    agent_name: str = "",
) -> tuple[SchemaT, dict[str, Any]]:
    """Call the model and convert the result to a JSON-ready dict.

    Providers occasionally return a structured output that is missing a required
    field (a truncated tool call, for example). Because that is a transient
    provider-side problem rather than a logic error, the call is retried exactly
    once; if the second attempt is also invalid, a named error is raised instead
    of letting a raw Pydantic error surface.

    Args:
        schema: Pydantic class describing the expected model output.
        messages: Full prompt, including the agent's system message.
        config: Optional LangChain runnable config forwarded to the model.
        agent_name: Name used in error messages when the call fails.

    Returns:
        The validated Pydantic object and its JSON-serialisable form. Only the
        second element may be stored in the graph state.

    Raises:
        RuntimeError: If the model returned no parsable structured output, twice.
            This happens when the provider marks the turn as failed, for example
            a tool call truncated by the output limit; reporting it here keeps
            the message useful instead of surfacing a confusing schema error.
    """
    label = agent_name or schema.__name__
    runnable = build_structured_runnable(schema)
    failure = "the model returned no structured output"
    for attempt in range(2):
        raw = await runnable.ainvoke(messages, config=config)
        if raw is None:
            failure = "the model returned no structured output"
        else:
            try:
                payload = schema.model_validate(raw)
            except ValidationError as exc:
                failure = _describe_validation_error(exc)
            else:
                return payload, payload.model_dump(mode="json")
        if attempt == 0:
            logger.warning(
                "%s: invalid structured output (%s); retrying once", label, failure
            )
    reason = await _diagnose_parsing_failure(schema, messages, config)
    raise RuntimeError(
        f"{label}: the model returned no parsable {schema.__name__} "
        f"({failure}; {reason}). This usually means the structured-output call "
        "was truncated or failed at the provider."
    )
