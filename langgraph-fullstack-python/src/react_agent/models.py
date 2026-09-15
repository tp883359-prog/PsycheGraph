"""Initialize the chat model used by the agent."""

import os

from langchain_deepseek import ChatDeepSeek


def get_chat_model() -> ChatDeepSeek:
    """Create a DeepSeek model using environment-based configuration."""
    return ChatDeepSeek(
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"),
        timeout=60.0,
        max_retries=2,
        extra_body={"thinking": {"type": "disabled"}},
    )
