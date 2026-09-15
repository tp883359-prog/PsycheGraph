"""A simple chatbot."""

from langgraph.prebuilt import create_react_agent

from react_agent.models import get_chat_model
from react_agent.prompts import PSYCHOANALYTIC_ANALYST_SYSTEM_PROMPT

model = get_chat_model()

graph = create_react_agent(
    model,
    tools=[],
    prompt=PSYCHOANALYTIC_ANALYST_SYSTEM_PROMPT,
)
