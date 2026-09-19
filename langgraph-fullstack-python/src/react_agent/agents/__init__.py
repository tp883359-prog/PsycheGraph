"""Agent nodes for PsycheGraph.

Each module in this package owns exactly one agent: its system prompt, its
output schema and its graph node. The graph wiring in `react_agent.graph` only
composes them.
"""

from react_agent.agents.critic import CRITIC_SYSTEM_PROMPT, critic_node
from react_agent.agents.evidence import evidence_node
from react_agent.agents.finalizer import finalize_node, safe_finalize_node
from react_agent.agents.freudian import FREUDIAN_SYSTEM_PROMPT, freudian_node
from react_agent.agents.lacanian import LACANIAN_SYSTEM_PROMPT, lacanian_node
from react_agent.agents.object_relations import (
    OBJECT_RELATIONS_SYSTEM_PROMPT,
    object_relations_node,
)
from react_agent.agents.revision import REVISION_SYSTEM_PROMPT, revise_synthesis_node
from react_agent.agents.supervisor import SUPERVISOR_SYSTEM_PROMPT, supervisor_node
from react_agent.agents.synthesizer import SYNTHESIZER_SYSTEM_PROMPT, synthesizer_node
from react_agent.agents.validator import deterministic_validator_node

__all__ = [
    "CRITIC_SYSTEM_PROMPT",
    "FREUDIAN_SYSTEM_PROMPT",
    "LACANIAN_SYSTEM_PROMPT",
    "OBJECT_RELATIONS_SYSTEM_PROMPT",
    "REVISION_SYSTEM_PROMPT",
    "SUPERVISOR_SYSTEM_PROMPT",
    "SYNTHESIZER_SYSTEM_PROMPT",
    "critic_node",
    "deterministic_validator_node",
    "evidence_node",
    "finalize_node",
    "freudian_node",
    "lacanian_node",
    "object_relations_node",
    "revise_synthesis_node",
    "safe_finalize_node",
    "supervisor_node",
    "synthesizer_node",
]
