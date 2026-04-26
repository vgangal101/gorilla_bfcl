"""Multi-turn data classes: Proposal, State, History, MultiTurnCriticResult.

Kept separate from the single-turn `Plan`-based schemas so the existing
single-turn loop is untouched.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Proposal:
    """One assistant turn. Exactly one of three shapes:

      function_call : function_name + arguments
      clarification : message (question for the user)
      final_answer  : message (task complete)
    """
    type: str
    function_name: Optional[str] = None
    arguments: Optional[dict] = None
    message: Optional[str] = None


@dataclass
class State:
    """Mutable state carried across turns.

    `tool_outputs` maps the most recent output by function name. `variables`
    is a free-form bag for downstream extractors (entity keys, intermediate
    values), unused by the core loop but available to critics if wanted.
    """
    tool_outputs: dict = field(default_factory=dict)
    variables: dict = field(default_factory=dict)


@dataclass
class HistoryEntry:
    user_message: Optional[str] = None
    proposal: Optional[Proposal] = None
    tool_output: Any = None


@dataclass
class History:
    entries: list[HistoryEntry] = field(default_factory=list)

    def add_user(self, msg: str) -> None:
        self.entries.append(HistoryEntry(user_message=msg))

    def add_assistant(self, proposal: Proposal, tool_output: Any = None) -> None:
        self.entries.append(HistoryEntry(proposal=proposal, tool_output=tool_output))

    def all_user_messages(self) -> list[str]:
        return [e.user_message for e in self.entries if e.user_message is not None]

    def all_tool_outputs(self) -> list[Any]:
        return [e.tool_output for e in self.entries if e.tool_output is not None]

    def render_chat(self) -> str:
        """Plain-text transcript of the conversation so far. Used in prompts."""
        lines: list[str] = []
        for e in self.entries:
            if e.user_message is not None:
                lines.append(f"USER: {e.user_message}")
            if e.proposal is not None:
                p = e.proposal
                if p.type == "function_call":
                    lines.append(
                        f"ASSISTANT (function_call): "
                        f"{p.function_name}({p.arguments})"
                    )
                elif p.type == "clarification":
                    lines.append(f"ASSISTANT (clarification): {p.message}")
                elif p.type == "final_answer":
                    lines.append(f"ASSISTANT (final_answer): {p.message}")
            if e.tool_output is not None:
                lines.append(f"TOOL_OUTPUT: {e.tool_output}")
        return "\n".join(lines)


@dataclass
class MultiTurnCriticResult:
    name: str
    passed: bool
    feedback: str = ""
    severity: str = "hard"
    metadata: Optional[dict] = None
