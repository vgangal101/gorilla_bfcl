"""Base class for multi-turn critics."""

from abc import ABC, abstractmethod
from typing import Optional

from ..schemas import MultiTurnCriticResult, Proposal


class MultiTurnCritic(ABC):
    name: str = "MultiTurnCritic"

    @abstractmethod
    def evaluate(self, proposal: Proposal, context: dict) -> MultiTurnCriticResult:
        """Inspect `proposal` under `context` and return a structured result.

        `context` carries:
            history       : History
            state         : State
            tool_specs    : list[dict] (JSON-Schema-shaped function definitions)
            tool_registry : dict[str, callable] (function-name -> implementation)
        """

    def _pass(self, metadata: Optional[dict] = None) -> MultiTurnCriticResult:
        return MultiTurnCriticResult(
            name=self.name, passed=True, severity="hard", metadata=metadata
        )

    def _fail(self, feedback: str, metadata: Optional[dict] = None) -> MultiTurnCriticResult:
        return MultiTurnCriticResult(
            name=self.name,
            passed=False,
            feedback=feedback,
            severity="hard",
            metadata=metadata,
        )
