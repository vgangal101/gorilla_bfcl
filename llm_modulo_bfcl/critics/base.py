"""Abstract critic interface. Every critic returns a `CriticResult`."""

from abc import ABC, abstractmethod
from typing import Optional

from schemas import CriticResult, Plan


class Critic(ABC):
    name: str = "Critic"

    @abstractmethod
    def evaluate(self, plan: Plan, context: dict) -> CriticResult:
        """Inspect `plan` under `context` and return a structured result."""

    # Convenience constructors ---------------------------------------------

    def _pass(self, metadata: Optional[dict] = None) -> CriticResult:
        return CriticResult(critic_name=self.name, status="pass", metadata=metadata)

    def _fail(
        self,
        error: str,
        step_index: Optional[int] = None,
        suggestion: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> CriticResult:
        return CriticResult(
            critic_name=self.name,
            status="fail",
            step_index=step_index,
            error=error,
            suggestion=suggestion,
            metadata=metadata,
        )
