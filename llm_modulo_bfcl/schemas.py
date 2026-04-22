"""Typed plan / result / feedback representations used across the framework."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class FunctionCall:
    function: str
    arguments: dict
    output_var: Optional[str] = None


@dataclass
class Plan:
    steps: list[FunctionCall] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.steps)


@dataclass
class CriticResult:
    critic_name: str
    status: str  # "pass" or "fail"
    step_index: Optional[int] = None
    error: str = ""
    suggestion: Optional[str] = None
    metadata: Optional[dict] = None


@dataclass
class AggregatedFeedback:
    failures: list[CriticResult] = field(default_factory=list)
    instructions: list[str] = field(default_factory=list)
    summary: str = ""
