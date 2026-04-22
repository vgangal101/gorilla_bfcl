"""ExecutionCritic: actually run the plan and require every step to succeed."""

from critics.base import Critic
from schemas import CriticResult, Plan


class ExecutionCritic(Critic):
    name = "ExecutionCritic"

    def __init__(self, executor):
        self.executor = executor

    def evaluate(self, plan: Plan, context: dict) -> CriticResult:
        try:
            trace = self.executor.execute_plan(plan)
        except Exception as e:  # defensive: executor should not raise
            return self._fail(error=f"executor raised unexpectedly: {e}")

        for i, entry in enumerate(trace):
            if entry.get("status") != "ok":
                return self._fail(
                    error=(
                        f"execution failed at step {i} "
                        f"({entry.get('function')}): {entry.get('error')}"
                    ),
                    step_index=i,
                    metadata={"trace": trace},
                    suggestion="Fix the argument(s) flagged above or replace the step.",
                )

        return self._pass(metadata={"trace": trace})
