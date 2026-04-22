"""RedundancyCritic: soft critic that flags duplicate (function, arguments) steps.

Not required for soundness. Surface it in feedback so the LLM can tighten plans.
"""

import json

from critics.base import Critic
from schemas import CriticResult, Plan


class RedundancyCritic(Critic):
    name = "RedundancyCritic"

    def evaluate(self, plan: Plan, context: dict) -> CriticResult:
        seen: dict[tuple, int] = {}

        for i, step in enumerate(plan.steps):
            # Hashable key: function + canonical-JSON-serialised arguments.
            key = (step.function, json.dumps(step.arguments, sort_keys=True, default=str))
            if key in seen:
                return self._fail(
                    error=(
                        f"step {i} duplicates step {seen[key]} "
                        f"(same function '{step.function}' and arguments)."
                    ),
                    step_index=i,
                    suggestion="Remove the redundant call or change its arguments.",
                )
            seen[key] = i

        return self._pass()
