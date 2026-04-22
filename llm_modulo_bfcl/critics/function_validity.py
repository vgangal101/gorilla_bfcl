"""FunctionValidityCritic: every called function must exist in the BFCL schema set."""

from critics.base import Critic
from schemas import CriticResult, Plan


class FunctionValidityCritic(Critic):
    name = "FunctionValidityCritic"

    def evaluate(self, plan: Plan, context: dict) -> CriticResult:
        schemas = context.get("function_schemas", [])
        known = {s["name"] for s in schemas if isinstance(s, dict) and s.get("name")}

        for i, step in enumerate(plan.steps):
            if step.function not in known:
                return self._fail(
                    error=f"function '{step.function}' is not in the available function set.",
                    step_index=i,
                    suggestion=f"Use one of: {sorted(known)}",
                )
        return self._pass()
