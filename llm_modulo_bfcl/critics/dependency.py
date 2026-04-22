"""DependencyCritic: multi-step references (`$var`) must be produced by earlier steps."""

import re

from critics.base import Critic
from schemas import CriticResult, Plan


_VAR_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")


class DependencyCritic(Critic):
    name = "DependencyCritic"

    def evaluate(self, plan: Plan, context: dict) -> CriticResult:
        defined: set[str] = set()

        for i, step in enumerate(plan.steps):
            for arg, val in step.arguments.items():
                for ref in self._refs(val):
                    if ref not in defined:
                        return self._fail(
                            error=(
                                f"argument '{arg}' in step {i} references '${ref}' "
                                f"which is not produced by any earlier step."
                            ),
                            step_index=i,
                            suggestion=(
                                f"Add a prior step with output_var '{ref}' "
                                f"or replace the reference with a concrete value."
                            ),
                        )
            if step.output_var:
                if step.output_var in defined:
                    return self._fail(
                        error=(
                            f"output_var '{step.output_var}' is redefined at step {i}."
                        ),
                        step_index=i,
                        suggestion="Use unique output_var names across the plan.",
                    )
                defined.add(step.output_var)

        return self._pass(metadata={"defined_vars": sorted(defined)})

    # Recursively collect $name references from any JSON-like value.
    def _refs(self, v) -> list[str]:
        if isinstance(v, str):
            return _VAR_RE.findall(v)
        if isinstance(v, list):
            out: list[str] = []
            for item in v:
                out.extend(self._refs(item))
            return out
        if isinstance(v, dict):
            out = []
            for item in v.values():
                out.extend(self._refs(item))
            return out
        return []
