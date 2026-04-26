"""MissingInformationCritic: missing required args -> instruct the model to clarify.

This overlaps with SchemaCritic on detection, but the *feedback* differs:
SchemaCritic asks the model to fix the call, while this critic instructs
the model to switch to a 'clarification' proposal when the missing value
is unrecoverable from context. The aggregated feedback gives both options
and the model picks one.
"""

from .base import MultiTurnCritic
from ..schemas import MultiTurnCriticResult, Proposal


_PLACEHOLDER_VALUES = {
    "todo", "?", "<missing>", "<unknown>", "unknown", "tbd", "n/a", "null", "none",
}


class MissingInformationCritic(MultiTurnCritic):
    name = "MissingInformationCritic"

    def evaluate(self, proposal: Proposal, context: dict) -> MultiTurnCriticResult:
        if proposal.type != "function_call":
            return self._pass()

        tool_specs = {
            s["name"]: s
            for s in context.get("tool_specs", [])
            if isinstance(s, dict) and s.get("name")
        }
        spec = tool_specs.get(proposal.function_name)
        if spec is None:
            # Unknown function — SchemaCritic owns that failure mode.
            return self._pass()

        required = (spec.get("parameters") or {}).get("required", []) or []
        args = proposal.arguments or {}

        missing = [r for r in required if r not in args]
        if missing:
            return self._fail(
                f"required parameter(s) {missing} for "
                f"'{proposal.function_name}' were not supplied. If they "
                f"cannot be inferred from the conversation, emit a "
                f"'clarification' proposal asking the user."
            )

        for a, v in args.items():
            if isinstance(v, str) and v.strip().lower() in _PLACEHOLDER_VALUES:
                return self._fail(
                    f"argument '{a}'={v!r} is a placeholder value. Emit a "
                    f"'clarification' proposal to ask the user for the real "
                    f"value rather than fabricating one."
                )

        return self._pass()
