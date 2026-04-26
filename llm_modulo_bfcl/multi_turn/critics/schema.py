"""SchemaCritic: function exists, required args present, no extras, types match."""

from .base import MultiTurnCritic
from ..schemas import MultiTurnCriticResult, Proposal


_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


class SchemaCritic(MultiTurnCritic):
    name = "SchemaCritic"

    def evaluate(self, proposal: Proposal, context: dict) -> MultiTurnCriticResult:
        # Only function_call proposals carry schema obligations.
        if proposal.type != "function_call":
            return self._pass()

        tool_specs = {
            s["name"]: s
            for s in context.get("tool_specs", [])
            if isinstance(s, dict) and s.get("name")
        }
        fn = proposal.function_name
        if fn not in tool_specs:
            return self._fail(
                f"function '{fn}' not in registry; available: {sorted(tool_specs)}"
            )

        params = tool_specs[fn].get("parameters") or {}
        properties = params.get("properties", {}) or {}
        required = params.get("required", []) or []
        args = proposal.arguments or {}

        for r in required:
            if r not in args:
                return self._fail(
                    f"missing required argument '{r}' for function '{fn}'"
                )

        for a in args:
            if a not in properties:
                return self._fail(
                    f"unknown argument '{a}' for function '{fn}'; "
                    f"valid arguments: {sorted(properties)}"
                )

        for a, v in args.items():
            t = (properties[a] or {}).get("type")
            if t is None:
                continue
            expected = _TYPE_MAP.get(t)
            if expected is None:
                continue
            # bool is a subclass of int in Python; reject bool where integer expected.
            if t == "integer" and isinstance(v, bool):
                return self._fail(
                    f"argument '{a}' for '{fn}' has wrong type; "
                    f"expected integer, got boolean"
                )
            if not isinstance(v, expected):
                return self._fail(
                    f"argument '{a}' for '{fn}' has wrong type; "
                    f"expected {t}, got {type(v).__name__}"
                )

        return self._pass()
