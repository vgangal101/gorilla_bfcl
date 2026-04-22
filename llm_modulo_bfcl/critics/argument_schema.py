"""ArgumentSchemaCritic: required args present, no unknown args, correct JSON-Schema types."""

from critics.base import Critic
from schemas import CriticResult, Plan


# JSON-Schema primitive -> Python type
_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def _is_var_ref(v) -> bool:
    """A bare `$name` reference is resolved at execution time, skip type check."""
    return isinstance(v, str) and v.startswith("$") and len(v) > 1 and " " not in v


class ArgumentSchemaCritic(Critic):
    name = "ArgumentSchemaCritic"

    def evaluate(self, plan: Plan, context: dict) -> CriticResult:
        schemas = {
            s["name"]: s
            for s in context.get("function_schemas", [])
            if isinstance(s, dict) and s.get("name")
        }

        for i, step in enumerate(plan.steps):
            schema = schemas.get(step.function)
            if schema is None:
                # FunctionValidityCritic owns this failure mode.
                continue

            params = schema.get("parameters") or {}
            properties = params.get("properties", {}) or {}
            required = params.get("required", []) or []

            for req in required:
                if req not in step.arguments:
                    return self._fail(
                        error=(
                            f"missing required argument '{req}' for function "
                            f"'{step.function}'."
                        ),
                        step_index=i,
                        suggestion=f"Add '{req}' to arguments.",
                    )

            for arg in step.arguments.keys():
                if arg not in properties:
                    return self._fail(
                        error=(
                            f"unknown argument '{arg}' for function '{step.function}'."
                        ),
                        step_index=i,
                        suggestion=f"Remove '{arg}' or use one of {sorted(properties)}.",
                    )

            for arg, val in step.arguments.items():
                if _is_var_ref(val):
                    continue  # resolved at execution time
                spec = properties.get(arg, {}) or {}
                t = spec.get("type")
                if t is None:
                    continue
                expected = _TYPE_MAP.get(t)
                if expected is None:
                    continue  # unknown type keyword; skip
                # bool is a subclass of int/float in Python; reject bool for any numeric type.
                if t in ("integer", "number") and isinstance(val, bool):
                    return self._fail(
                        error=(
                            f"argument '{arg}' for '{step.function}' has wrong type; "
                            f"expected {t}, got boolean."
                        ),
                        step_index=i,
                        suggestion=f"Provide {arg} as {t}.",
                    )
                if not isinstance(val, expected):
                    return self._fail(
                        error=(
                            f"argument '{arg}' for '{step.function}' has wrong type; "
                            f"expected {t}, got {type(val).__name__}."
                        ),
                        step_index=i,
                        suggestion=f"Provide {arg} as {t}.",
                    )

        return self._pass()
