"""ArgumentValueCritic: semantic validation beyond what JSON-Schema types cover.

Checks enums, format hints (email/date), numeric bounds, and basic domain rules
that are commonly encoded in BFCL schemas.
"""

import re

from critics.base import Critic
from schemas import CriticResult, Plan


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _is_var_ref(v) -> bool:
    return isinstance(v, str) and v.startswith("$") and len(v) > 1 and " " not in v


class ArgumentValueCritic(Critic):
    name = "ArgumentValueCritic"

    def evaluate(self, plan: Plan, context: dict) -> CriticResult:
        schemas = {
            s["name"]: s
            for s in context.get("function_schemas", [])
            if isinstance(s, dict) and s.get("name")
        }

        for i, step in enumerate(plan.steps):
            schema = schemas.get(step.function)
            if schema is None:
                continue
            properties = (schema.get("parameters") or {}).get("properties", {}) or {}

            for arg, val in step.arguments.items():
                if _is_var_ref(val):
                    continue
                spec = properties.get(arg, {}) or {}

                if "enum" in spec and val not in spec["enum"]:
                    return self._fail(
                        error=(
                            f"argument '{arg}' for '{step.function}' has invalid value "
                            f"{val!r}; not in enum {spec['enum']}."
                        ),
                        step_index=i,
                        suggestion=f"Use one of {spec['enum']}.",
                    )

                fmt = spec.get("format")
                if fmt == "email" and isinstance(val, str) and not _EMAIL_RE.match(val):
                    return self._fail(
                        error=f"argument '{arg}' is not a valid email: {val!r}.",
                        step_index=i,
                        suggestion="Provide a syntactically valid email address.",
                    )
                if fmt == "date" and isinstance(val, str) and not _ISO_DATE_RE.match(val):
                    return self._fail(
                        error=f"argument '{arg}' is not an ISO date (YYYY-MM-DD): {val!r}.",
                        step_index=i,
                        suggestion="Use ISO date format, e.g. 2025-07-14.",
                    )

                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    if "minimum" in spec and val < spec["minimum"]:
                        return self._fail(
                            error=(
                                f"argument '{arg}' = {val} is below minimum "
                                f"{spec['minimum']}."
                            ),
                            step_index=i,
                            suggestion=f"Use a value >= {spec['minimum']}.",
                        )
                    if "maximum" in spec and val > spec["maximum"]:
                        return self._fail(
                            error=(
                                f"argument '{arg}' = {val} is above maximum "
                                f"{spec['maximum']}."
                            ),
                            step_index=i,
                            suggestion=f"Use a value <= {spec['maximum']}.",
                        )

                if isinstance(val, str) and "pattern" in spec:
                    if not re.search(spec["pattern"], val):
                        return self._fail(
                            error=(
                                f"argument '{arg}' = {val!r} does not match pattern "
                                f"{spec['pattern']!r}."
                            ),
                            step_index=i,
                            suggestion=f"Provide a value matching /{spec['pattern']}/.",
                        )

        return self._pass()
