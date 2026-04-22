"""Tool executor. Runs a plan, resolving `$var` references between steps.

For real BFCL use, inject `handlers` that dispatch to concrete Python
implementations or a sandboxed runtime. For tests/demos a dict of callables
is enough.
"""

import re
from typing import Callable

from schemas import Plan


_VAR_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")


class ToolExecutor:
    def __init__(self, handlers: dict[str, Callable[[dict], dict]] | None = None):
        self.handlers = handlers or {}

    def execute_step(self, function_name: str, arguments: dict) -> dict:
        handler = self.handlers.get(function_name)
        if handler is None:
            return {"status": "error", "error": f"no handler for '{function_name}'"}
        try:
            output = handler(arguments)
            return {"status": "ok", "output": output}
        except Exception as e:  # noqa: BLE001 - surface any tool error to critics
            return {"status": "error", "error": f"{type(e).__name__}: {e}"}

    def execute_plan(self, plan: Plan) -> list[dict]:
        bindings: dict = {}
        trace: list[dict] = []

        for step in plan.steps:
            resolved = self._resolve(step.arguments, bindings)
            if isinstance(resolved, dict) and resolved.get("__unresolved__"):
                trace.append({
                    "status": "error",
                    "error": resolved["error"],
                    "function": step.function,
                    "arguments": step.arguments,
                })
                continue

            result = self.execute_step(step.function, resolved)
            result["function"] = step.function
            result["arguments"] = resolved
            trace.append(result)

            if result["status"] == "ok" and step.output_var:
                bindings[step.output_var] = result["output"]

        return trace

    # ------------------------------------------------------------------
    # Variable resolution
    # ------------------------------------------------------------------

    def _resolve(self, v, bindings: dict):
        if isinstance(v, str):
            m = _VAR_RE.fullmatch(v)
            if m:
                name = m.group(1)
                if name not in bindings:
                    return {"__unresolved__": True, "error": f"unresolved var '${name}'"}
                return bindings[name]
            # inline interpolation — every referenced var must exist
            for match in _VAR_RE.finditer(v):
                if match.group(1) not in bindings:
                    return {
                        "__unresolved__": True,
                        "error": f"unresolved var '${match.group(1)}'",
                    }
            return _VAR_RE.sub(lambda m: str(bindings[m.group(1)]), v)

        if isinstance(v, list):
            out = []
            for item in v:
                r = self._resolve(item, bindings)
                if isinstance(r, dict) and r.get("__unresolved__"):
                    return r
                out.append(r)
            return out

        if isinstance(v, dict):
            out = {}
            for k, item in v.items():
                r = self._resolve(item, bindings)
                if isinstance(r, dict) and r.get("__unresolved__"):
                    return r
                out[k] = r
            return out

        return v
