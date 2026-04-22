"""Proposal parser: raw LLM output -> structured Plan."""

import json
import re
from typing import Optional

from schemas import FunctionCall, Plan


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
# Qwen3 (and other reasoning models) prepend a <think>...</think> block with
# free-form reasoning before the answer. Strip it so the downstream JSON
# parse sees only the plan.
_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)


class ProposalParser:
    def parse(self, raw: str) -> tuple[Optional[Plan], Optional[str]]:
        """Return `(plan, error)`.

        On success, `error` is None. On failure, `plan` is None and `error` is
        a human-readable message describing what went wrong.
        """
        s = (raw or "").strip()
        s = _THINK_RE.sub("", s).strip()
        m = _FENCE_RE.search(s)
        if m:
            s = m.group(1).strip()

        try:
            data = json.loads(s)
        except json.JSONDecodeError as e:
            return None, f"JSON parse error: {e.msg} (line {e.lineno}, col {e.colno})"

        if not isinstance(data, dict):
            return None, "Top-level proposal must be a JSON object with a 'steps' field."

        steps = data.get("steps")
        if not isinstance(steps, list):
            return None, "'steps' must be a list of function-call objects."

        plan = Plan()
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                return None, f"Step {i} is not an object."
            fn = step.get("function")
            if not isinstance(fn, str) or not fn:
                return None, f"Step {i} missing required string field 'function'."
            args = step.get("arguments", {})
            if not isinstance(args, dict):
                return None, f"Step {i} 'arguments' must be an object."
            out = step.get("output_var")
            if out is not None and (not isinstance(out, str) or not out):
                return None, f"Step {i} 'output_var' must be a non-empty string or omitted."
            plan.steps.append(FunctionCall(function=fn, arguments=args, output_var=out))

        return plan, None
