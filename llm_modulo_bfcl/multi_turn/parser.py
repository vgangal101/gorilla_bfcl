"""Raw LLM output -> Proposal."""

import json
import re
from typing import Optional

from .schemas import Proposal


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_VALID_TYPES = {"function_call", "clarification", "final_answer"}


class ProposalParser:
    def parse(self, raw: str) -> tuple[Optional[Proposal], Optional[str]]:
        """Return `(proposal, error)`. On success error is None."""
        s = (raw or "").strip()
        m = _FENCE_RE.search(s)
        if m:
            s = m.group(1).strip()

        try:
            data = json.loads(s)
        except json.JSONDecodeError as e:
            return None, f"JSON parse error: {e.msg} (line {e.lineno}, col {e.colno})"

        if not isinstance(data, dict):
            return None, "proposal must be a JSON object"

        ptype = data.get("type")
        if ptype not in _VALID_TYPES:
            return (
                None,
                f"unknown proposal type {ptype!r}; must be one of "
                f"{sorted(_VALID_TYPES)}",
            )

        if ptype == "function_call":
            fn = data.get("function_name")
            if not isinstance(fn, str) or not fn:
                return None, "function_call requires a non-empty 'function_name' string"
            args = data.get("arguments", {})
            if not isinstance(args, dict):
                return None, "function_call 'arguments' must be a JSON object"
            return Proposal(type=ptype, function_name=fn, arguments=args), None

        # clarification or final_answer
        msg = data.get("message")
        if not isinstance(msg, str) or not msg.strip():
            return None, f"{ptype} requires a non-empty 'message' string"
        return Proposal(type=ptype, message=msg), None
