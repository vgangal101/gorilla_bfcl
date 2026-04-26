"""ContextGroundingCritic: argument values must come from user messages or tool outputs.

Heuristic:
  * Only enforce grounding for "entity-like" string values: a non-empty
    string with no internal whitespace (e.g. a city name, an email, an id).
  * Free-form prose values (containing spaces) are skipped — these are
    typically generated, not retrieved (e.g. an email subject or body).
  * Non-string values (numbers, booleans, lists, dicts) are skipped.

A value is grounded if its lowercased form appears as a substring of either
the concatenated user messages or the str()-rendered tool outputs from
prior turns.
"""

from .base import MultiTurnCritic
from ..schemas import MultiTurnCriticResult, Proposal


class ContextGroundingCritic(MultiTurnCritic):
    name = "ContextGroundingCritic"

    def evaluate(self, proposal: Proposal, context: dict) -> MultiTurnCriticResult:
        if proposal.type != "function_call":
            return self._pass()

        history = context["history"]
        haystack = (
            " ".join(history.all_user_messages())
            + " "
            + " ".join(str(o) for o in history.all_tool_outputs())
        ).lower()

        for arg, val in (proposal.arguments or {}).items():
            if not self._is_entity_like(val):
                continue
            if str(val).lower() not in haystack:
                return self._fail(
                    f"argument '{arg}'={val!r} is not grounded in the user's "
                    f"messages or any prior tool output. Use a value present "
                    f"in the conversation, or emit a 'clarification' proposal."
                )

        return self._pass()

    @staticmethod
    def _is_entity_like(v) -> bool:
        if not isinstance(v, str):
            return False
        s = v.strip()
        if not s:
            return False
        if " " in s:
            return False
        return True
