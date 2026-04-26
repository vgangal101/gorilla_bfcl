"""Multi-turn meta-controller: builds initial prompt and backprompt strings.

Mirrors the single-turn controller's role but emits a Proposal-shaped
output contract (function_call | clarification | final_answer) and renders
the chat history + persistent state into the prompt.
"""

import json

from .schemas import History, State


_PROPOSAL_FORMAT = """\
Output format (strict JSON, no prose, no fences). Emit EXACTLY ONE proposal:

(1) function_call:
  {"type": "function_call", "function_name": "<name>", "arguments": {...}}

(2) clarification:
  {"type": "clarification", "message": "<question for the user>"}

(3) final_answer:
  {"type": "final_answer", "message": "<answer; the task is now complete>"}

Rules:
- Use ONLY function names listed in the tool catalog below.
- Provide ALL required arguments with correct JSON types.
- If a required value is not present in the user's messages or any prior
  tool output, ask for it via 'clarification' rather than fabricating it.
- When the user's request is fully satisfied, emit 'final_answer'.
"""


class MultiTurnMetaController:
    def build_prompt(
        self,
        history: History,
        state: State,
        tool_specs: list[dict],
    ) -> str:
        chat = history.render_chat() or "(no prior turns)"
        tools_json = json.dumps(tool_specs, indent=2)
        state_json = json.dumps(
            {
                "tool_outputs": _to_jsonable(state.tool_outputs),
                "variables": _to_jsonable(state.variables),
            },
            indent=2,
        )
        return (
            "You are a multi-turn function-calling agent.\n"
            "\n"
            f"{_PROPOSAL_FORMAT}\n"
            f"Tool catalog:\n{tools_json}\n"
            "\n"
            f"Conversation so far:\n{chat}\n"
            "\n"
            f"Persistent state:\n{state_json}\n"
            "\n"
            "Emit your next proposal as strict JSON only.\n"
        )

    def build_backprompt(
        self,
        history: History,
        state: State,
        tool_specs: list[dict],
        previous_proposal: str,
        feedback: str,
    ) -> str:
        return (
            f"{self.build_prompt(history, state, tool_specs)}"
            "\n"
            "Your previous proposal:\n"
            f"{previous_proposal}\n"
            "\n"
            "Validation errors:\n"
            f"{feedback}\n"
            "\n"
            "Fix the errors. Either:\n"
            "- correct the function call,\n"
            "- OR ask the user for missing information via a 'clarification' "
            "proposal,\n"
            "- OR provide a 'final_answer' if the task is already complete.\n"
            "Return only valid JSON.\n"
        )


def _to_jsonable(v):
    """Best-effort conversion to JSON-serialisable form (for the state dump)."""
    try:
        json.dumps(v)
        return v
    except TypeError:
        return repr(v)
