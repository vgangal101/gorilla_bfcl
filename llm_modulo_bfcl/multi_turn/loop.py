"""Multi-turn LLM-Modulo loop.

For each user message:
  inner step loop (assistant may call multiple tools per user turn):
    modulo loop: generate -> parse -> critique -> backprompt, until accepted
                 (or budget exhausted)
    apply the accepted proposal:
      function_call : pull the cached tool output from RuntimeCritic's
                      metadata, update state, log, continue stepping.
      clarification : log, break to next user message.
      final_answer  : log, terminate.

The function is a generator. Each iteration yields one event dict so callers
can stream the conversation:

    {"event": "user_message",  "message": str}
    {"event": "tool_call",     "function": str, "arguments": dict, "output": Any}
    {"event": "clarification", "message": str}
    {"event": "final_answer",  "message": str}
    {"event": "error",         "message": str}
    {"event": "end_of_user_messages"}
"""

from typing import Iterable, Iterator

from .aggregator import aggregate
from .meta_controller import MultiTurnMetaController
from .parser import ProposalParser
from .schemas import History, Proposal, State


def run_multi_turn(
    *,
    user_messages: Iterable[str],
    llm,
    parser: ProposalParser,
    critics: list,
    meta: MultiTurnMetaController,
    tool_specs: list[dict],
    tool_registry: dict,
    max_modulo_iters: int = 5,
    max_steps_per_turn: int = 10,
) -> Iterator[dict]:
    state = State()
    history = History()

    for user_msg in user_messages:
        history.add_user(user_msg)
        yield {"event": "user_message", "message": user_msg}

        for _ in range(max_steps_per_turn):
            outcome = _run_modulo(
                history=history,
                state=state,
                tool_specs=tool_specs,
                tool_registry=tool_registry,
                llm=llm,
                parser=parser,
                meta=meta,
                critics=critics,
                max_iters=max_modulo_iters,
            )

            if outcome["status"] == "exhausted":
                yield {
                    "event": "error",
                    "message": (
                        f"modulo loop exhausted after {max_modulo_iters} "
                        f"iterations; last feedback:\n{outcome['feedback']}"
                    ),
                }
                return

            proposal: Proposal = outcome["proposal"]
            results = outcome["results"]

            if proposal.type == "function_call":
                output = _runtime_output(results)
                state.tool_outputs[proposal.function_name] = output
                history.add_assistant(proposal, tool_output=output)
                yield {
                    "event": "tool_call",
                    "function": proposal.function_name,
                    "arguments": proposal.arguments,
                    "output": output,
                }
                continue  # next step within the same user turn

            if proposal.type == "clarification":
                history.add_assistant(proposal)
                yield {"event": "clarification", "message": proposal.message}
                break  # leave step loop, pull next user message

            if proposal.type == "final_answer":
                history.add_assistant(proposal)
                yield {"event": "final_answer", "message": proposal.message}
                return

            yield {
                "event": "error",
                "message": f"unrecognised proposal type: {proposal.type!r}",
            }
            return
        else:
            yield {
                "event": "error",
                "message": (
                    f"reached max_steps_per_turn={max_steps_per_turn} without "
                    f"a clarification or final_answer"
                ),
            }
            return

    yield {"event": "end_of_user_messages"}


def _run_modulo(
    *,
    history: History,
    state: State,
    tool_specs: list[dict],
    tool_registry: dict,
    llm,
    parser: ProposalParser,
    meta: MultiTurnMetaController,
    critics: list,
    max_iters: int,
) -> dict:
    """Single Generate -> Test -> Critique -> Backprompt loop for one proposal.

    Returns:
      {"status": "ok",        "proposal": Proposal, "results": list[CriticResult]}
      {"status": "exhausted", "feedback": str}
    """
    feedback = ""
    raw = ""
    ctx = {
        "history": history,
        "state": state,
        "tool_specs": tool_specs,
        "tool_registry": tool_registry,
    }

    for iteration in range(max_iters):
        prompt = (
            meta.build_prompt(history, state, tool_specs)
            if iteration == 0
            else meta.build_backprompt(history, state, tool_specs, raw, feedback)
        )

        raw = llm.generate(prompt)
        proposal, parse_err = parser.parse(raw)

        if parse_err is not None:
            feedback = f"- [Parser] {parse_err}"
            continue

        results = [c.evaluate(proposal, ctx) for c in critics]
        if all(r.passed for r in results):
            return {"status": "ok", "proposal": proposal, "results": results}

        feedback = aggregate(results)

    return {"status": "exhausted", "feedback": feedback}


def _runtime_output(results) -> object:
    """Read the tool output that RuntimeCritic captured during evaluation."""
    for r in results:
        if r.name == "RuntimeCritic" and r.metadata:
            return r.metadata.get("output")
    return None
