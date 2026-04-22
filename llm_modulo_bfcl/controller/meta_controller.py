"""Meta-controller: orchestrates prompts + aggregates critic feedback.

Responsibilities:
  * build the initial prompt given the query + schemas
  * rank critic failures (hard > soft, then by step index)
  * render a structured backprompt that tells the LLM exactly what to fix
"""

import json

from config import is_hard_critic
from schemas import AggregatedFeedback, CriticResult


class MetaController:
    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def build_initial_prompt(self, query: str, function_schemas: list[dict]) -> str:
        schemas_json = json.dumps(function_schemas, indent=2)
        return (
            "You are a function-calling planner. Produce a JSON plan that solves the query.\n"
            "\n"
            "Output format (strict JSON, no prose, no markdown fences):\n"
            "{\n"
            '  "steps": [\n'
            '    {"function": "<name>", "arguments": {...}, "output_var": "<optional>"}\n'
            "  ]\n"
            "}\n"
            "\n"
            "Rules:\n"
            "- Use ONLY functions defined in the schema below.\n"
            "- Provide ALL required arguments with correct JSON types.\n"
            "- To reuse a previous step's result, set its `output_var` and reference it\n"
            "  as `$<output_var>` in a later argument.\n"
            "- Return valid JSON only.\n"
            "\n"
            f"Query:\n{query}\n"
            "\n"
            f"Available functions:\n{schemas_json}\n"
        )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate(self, results: list[CriticResult]) -> AggregatedFeedback:
        failures = [r for r in results if r.status == "fail"]
        failures.sort(
            key=lambda r: (
                0 if is_hard_critic(r.critic_name) else 1,
                r.step_index if r.step_index is not None else -1,
            )
        )

        instructions: list[str] = []
        for r in failures:
            severity = "hard" if is_hard_critic(r.critic_name) else "soft"
            loc = f"step {r.step_index}" if r.step_index is not None else "plan"
            line = f"- [{severity}][{r.critic_name}][{loc}] {r.error}"
            if r.suggestion:
                line += f" (fix: {r.suggestion})"
            instructions.append(line)

        hard_count = sum(1 for r in failures if is_hard_critic(r.critic_name))
        summary = (
            f"{len(failures)} validation issue(s); {hard_count} hard, "
            f"{len(failures) - hard_count} soft."
        )
        return AggregatedFeedback(
            failures=failures, instructions=instructions, summary=summary
        )

    # ------------------------------------------------------------------
    # Backprompting
    # ------------------------------------------------------------------

    def build_backprompt(
        self,
        original_query: str,
        function_schemas: list[dict],
        previous_proposal: str,
        feedback: AggregatedFeedback,
        history: list,
    ) -> str:
        schemas_json = json.dumps(function_schemas, indent=2)
        issues = "\n".join(feedback.instructions) if feedback.instructions else "- (none)"

        prior = ""
        # Include a compact trail of prior attempts to discourage loops.
        if len(history) >= 2:
            lines = [
                f"- iteration {h['iteration']}: {h['feedback'].summary}"
                for h in history[:-1][-3:]
            ]
            prior = "\n".join(lines)

        msg = (
            "You previously proposed this function call plan:\n\n"
            f"{previous_proposal}\n\n"
            "The following validation errors were found:\n\n"
            f"{issues}\n\n"
            "Regenerate the FULL function-call plan as strict JSON.\n"
            "Requirements:\n"
            "- use only valid functions from the schema\n"
            "- satisfy all required arguments with correct types\n"
            "- preserve valid step ordering\n"
            "- correctly chain outputs between steps (use `$var` only for previously\n"
            "  produced `output_var` values)\n"
            "- do not repeat the same incorrect plan\n"
            "Return only valid JSON. No prose, no code fences.\n"
            "\n"
            f"Original query:\n{original_query}\n"
            "\n"
            f"Available functions:\n{schemas_json}\n"
        )
        if prior:
            msg += f"\nEarlier attempts summary:\n{prior}\n"
        return msg
