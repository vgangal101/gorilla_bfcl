"""Core LLM-Modulo loop.

Generate -> Parse -> Test (bank of critics) -> Aggregate -> Backprompt,
until every hard critic passes or the iteration budget is exhausted.

Soundness comes from the critic bank; the LLM is only a proposal generator.
"""

from config import MAX_ITERATIONS, is_hard_critic
from schemas import CriticResult


def run_llm_modulo(
    query: str,
    function_schemas: list[dict],
    llm,
    critics: list,
    meta_controller,
    parser,
    max_iters: int = MAX_ITERATIONS,
) -> dict:
    """Run the LLM-Modulo loop and return a result dict.

    Return shape:
        {
          "status": "success" | "failure",
          "plan": Plan | None,
          "proposal": str | None,
          "results": list[CriticResult],   # from the terminating iteration
          "iterations": int,               # 1-indexed count of iterations used
          "history": list[dict],           # per-iteration records
        }
    """
    prompt = meta_controller.build_initial_prompt(query, function_schemas)
    history: list[dict] = []

    for iteration in range(max_iters):
        # -- Generate ---------------------------------------------------
        proposal = llm.generate(prompt)

        # -- Parse ------------------------------------------------------
        plan, parse_err = parser.parse(proposal)

        # -- Test -------------------------------------------------------
        if parse_err is not None:
            results = [CriticResult(
                critic_name="ParserCritic",
                status="fail",
                error=parse_err,
                suggestion=(
                    "Return only strict JSON with a 'steps' array. "
                    "No markdown fences, no commentary."
                ),
            )]
        else:
            context = {
                "query": query,
                "function_schemas": function_schemas,
                "history": history,
            }
            results = [c.evaluate(plan, context) for c in critics]

        # -- Acceptance check ------------------------------------------
        hard_results = [r for r in results if is_hard_critic(r.critic_name)]
        all_hard_pass = bool(hard_results) and all(r.status == "pass" for r in hard_results)

        if all_hard_pass and plan is not None:
            return {
                "status": "success",
                "plan": plan,
                "proposal": proposal,
                "results": results,
                "iterations": iteration + 1,
                "history": history,
            }

        # -- Critique + Backprompt --------------------------------------
        feedback = meta_controller.aggregate(results)
        history.append({
            "iteration": iteration,
            "proposal": proposal,
            "plan": plan,
            "results": results,
            "feedback": feedback,
        })
        prompt = meta_controller.build_backprompt(
            original_query=query,
            function_schemas=function_schemas,
            previous_proposal=proposal,
            feedback=feedback,
            history=history,
        )

    # Budget exhausted without acceptance.
    return {
        "status": "failure",
        "plan": history[-1]["plan"] if history else None,
        "proposal": history[-1]["proposal"] if history else None,
        "results": history[-1]["results"] if history else [],
        "iterations": max_iters,
        "history": history,
    }
