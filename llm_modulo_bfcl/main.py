"""Runnable demo.

Uses the sample BFCL task + a scripted MockLLM that improves across iterations:
  iter 1: invalid function name
  iter 2: missing required argument
  iter 3: valid plan -> accepted by all hard critics

Run from inside `llm_modulo_bfcl/`:
    python main.py
"""

import json

from bfcl_adapter import load_bfcl_task
from controller.meta_controller import MetaController
from critics.argument_schema import ArgumentSchemaCritic
from critics.argument_value import ArgumentValueCritic
from critics.dependency import DependencyCritic
from critics.execution import ExecutionCritic
from critics.function_validity import FunctionValidityCritic
from critics.redundancy import RedundancyCritic
from examples.sample_bfcl_task import SAMPLE_TASK
from execution.executor import ToolExecutor
from llm_interface import MockLLM
from loop import run_llm_modulo
from parser import ProposalParser


def build_executor() -> ToolExecutor:
    def get_weather(args):
        return {"city": args["city"], "temp_f": 102, "conditions": "sunny"}

    def send_email(args):
        return {"delivered": True, "to": args["to"], "subject": args["subject"]}

    return ToolExecutor(handlers={
        "get_weather": get_weather,
        "send_email": send_email,
    })


def build_mock_llm() -> MockLLM:
    # Attempt 1: invalid function name.
    attempt_1 = {
        "steps": [
            {
                "function": "fetch_forecast",
                "arguments": {"location": "Phoenix"},
                "output_var": "weather",
            },
            {
                "function": "send_email",
                "arguments": {
                    "to": "user@example.com",
                    "subject": "Weather Report",
                    "body": "Today: $weather",
                },
            },
        ]
    }
    # Attempt 2: correct function, but missing required 'city'.
    attempt_2 = {
        "steps": [
            {
                "function": "get_weather",
                "arguments": {},
                "output_var": "weather",
            },
            {
                "function": "send_email",
                "arguments": {
                    "to": "user@example.com",
                    "subject": "Weather Report",
                    "body": "Today: $weather",
                },
            },
        ]
    }
    # Attempt 3: valid.
    attempt_3 = {
        "steps": [
            {
                "function": "get_weather",
                "arguments": {"city": "Phoenix", "units": "fahrenheit"},
                "output_var": "weather",
            },
            {
                "function": "send_email",
                "arguments": {
                    "to": "user@example.com",
                    "subject": "Weather Report",
                    "body": "Phoenix weather today: $weather",
                },
            },
        ]
    }
    return MockLLM(responses=[
        json.dumps(attempt_1),
        json.dumps(attempt_2),
        json.dumps(attempt_3),
    ])


def main() -> None:
    query, schemas = load_bfcl_task(SAMPLE_TASK)

    llm = build_mock_llm()
    executor = build_executor()
    critics = [
        FunctionValidityCritic(),
        ArgumentSchemaCritic(),
        ArgumentValueCritic(),
        DependencyCritic(),
        ExecutionCritic(executor),
        RedundancyCritic(),
    ]

    result = run_llm_modulo(
        query=query,
        function_schemas=schemas,
        llm=llm,
        critics=critics,
        meta_controller=MetaController(),
        parser=ProposalParser(),
        max_iters=5,
    )

    print("=" * 70)
    print(f"STATUS: {result['status']}")
    print(f"ITERATIONS: {result['iterations']}")
    print("=" * 70)

    if result["plan"] is not None:
        print("\nFinal plan:")
        for i, step in enumerate(result["plan"].steps):
            tail = f" -> ${step.output_var}" if step.output_var else ""
            print(f"  [{i}] {step.function}({step.arguments}){tail}")

    print("\nCritic results (final iteration):")
    for r in result["results"]:
        mark = "PASS" if r.status == "pass" else "FAIL"
        where = f" @ step {r.step_index}" if r.step_index is not None else ""
        msg = f" — {r.error}" if r.error else ""
        print(f"  [{mark}] {r.critic_name}{where}{msg}")

    print("\nPer-iteration summary:")
    for h in result.get("history", []):
        print(f"  iter {h['iteration']}: {h['feedback'].summary}")

    # Execution trace from the accepting iteration (if any).
    exec_results = [r for r in result["results"] if r.critic_name == "ExecutionCritic"]
    if exec_results and exec_results[0].metadata and "trace" in exec_results[0].metadata:
        print("\nExecution trace:")
        for i, entry in enumerate(exec_results[0].metadata["trace"]):
            print(f"  step {i} [{entry.get('status')}] {entry.get('function')} "
                  f"-> {entry.get('output', entry.get('error'))}")


if __name__ == "__main__":
    main()
