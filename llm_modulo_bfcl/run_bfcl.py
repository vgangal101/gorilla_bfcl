"""BFCL-style runner.

Routes BFCL task instances through the LLM-Modulo loop. Replace
`default_llm_for` with a real client (OpenAI / Anthropic / local vLLM)
to evaluate against BFCL data.

Usage:
    python run_bfcl.py                       # runs the sample task
    python run_bfcl.py path/to/tasks.jsonl   # runs every task in a JSONL file
"""

import json
import sys

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


def default_llm_for(query: str, schemas: list[dict]) -> MockLLM:
    """Placeholder LLM. Swap for a real client in production runs."""
    if not schemas:
        return MockLLM(responses=['{"steps": []}'])
    fn = schemas[0]
    props = (fn.get("parameters") or {}).get("properties", {}) or {}
    args = {k: "TODO" for k in props.keys()}
    plan = {"steps": [{"function": fn["name"], "arguments": args}]}
    return MockLLM(responses=[json.dumps(plan)])


def default_executor(schemas: list[dict]) -> ToolExecutor:
    """Stub executor: every function returns a marker dict echoing its args."""
    handlers = {
        s["name"]: (lambda args, n=s["name"]: {"_called": n, **args})
        for s in schemas
    }
    return ToolExecutor(handlers=handlers)


def run_task(task: dict, llm=None, executor=None, max_iters: int = 10) -> dict:
    query, schemas = load_bfcl_task(task)
    llm = llm or default_llm_for(query, schemas)
    executor = executor or default_executor(schemas)

    critics = [
        FunctionValidityCritic(),
        ArgumentSchemaCritic(),
        ArgumentValueCritic(),
        DependencyCritic(),
        ExecutionCritic(executor),
        RedundancyCritic(),
    ]

    return run_llm_modulo(
        query=query,
        function_schemas=schemas,
        llm=llm,
        critics=critics,
        meta_controller=MetaController(),
        parser=ProposalParser(),
        max_iters=max_iters,
    )


def _load_jsonl(path: str) -> list[dict]:
    tasks: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))
    return tasks


def main(argv: list[str]) -> int:
    tasks = _load_jsonl(argv[1]) if len(argv) > 1 else [SAMPLE_TASK]

    n_success = 0
    for i, task in enumerate(tasks):
        task_id = task.get("id", f"task_{i}")
        print(f"\n=== {task_id} ===")
        try:
            result = run_task(task)
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR running task: {e}")
            continue

        print(f"status={result['status']} iterations={result['iterations']}")
        if result["plan"] is not None:
            for j, step in enumerate(result["plan"].steps):
                print(f"  step {j}: {step.function}({step.arguments})")
        for r in result["results"]:
            if r.status == "fail":
                loc = f"@step {r.step_index}" if r.step_index is not None else ""
                print(f"  FAIL {r.critic_name} {loc}: {r.error}")

        if result["status"] == "success":
            n_success += 1

    print(f"\n{n_success} / {len(tasks)} tasks succeeded.")
    return 0 if n_success == len(tasks) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
