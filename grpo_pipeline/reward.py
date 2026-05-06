"""
reward.py — Wraps the BFCL AST checker as a scalar reward signal for GRPO.

The reward is 1.0 if the generated function call(s) pass ast_checker, else 0.0.
"""

import ast as _ast
import sys
from pathlib import Path

# Make bfcl_eval importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "berkeley-function-call-leaderboard"))

from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
from bfcl_eval.constants.enums import Language

LANG_MAP = {
    "simple_python": Language.PYTHON,
    "parallel": Language.PYTHON,
    "multiple": Language.PYTHON,
    "simple_java": Language.JAVA,
    "simple_javascript": Language.JAVASCRIPT,
}


def _get_func_name(node) -> str:
    """Reconstruct function name from AST node (handles dotted names)."""
    if isinstance(node, _ast.Name):
        return node.id
    if isinstance(node, _ast.Attribute):
        return f"{_get_func_name(node.value)}.{node.attr}"
    return ""


def parse_model_output(text: str, language: Language) -> list[dict]:
    """
    Convert '[func(a=1, b=2)]' → [{"func": {"a": 1, "b": 2}}].

    Uses Python's built-in ast module — bfcl_eval has no public parser API
    at a stable path, so we parse with stdlib and produce the dict format
    that ast_checker expects.
    Handles single calls, multiple calls (parallel), and dotted names.
    """
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text.strip():
        return []
    try:
        # Wrap in a list so ast.parse handles comma-separated calls cleanly
        tree = _ast.parse(f"[{text}]", mode="eval")
        results = []
        for elt in tree.body.elts:
            if not isinstance(elt, _ast.Call):
                return []
            func_name = _get_func_name(elt.func)
            if not func_name:
                return []
            args = {kw.arg: _ast.literal_eval(kw.value) for kw in elt.keywords}
            results.append({func_name: args})
        return results
    except Exception:
        return []


def compute_reward(
    generated: str,
    function_schemas: list[dict],
    ground_truth: list[dict],
    category: str,
) -> float:
    language = LANG_MAP.get(category, Language.PYTHON)
    parsed = parse_model_output(generated, language)
    if not parsed:
        return 0.0

    try:
        result = ast_checker(
            func_description=function_schemas,
            model_output=parsed,
            possible_answer=ground_truth[0],
            language=language,
            test_category=category,
            model_name="grpo_model",
        )
        return 1.0 if result["valid"] else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# TRL-compatible reward function
# GRPOTrainer (TRL 0.12+) calls: reward_fn(completions, **dataset_columns)
# completions is the only positional argument; dataset columns come as kwargs.
# ---------------------------------------------------------------------------

def bfcl_reward_fn(completions: list[str], **kwargs) -> list[float]:
    category     = kwargs["category"]
    function     = kwargs["function"]
    ground_truth = kwargs["ground_truth"]

    rewards = []
    for completion, cat, fn_schemas, gt in zip(
        completions, category, function, ground_truth
    ):
        rewards.append(
            compute_reward(
                generated=completion,
                function_schemas=fn_schemas,
                ground_truth=gt,
                category=cat,
            )
        )
    return rewards


if __name__ == "__main__":
    # Quick smoke test
    sample_schemas = [{
        "name": "calculate_triangle_area",
        "description": "Calculate the area of a triangle.",
        "parameters": {
            "type": "dict",
            "properties": {
                "base": {"type": "integer", "description": "Base"},
                "height": {"type": "integer", "description": "Height"},
            },
            "required": ["base", "height"],
        },
    }]
    sample_gt = [{"calculate_triangle_area": {"base": [10], "height": [5]}}]

    good = compute_reward(
        "calculate_triangle_area(base=10, height=5)",
        sample_schemas, sample_gt, "simple_python",
    )
    bad = compute_reward(
        "calculate_triangle_area(base=99, height=5)",
        sample_schemas, sample_gt, "simple_python",
    )
    unparseable = compute_reward(
        "I cannot determine the answer.",
        sample_schemas, sample_gt, "simple_python",
    )
    print(f"correct call → reward={good}")       # expect 1.0
    print(f"wrong value  → reward={bad}")        # expect 0.0
    print(f"unparseable  → reward={unparseable}") # expect 0.0
