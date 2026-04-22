"""BFCL adapter. Normalizes a BFCL-style task instance into our framework inputs.

BFCL stores tasks in a few shapes across categories; this adapter handles the
common variants and returns `(query, function_schemas)` ready for the loop.
"""

from typing import Any


def load_bfcl_task(task: dict) -> tuple[str, list[dict]]:
    """Return `(query, function_schemas)` extracted from a BFCL task dict."""
    return _extract_query(task), _extract_schemas(task)


def _extract_query(task: dict) -> str:
    if isinstance(task.get("query"), str):
        return task["query"]
    if isinstance(task.get("prompt"), str):
        return task["prompt"]

    q = task.get("question")
    # BFCL v3 multi-turn: list of turn-lists, each with chat-style dicts.
    if isinstance(q, list) and q and isinstance(q[0], list):
        parts = []
        for turn in q[0]:
            if isinstance(turn, dict) and "content" in turn:
                parts.append(str(turn["content"]))
        return "\n".join(parts)
    if isinstance(q, list) and q and isinstance(q[0], dict):
        return str(q[0].get("content", ""))
    if isinstance(q, str):
        return q

    raise ValueError("Could not extract query from BFCL task.")


def _extract_schemas(task: dict) -> list[dict]:
    for key in ("function", "functions", "tools"):
        val = task.get(key)
        if isinstance(val, list):
            return [_normalize_schema(s) for s in val]
    raise ValueError("Could not extract function schemas from BFCL task.")


def _normalize_schema(s: Any) -> dict:
    """Normalize a BFCL function entry to `{name, description, parameters}`."""
    if isinstance(s, dict) and "function" in s and isinstance(s["function"], dict):
        # OpenAI tool-style wrapper: {"type": "function", "function": {...}}
        s = s["function"]

    if not isinstance(s, dict):
        raise ValueError(f"Function schema must be a dict, got {type(s).__name__}.")

    return {
        "name": s.get("name"),
        "description": s.get("description", ""),
        "parameters": s.get("parameters")
        or {"type": "object", "properties": {}, "required": []},
    }
