"""Plan -> BFCL call-string rendering, language-aware.

The runner serializes accepted Plans into the call-string shape that BFCL's
AST decoder consumes (e.g. "[f(x=1, y='a')]"). Different test categories
target different languages (Python / Java / JavaScript) and BFCL's per-
language decoders accept different syntax. Rendering with Python `repr()`
unconditionally produces strings that the Java decoder rejects (`True`
vs `true`, `{'k': 'v'}` Python dict literals).

This module exposes a single language-aware entry point so both the runner
(`run_bfcl_llm_modulo.py`) and the safety-net critic (`LanguageASTCritic`)
emit identical strings.

Supported languages: "python" (default), "java", "javascript".
"""

from __future__ import annotations

from typing import Any

from schemas import Plan


def render_value(value: Any, language: str = "python") -> str:
    """Render a single argument value as a literal in the target language."""
    if language in ("java", "javascript"):
        return _render_value_jvm_js(value, language)
    return repr(value)  # python: keep existing repr() behavior


def _render_value_jvm_js(value: Any, language: str) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        if language == "java":
            # Java doesn't accept bare `[a, b]` as a kwarg value (no array
            # literals in call positions), so render as a stringified list
            # using the same Python-repr convention as dicts. BFCL's eval
            # checker accepts the stringified form.
            inner = ", ".join(_render_value_jvm_js(x, language) for x in value)
            return '"[' + inner.replace('"', '\\"') + ']"'
        # JavaScript: bare array literals are fine.
        return "[" + ", ".join(_render_value_jvm_js(x, language) for x in value) + "]"
    if isinstance(value, dict):
        if language == "java":
            # Java: BFCL's eval accepts a stringified Python-repr dict (matches
            # the gold answer's HashMap representation, e.g.
            # params="{'limit': '50', 'schemaFilter': 'public'}").
            return '"' + repr(value).replace('"', '\\"') + '"'
        # JavaScript: standard JSON object literal — quoted string keys, language-
        # rendered values. JS's parser accepts this and the BFCL JS decoder is happy.
        items = ", ".join(
            f'"{k}": {_render_value_jvm_js(v, language)}' for k, v in value.items()
        )
        return "{" + items + "}"
    return repr(value)


def plan_to_bfcl_ast(plan: Plan, language: str = "python") -> str:
    """Render a Plan as a BFCL-decodable call-string for the target language."""
    parts: list[str] = []
    for step in plan.steps:
        arg_str = ", ".join(
            f"{k}={render_value(v, language)}" for k, v in step.arguments.items()
        )
        parts.append(f"{step.function}({arg_str})")
    return "[" + ", ".join(parts) + "]"
