"""LanguageASTCritic: reject plans whose rendered call-string can't be
parsed by BFCL's own per-language AST decoder.

Rationale: even with the language-aware serializer, edge cases (oddly
nested values, escaped quotes, model-supplied placeholders that leak the
prompt's `$varname` syntax) can produce strings BFCL won't decode at eval
time. Running BFCL's own `ast_parse` against the *exact string we will
ship to the result file* means a critic-accepted plan is guaranteed to be
eval-decodable. Methodology note: this critic intentionally reuses BFCL's
own decoder so there is zero parser-vs-evaluator gap.

Language is read from `context["language"]` (set by the runner from the
test category). If unset or "python", the critic defers — the existing
Python AST path already accepts what `repr()` emits.
"""

from __future__ import annotations

from critics.base import Critic
from render import plan_to_bfcl_ast
from schemas import CriticResult, Plan


_LANG_TO_RETURN_FORMAT = {
    "python": "python",
    "java": "java",
    "javascript": "javascript",
}


class LanguageASTCritic(Critic):
    """Hard critic. Calls bfcl_eval.model_handler.utils.ast_parse on the
    plan's rendered call string and fails if the decoder raises."""

    name = "LanguageASTCritic"

    def evaluate(self, plan: Plan, context: dict) -> CriticResult:
        language = (context.get("language") or "python").lower()
        if language not in _LANG_TO_RETURN_FORMAT:
            return self._pass()

        try:
            from bfcl_eval.constants.enums import ReturnFormat
            from bfcl_eval.model_handler.utils import ast_parse
        except ImportError as e:
            # The BFCL runner always has these installed. If the framework
            # is being exercised standalone (e.g. multi_turn demo), don't
            # block — just no-op.
            return self._pass(metadata={"skipped_reason": f"bfcl_eval unavailable: {e}"})

        rendered = plan_to_bfcl_ast(plan, language)
        return_format = getattr(ReturnFormat, language.upper())

        try:
            ast_parse(rendered, language=return_format)
        except Exception as e:  # noqa: BLE001 — BFCL raises a wide variety
            return self._fail(
                error=(
                    f"Rendered call-string failed to parse as {language.upper()}: "
                    f"{type(e).__name__}: {e}"
                ),
                suggestion=self._language_hints(language, rendered),
                metadata={"rendered": rendered, "language": language},
            )
        return self._pass(metadata={"rendered": rendered, "language": language})

    @staticmethod
    def _language_hints(language: str, rendered: str) -> str:
        if language == "java":
            return (
                "Rewrite the JSON plan so the resulting Java call parses. "
                "Use Java syntax: booleans are 'true'/'false' (lowercase, "
                "no quotes); pass map-typed args as a quoted string of the "
                "form \"{'key': 'value'}\"; do not emit Python literals "
                "like True / False / {'k':'v'}; do not leave the prompt's "
                "`$varname` placeholder syntax in argument values — write "
                "the bare identifier instead."
            )
        if language == "javascript":
            return (
                "Rewrite the JSON plan so the resulting JavaScript call "
                "parses. Use JS syntax: booleans are 'true'/'false' "
                "(lowercase, no quotes); object args use {\"key\": value} "
                "with quoted keys; do not leave the prompt's `$varname` "
                "syntax in argument values."
            )
        return "Rewrite the plan so the rendered call-string parses."
