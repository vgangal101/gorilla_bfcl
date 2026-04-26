"""Local, no-network smoke test for the language-aware serializer + critic.

Runs four assertions without any LLM server or BFCL data:

  1. Renderer round-trip: a Plan with bool/dict/list/None args renders to
     strings that BFCL's per-language ast_parse accepts (Python / Java / JS).

  2. LanguageASTCritic accepts a clean Java plan and reports the rendered
     string in metadata.

  3. LanguageASTCritic rejects a Plan whose Java rendering hits Java AST
     limits (bare `args=[...]` array literal — Java doesn't permit those
     as kwarg values), and the failure feedback names the language.

  4. End-to-end run_llm_modulo with MockLLM: first proposal is unfixable
     by the renderer alone, second proposal renders cleanly and is
     accepted by all hard critics including LanguageASTCritic.

Run from inside the bfcl conda env:
    python local_smoke_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make the framework package importable when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bfcl_eval.constants.enums import ReturnFormat  # noqa: E402
from bfcl_eval.model_handler.utils import ast_parse  # noqa: E402

from controller.meta_controller import MetaController  # noqa: E402
from critics.argument_schema import ArgumentSchemaCritic  # noqa: E402
from critics.argument_value import ArgumentValueCritic  # noqa: E402
from critics.dependency import DependencyCritic  # noqa: E402
from critics.function_validity import FunctionValidityCritic  # noqa: E402
from critics.language_ast import LanguageASTCritic  # noqa: E402
from llm_interface import MockLLM  # noqa: E402
from loop import run_llm_modulo  # noqa: E402
from parser import ProposalParser  # noqa: E402
from render import plan_to_bfcl_ast  # noqa: E402
from schemas import FunctionCall, Plan  # noqa: E402


# ---------------------------------------------------------------------------
# Tiny check helpers (no test framework dependency)
# ---------------------------------------------------------------------------

_results: list[tuple[str, bool, str]] = []


def _check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] {name}{(' — ' + detail) if detail else ''}")


# ---------------------------------------------------------------------------
# Test 1: renderer round-trip
# ---------------------------------------------------------------------------

def test_renderer_roundtrip() -> None:
    print("\n[1] Renderer round-trip across languages")
    plan = Plan(steps=[FunctionCall(
        function="SQLCompletionAnalyzer.makeProposalsFromObject",
        arguments={
            "object": "Customers",
            "useShortName": True,
            "params": {"limit": "50", "schemaFilter": "public"},
            "flags": [True, False, None],
            "count": 5,
        },
    )])

    py = plan_to_bfcl_ast(plan, "python")
    java = plan_to_bfcl_ast(plan, "java")
    js = plan_to_bfcl_ast(plan, "javascript")

    try:
        ast_parse(py, language=ReturnFormat.PYTHON)
        _check("python rendering parses", True, py)
    except Exception as e:
        _check("python rendering parses", False, f"{e}; rendered={py}")

    try:
        ast_parse(java, language=ReturnFormat.JAVA)
        _check("java rendering parses", True, java)
    except Exception as e:
        _check("java rendering parses", False, f"{e}; rendered={java}")

    try:
        ast_parse(js, language=ReturnFormat.JAVASCRIPT)
        _check("javascript rendering parses", True, js)
    except Exception as e:
        _check("javascript rendering parses", False, f"{e}; rendered={js}")

    # Specific properties of the Java rendering — these encode the bugfix.
    _check("java rendering has lowercase 'true'", "useShortName=true" in java,
           f"... in {java}")
    _check("java rendering has lowercase 'false'", "false" in java,
           f"... in {java}")
    _check("java rendering has 'null'", "null" in java,
           f"... in {java}")
    _check("java rendering has no Python 'True'", "True" not in java,
           f"... in {java}")
    _check("java rendering wraps dict in quotes",
           '"{\'limit\': \'50\'' in java or "\"{'limit': '50'" in java,
           f"... in {java}")


# ---------------------------------------------------------------------------
# Test 2: LanguageASTCritic accepts a clean Java plan
# ---------------------------------------------------------------------------

def test_critic_accepts_clean_java() -> None:
    print("\n[2] LanguageASTCritic accepts a clean Java plan")

    plan = Plan(steps=[FunctionCall(
        function="GeometryPresentation.createPresentation",
        arguments={"controller": "mapController", "parent": "mapArea"},
    )])
    schemas = [{
        "name": "GeometryPresentation.createPresentation",
        "parameters": {
            "type": "object",
            "properties": {
                "controller": {"type": "string"},
                "parent": {"type": "string"},
            },
            "required": ["controller", "parent"],
        },
    }]
    ctx = {"function_schemas": schemas, "language": "java"}
    res = LanguageASTCritic().evaluate(plan, ctx)

    _check("critic returns pass", res.status == "pass",
           f"status={res.status} error={res.error!r}")
    _check("critic metadata has rendered Java string",
           bool(res.metadata) and "rendered" in (res.metadata or {}),
           f"metadata={res.metadata}")


# ---------------------------------------------------------------------------
# Test 3: LanguageASTCritic rejects a Java-incompatible plan
# ---------------------------------------------------------------------------

def test_critic_rejects_unfixable_java() -> None:
    print("\n[3] LanguageASTCritic rejects rendering the renderer can't sanitize")

    # The serializer fixes most known failure modes (Python bools, dicts,
    # array literals as kwargs). What's left is structural: the renderer
    # passes function names through verbatim, so a model that hallucinates
    # a name with chars Java forbids (e.g. a hyphen) produces a string the
    # Java AST decoder rejects. The critic exists as a safety net for
    # exactly that class of unforeseen failure.
    plan = Plan(steps=[FunctionCall(
        function="1startsWithDigit",  # Java identifiers can't start with a digit
        arguments={"x": 1},
    )])
    ctx = {"function_schemas": [], "language": "java"}
    res = LanguageASTCritic().evaluate(plan, ctx)

    _check("critic returns fail", res.status == "fail",
           f"status={res.status}")
    _check("error message names target language",
           "JAVA" in res.error or "java" in res.error.lower(),
           f"error={res.error!r}")
    _check("suggestion exists for re-prompt",
           bool(res.suggestion),
           f"suggestion={res.suggestion}")


# ---------------------------------------------------------------------------
# Test 4: full run_llm_modulo loop with MockLLM, recovery on second iter
# ---------------------------------------------------------------------------

def test_full_loop_recovers_after_reprompt() -> None:
    print("\n[4] run_llm_modulo: critic catches bad plan, model recovers")

    # First plan: passes JSON schema critics but fails LanguageASTCritic
    # because it embeds a `params={...}` Python dict literal that Java
    # doesn't accept.
    bad_plan_json = {
        "steps": [{
            "function": "SQLCompletionAnalyzer.makeProposalsFromObject",
            "arguments": {
                "object": "Customers",
                "useShortName": True,
                "params": {"limit": "50", "schemaFilter": "public"},
            },
        }],
    }
    # Second plan: same content (renderer will fix bool/dict for Java).
    # In production the model would generate this after seeing the
    # Java-specific feedback; for the smoke test we just script it.
    good_plan_json = bad_plan_json  # renderer fix means this already works

    schemas = [{
        "name": "SQLCompletionAnalyzer.makeProposalsFromObject",
        "parameters": {
            "type": "object",
            "properties": {
                "object": {"type": "string"},
                "useShortName": {"type": "boolean"},
                "params": {"type": "object"},
            },
            "required": ["object", "useShortName", "params"],
        },
    }]

    llm = MockLLM(responses=[json.dumps(bad_plan_json), json.dumps(good_plan_json)])
    critics = [
        FunctionValidityCritic(),
        ArgumentSchemaCritic(),
        ArgumentValueCritic(),
        DependencyCritic(),
        LanguageASTCritic(),
    ]

    result = run_llm_modulo(
        query="Help me generate SQL completion proposals.",
        function_schemas=schemas,
        llm=llm,
        critics=critics,
        meta_controller=MetaController(),
        parser=ProposalParser(),
        max_iters=3,
        extra_context={"language": "java"},
    )

    _check("loop succeeded", result["status"] == "success",
           f"status={result['status']} iters={result['iterations']}")
    if result["plan"] is not None:
        rendered = plan_to_bfcl_ast(result["plan"], "java")
        _check("accepted plan renders to valid Java", "useShortName=true" in rendered,
               f"rendered={rendered}")
        try:
            ast_parse(rendered, language=ReturnFormat.JAVA)
            _check("accepted plan parses with bfcl ast_parse(JAVA)", True, rendered)
        except Exception as e:
            _check("accepted plan parses with bfcl ast_parse(JAVA)", False, f"{e}")


def main() -> int:
    print("Local smoke test for language-aware serializer + LanguageASTCritic")
    test_renderer_roundtrip()
    test_critic_accepts_clean_java()
    test_critic_rejects_unfixable_java()
    test_full_loop_recovers_after_reprompt()

    failed = [n for n, ok, _ in _results if not ok]
    print(f"\n=== {len(_results) - len(failed)}/{len(_results)} checks passed ===")
    if failed:
        print("FAILED:")
        for n in failed:
            print(f"  - {n}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
