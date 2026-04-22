"""Adversarial / edge-case tests for the LLM-Modulo critic bank.

These probe real boundary conditions and implementation assumptions that
weren't covered by test_critics.py. Some of these are expected to FAIL —
that's intentional. A failing test here means a real gap in the
implementation that should be fixed or consciously accepted.

Each test is annotated with the expected behavior and whether it reveals
a known gap.
"""

import pytest

from schemas import FunctionCall, Plan, CriticResult
from critics.argument_schema import ArgumentSchemaCritic
from critics.argument_value import ArgumentValueCritic
from critics.dependency import DependencyCritic
from critics.redundancy import RedundancyCritic
from critics.function_validity import FunctionValidityCritic
from execution.executor import ToolExecutor
from parser import ProposalParser


def make_plan(*steps) -> Plan:
    plan = Plan()
    for step in steps:
        fn, args = step[0], step[1]
        out = step[2] if len(step) > 2 else None
        plan.steps.append(FunctionCall(function=fn, arguments=args, output_var=out))
    return plan


def ctx(schemas=None) -> dict:
    return {"query": "adversarial", "function_schemas": schemas or [], "history": []}


# ── ArgumentSchemaCritic edge cases ─────────────────────────────────────────────

NUMERIC_SCHEMA = [{
    "name": "fn",
    "parameters": {
        "properties": {
            "count": {"type": "integer"},
            "ratio": {"type": "number"},
        },
        "required": ["count"],
    },
}]


class TestArgumentSchemaEdgeCases:

    critic = ArgumentSchemaCritic()

    def test_float_rejected_for_integer_field(self):
        """3.0 is a float in Python; should be rejected for an integer field.

        The JSON spec technically allows 3.0 to represent the integer 3,
        but LLM output of 3.0 for an integer field is a type error in strict mode.
        EXPECTED: fail  — critic should reject float for integer.
        """
        plan = make_plan(("fn", {"count": 3.0}))
        r = self.critic.evaluate(plan, ctx(NUMERIC_SCHEMA))
        assert r.status == "fail", (
            "float 3.0 should be rejected for an integer field "
            "(LLMs sometimes output 3.0 instead of 3)"
        )

    def test_bool_rejected_for_number_field(self):
        """True is isinstance(True, (int, float)) == True in Python.

        ArgumentSchemaCritic only guards bool-for-integer, NOT bool-for-number.
        This means True/False silently pass for 'number' fields — likely a bug.
        EXPECTED: fail  — critic should reject bool for number too.
        """
        plan = make_plan(("fn", {"count": 1, "ratio": True}))
        r = self.critic.evaluate(plan, ctx(NUMERIC_SCHEMA))
        assert r.status == "fail", (
            "bool True should be rejected for a 'number' field, "
            "but the critic currently only guards bool-for-integer"
        )

    def test_none_rejected_for_string_field(self):
        """None should fail for a string-typed field."""
        schema = [{"name": "fn", "parameters": {
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }}]
        plan = make_plan(("fn", {"name": None}))
        r = self.critic.evaluate(plan, ctx(schema))
        assert r.status == "fail", "None should not satisfy a string type"

    def test_list_accepted_for_array_field(self):
        """A Python list should pass for an 'array' typed field."""
        schema = [{"name": "fn", "parameters": {
            "properties": {"items": {"type": "array"}},
            "required": ["items"],
        }}]
        plan = make_plan(("fn", {"items": [1, 2, 3]}))
        r = self.critic.evaluate(plan, ctx(schema))
        assert r.status == "pass"

    def test_schema_with_no_properties_key(self):
        """Schema missing 'properties' entirely should not crash."""
        schema = [{"name": "fn", "parameters": {"required": []}}]
        plan = make_plan(("fn", {"extra": "val"}))
        r = self.critic.evaluate(plan, ctx(schema))
        # No properties defined → 'extra' is unknown → should fail
        assert r.status == "fail", (
            "Argument not in schema properties should be rejected "
            "even when 'properties' key is missing from schema"
        )

    def test_schema_with_null_parameters(self):
        """Schema with parameters=null should not crash."""
        schema = [{"name": "fn", "parameters": None}]
        plan = make_plan(("fn", {}))
        r = self.critic.evaluate(plan, ctx(schema))
        assert r.status == "pass", "Empty args against null-parameters schema should pass"


# ── ArgumentValueCritic edge cases ───────────────────────────────────────────────

class TestArgumentValueEdgeCases:

    critic = ArgumentValueCritic()

    def test_exclusive_minimum_not_enforced(self):
        """exclusiveMinimum is a valid JSON Schema keyword but is NOT implemented.

        If a schema says exclusiveMinimum: 5, the value 5 should be rejected,
        but our critic only checks 'minimum', so 5 would pass.
        EXPECTED: fail  — reveals that exclusiveMinimum is silently ignored.
        """
        schema = [{"name": "fn", "parameters": {
            "properties": {"score": {"type": "integer", "exclusiveMinimum": 5}},
            "required": ["score"],
        }}]
        plan = make_plan(("fn", {"score": 5}))
        r = self.critic.evaluate(plan, ctx(schema))
        assert r.status == "fail", (
            "score=5 should violate exclusiveMinimum=5, "
            "but critic ignores exclusiveMinimum — known gap"
        )

    def test_enum_with_integer_values(self):
        """Enum containing integers (not strings) should still be enforced."""
        schema = [{"name": "fn", "parameters": {
            "properties": {"priority": {"type": "integer", "enum": [1, 2, 3]}},
            "required": ["priority"],
        }}]
        plan_pass = make_plan(("fn", {"priority": 2}))
        plan_fail = make_plan(("fn", {"priority": 5}))
        assert self.critic.evaluate(plan_pass, ctx(schema)).status == "pass"
        assert self.critic.evaluate(plan_fail, ctx(schema)).status == "fail"

    def test_email_with_subdomain(self):
        """user@mail.example.com is a valid email and should pass."""
        schema = [{"name": "fn", "parameters": {
            "properties": {"to": {"type": "string", "format": "email"}},
            "required": ["to"],
        }}]
        plan = make_plan(("fn", {"to": "user@mail.example.com"}))
        r = self.critic.evaluate(plan, ctx(schema))
        assert r.status == "pass"

    def test_minimum_zero_boundary(self):
        """minimum=0, value=0 is on the boundary — should pass (inclusive)."""
        schema = [{"name": "fn", "parameters": {
            "properties": {"n": {"type": "integer", "minimum": 0}},
            "required": ["n"],
        }}]
        plan = make_plan(("fn", {"n": 0}))
        assert self.critic.evaluate(plan, ctx(schema)).status == "pass"

    def test_float_value_against_float_minimum(self):
        """Numeric minimum check should work for floats, not just ints."""
        schema = [{"name": "fn", "parameters": {
            "properties": {"ratio": {"type": "number", "minimum": 0.5}},
            "required": ["ratio"],
        }}]
        plan_pass = make_plan(("fn", {"ratio": 0.5}))
        plan_fail = make_plan(("fn", {"ratio": 0.1}))
        assert self.critic.evaluate(plan_pass, ctx(schema)).status == "pass"
        assert self.critic.evaluate(plan_fail, ctx(schema)).status == "fail"


# ── DependencyCritic edge cases ──────────────────────────────────────────────────

class TestDependencyEdgeCases:

    critic = DependencyCritic()

    def test_var_ref_in_nested_dict_argument(self):
        """$var inside a nested dict value must also be resolved.

        The _refs() method recurses into dicts — this should be caught.
        """
        plan = make_plan(
            ("fn", {"config": {"key": "$undefined_nested"}})
        )
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "fail", "$var inside nested dict must be detected"

    def test_dollar_alone_is_not_a_ref(self):
        """A bare '$' with no name should NOT be treated as a variable reference."""
        plan = make_plan(("fn", {"x": "$"}))
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "pass", "bare '$' is not a variable reference"

    def test_dollar_with_space_is_not_a_ref(self):
        """'$ var' (with space) should not be a variable reference."""
        plan = make_plan(("fn", {"x": "$ not_a_var"}))
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "pass", "'$ name' with leading space is not a ref"

    def test_same_fn_same_args_different_output_var_is_duplicate(self):
        """Two steps with same (fn, args) but different output_var names.

        RedundancyCritic keys only on (function, arguments), NOT output_var.
        This pair IS flagged as a duplicate even though they bind different vars.
        Confirm that is the actual behavior (not necessarily correct, but known).
        """
        red = RedundancyCritic()
        plan = make_plan(
            ("get_weather", {"location": "NYC"}, "wx1"),
            ("get_weather", {"location": "NYC"}, "wx2"),
        )
        r = red.evaluate(plan, ctx())
        # Document actual behavior: currently fails as duplicate
        assert r.status == "fail", (
            "RedundancyCritic flags same (fn, args) as duplicate "
            "even when output_var differs — known design choice"
        )

    def test_output_var_with_numeric_start_rejected_by_regex(self):
        """output_var starting with a digit like '1result' cannot be $-referenced.

        The _VAR_RE = r'$([A-Za-z_][A-Za-z0-9_]*)' won't match '1result'.
        So a step with output_var='1result' produces a binding that can never
        be referenced. Dependency critic should still pass (no error on define).
        """
        plan = make_plan(
            ("fn", {}, "1result"),
        )
        r = self.critic.evaluate(plan, ctx())
        # '1result' IS stored as output_var; no reference error on production side
        assert r.status == "pass", (
            "Defining an output_var starting with digit is accepted "
            "(it just can't be referenced via $)"
        )


# ── ProposalParser edge cases ────────────────────────────────────────────────────

class TestParserEdgeCases:

    parser = ProposalParser()

    def test_whitespace_only_input(self):
        plan, err = self.parser.parse("   \n\t  ")
        assert plan is None and err is not None

    def test_empty_string_input(self):
        plan, err = self.parser.parse("")
        assert plan is None and err is not None

    def test_json_with_trailing_garbage(self):
        """Valid JSON followed by trailing text should fail."""
        raw = '{"steps": []} this is garbage'
        plan, err = self.parser.parse(raw)
        assert plan is None and err is not None, (
            "Trailing non-JSON text after a valid object should fail"
        )

    def test_output_var_empty_string_rejected(self):
        """output_var of '' (empty string) should be rejected."""
        raw = '{"steps": [{"function": "fn", "arguments": {}, "output_var": ""}]}'
        plan, err = self.parser.parse(raw)
        assert plan is None and err is not None

    def test_output_var_null_is_treated_as_none(self):
        """output_var: null in JSON should produce output_var=None (not a string)."""
        raw = '{"steps": [{"function": "fn", "arguments": {}, "output_var": null}]}'
        plan, err = self.parser.parse(raw)
        assert err is None and plan.steps[0].output_var is None

    def test_deeply_nested_markdown_fence(self):
        """JSON inside a code fence with leading prose."""
        raw = (
            "Here is my plan:\n"
            "```json\n"
            '{"steps": [{"function": "search", "arguments": {"query": "news"}}]}\n'
            "```\n"
            "Hope that helps!"
        )
        plan, err = self.parser.parse(raw)
        assert err is None
        assert plan.steps[0].function == "search"


# ── ToolExecutor edge cases ──────────────────────────────────────────────────────

class TestExecutorEdgeCases:

    def test_handler_returns_none(self):
        """Handler returning None should be stored as output and not error."""
        ex = ToolExecutor({"fn": lambda args: None})
        plan = make_plan(("fn", {}, "result"))
        trace = ex.execute_plan(plan)
        assert trace[0]["status"] == "ok"
        assert trace[0]["output"] is None

    def test_var_bound_to_none_resolves_to_none(self):
        """$var bound to None should resolve to None in subsequent steps."""
        ex = ToolExecutor({
            "produce": lambda args: None,
            "consume": lambda args: args.get("val"),
        })
        plan = make_plan(
            ("produce", {}, "x"),
            ("consume", {"val": "$x"}),
        )
        trace = ex.execute_plan(plan)
        assert trace[1]["status"] == "ok"
        assert trace[1]["output"] is None

    def test_var_bound_to_dict_resolves_correctly(self):
        """$var bound to a dict is substituted as the whole dict object."""
        ex = ToolExecutor({
            "get": lambda args: {"temp": 72, "condition": "sunny"},
            "use": lambda args: args["data"]["temp"],
        })
        plan = make_plan(
            ("get", {}, "weather"),
            ("use", {"data": "$weather"}),
        )
        trace = ex.execute_plan(plan)
        assert trace[1]["status"] == "ok"
        assert trace[1]["output"] == 72

    def test_multiple_inline_vars_in_one_string(self):
        """Multiple $var references in a single string argument."""
        ex = ToolExecutor({
            "city": lambda args: "NYC",
            "date": lambda args: "Monday",
            "fmt":  lambda args: args["msg"],
        })
        plan = make_plan(
            ("city", {}, "c"),
            ("date", {}, "d"),
            ("fmt",  {"msg": "Weather in $c on $d"}),
        )
        trace = ex.execute_plan(plan)
        assert trace[2]["status"] == "ok"
        assert trace[2]["output"] == "Weather in NYC on Monday"

    def test_continues_after_unresolved_var(self):
        """Executor records an error for the failed step and continues (no exception)."""
        ex = ToolExecutor({
            "fn": lambda args: "ok",
            "fn2": lambda args: "also ok",
        })
        plan = make_plan(
            ("fn",  {"x": "$missing"}),
            ("fn2", {"y": "literal"}),
        )
        trace = ex.execute_plan(plan)
        assert len(trace) == 2
        assert trace[0]["status"] == "error"
        assert trace[1]["status"] == "ok"


# ── FunctionValidityCritic edge cases ───────────────────────────────────────────

class TestFunctionValidityEdgeCases:

    critic = FunctionValidityCritic()

    def test_schema_entry_missing_name_key_is_skipped(self):
        """A schema entry with no 'name' key should be silently skipped
        (not crash), and the function set built from remaining entries."""
        schemas = [
            {"description": "no name here", "parameters": {}},
            {"name": "valid_fn", "parameters": {}},
        ]
        plan = make_plan(("valid_fn", {}))
        r = self.critic.evaluate(plan, ctx(schemas))
        assert r.status == "pass"

    def test_function_name_case_sensitive(self):
        """Function name matching must be case-sensitive."""
        schemas = [{"name": "get_weather", "parameters": {}}]
        plan = make_plan(("Get_Weather", {}))
        r = self.critic.evaluate(plan, ctx(schemas))
        assert r.status == "fail", "get_weather != Get_Weather"

    def test_empty_string_function_name_fails(self):
        """A step with function='' should fail (not in any schema)."""
        schemas = [{"name": "fn", "parameters": {}}]
        plan = Plan()
        plan.steps.append(FunctionCall(function="", arguments={}))
        r = self.critic.evaluate(plan, ctx(schemas))
        assert r.status == "fail"
