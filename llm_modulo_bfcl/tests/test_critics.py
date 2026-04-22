"""Comprehensive unit tests for the LLM-Modulo framework.

Run from repo root:
    python -m pytest llm_modulo_bfcl/tests/ -v

Covers:
  - All 6 critics (FunctionValidity, ArgumentSchema, ArgumentValue,
    Dependency, Execution, Redundancy) with explicit pass AND fail cases
  - ProposalParser (valid JSON, markdown fences, malformed inputs)
  - ToolExecutor (execution, variable resolution, error handling)
  - MetaController (prompt building, aggregation, backprompting)
  - BFCLAdapter (multi-format task normalisation)
  - Config (hard/soft critic classification)
"""

import pytest

from schemas import FunctionCall, Plan, CriticResult, AggregatedFeedback
from critics.function_validity import FunctionValidityCritic
from critics.argument_schema import ArgumentSchemaCritic
from critics.argument_value import ArgumentValueCritic
from critics.dependency import DependencyCritic
from critics.execution import ExecutionCritic
from critics.redundancy import RedundancyCritic
from execution.executor import ToolExecutor
from parser import ProposalParser
from controller.meta_controller import MetaController
from bfcl_adapter import load_bfcl_task
from config import is_hard_critic, HARD_CRITICS, SOFT_CRITICS


# ── Shared fixtures ─────────────────────────────────────────────────────────────

SCHEMAS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "units":    {"type": "string", "enum": ["celsius", "fahrenheit"]},
                "days":     {"type": "integer", "minimum": 1, "maximum": 14},
            },
            "required": ["location"],
        },
    },
    {
        "name": "send_email",
        "description": "Send an email message.",
        "parameters": {
            "type": "object",
            "properties": {
                "to":      {"type": "string", "format": "email"},
                "subject": {"type": "string"},
                "body":    {"type": "string"},
                "date":    {"type": "string", "format": "date"},
            },
            "required": ["to", "subject", "body"],
        },
    },
    {
        "name": "search",
        "description": "Search the web.",
        "parameters": {
            "type": "object",
            "properties": {
                "query":       {"type": "string", "pattern": r"^.{3,}$"},
                "max_results": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
]


def make_plan(*steps) -> Plan:
    """Build a Plan from (function, arguments[, output_var]) tuples."""
    plan = Plan()
    for step in steps:
        fn, args = step[0], step[1]
        out_var = step[2] if len(step) > 2 else None
        plan.steps.append(FunctionCall(function=fn, arguments=args, output_var=out_var))
    return plan


def ctx(schemas=None) -> dict:
    return {"query": "unit-test", "function_schemas": schemas if schemas is not None else SCHEMAS, "history": []}


# ── Config ──────────────────────────────────────────────────────────────────────

class TestConfig:
    def test_hard_critics_classified_correctly(self):
        for name in HARD_CRITICS:
            assert is_hard_critic(name), f"{name} should be hard"

    def test_soft_critics_classified_correctly(self):
        for name in SOFT_CRITICS:
            assert not is_hard_critic(name), f"{name} should be soft"

    def test_unknown_name_is_not_hard(self):
        assert not is_hard_critic("MadeUpCritic")

    def test_parser_critic_is_hard(self):
        assert is_hard_critic("ParserCritic")


# ── FunctionValidityCritic ──────────────────────────────────────────────────────

class TestFunctionValidityCritic:
    critic = FunctionValidityCritic()

    # ── PASS cases ──────────────────────────────────────────────────────────────

    def test_pass_single_known_function(self):
        plan = make_plan(("get_weather", {"location": "NYC"}))
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "pass"

    def test_pass_empty_plan(self):
        r = self.critic.evaluate(Plan(), ctx())
        assert r.status == "pass"

    def test_pass_multi_step_all_valid(self):
        plan = make_plan(
            ("get_weather", {"location": "NYC"}),
            ("send_email",  {"to": "a@b.com", "subject": "wx", "body": "text"}),
            ("search",      {"query": "news"}),
        )
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "pass"

    # ── FAIL cases ──────────────────────────────────────────────────────────────

    def test_fail_unknown_function_at_step_0(self):
        plan = make_plan(("fly_to_moon", {"destination": "moon"}))
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "fail"
        assert "fly_to_moon" in r.error
        assert r.step_index == 0

    def test_fail_unknown_function_at_step_1(self):
        plan = make_plan(
            ("get_weather",  {"location": "NYC"}),
            ("launch_rocket", {"fuel": "hydrogen"}),
        )
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "fail"
        assert r.step_index == 1

    def test_fail_empty_schema_set(self):
        plan = make_plan(("get_weather", {"location": "NYC"}))
        r = self.critic.evaluate(plan, ctx(schemas=[]))
        assert r.status == "fail"

    def test_fail_suggestion_lists_known_functions(self):
        plan = make_plan(("bad_func", {}))
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "fail"
        assert r.suggestion is not None
        assert "get_weather" in r.suggestion


# ── ArgumentSchemaCritic ────────────────────────────────────────────────────────

class TestArgumentSchemaCritic:
    critic = ArgumentSchemaCritic()

    # ── PASS cases ──────────────────────────────────────────────────────────────

    def test_pass_only_required_arg(self):
        plan = make_plan(("get_weather", {"location": "NYC"}))
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "pass"

    def test_pass_required_plus_optional(self):
        plan = make_plan(("get_weather", {"location": "Paris", "days": 3}))
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "pass"

    def test_pass_var_refs_skip_type_check(self):
        plan = make_plan(("get_weather", {"location": "$city", "days": "$n"}))
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "pass"

    def test_pass_unknown_function_skipped(self):
        # FunctionValidityCritic owns this; ArgumentSchemaCritic must not double-fire
        plan = make_plan(("nonexistent_fn", {"x": 1}))
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "pass"

    def test_pass_all_required_email_args(self):
        plan = make_plan(("send_email", {"to": "a@b.com", "subject": "hi", "body": "text"}))
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "pass"

    # ── FAIL cases ──────────────────────────────────────────────────────────────

    def test_fail_missing_required_arg(self):
        plan = make_plan(("get_weather", {}))  # 'location' required
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "fail"
        assert "location" in r.error

    def test_fail_unknown_arg(self):
        plan = make_plan(("get_weather", {"location": "NYC", "bogus_param": "x"}))
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "fail"
        assert "bogus_param" in r.error

    def test_fail_wrong_type_string_for_integer(self):
        plan = make_plan(("get_weather", {"location": "NYC", "days": "seven"}))
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "fail"
        assert "days" in r.error

    def test_fail_bool_for_integer(self):
        # bool is a subclass of int in Python; critic must reject it for integer fields
        plan = make_plan(("get_weather", {"location": "NYC", "days": True}))
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "fail"
        assert "days" in r.error
        assert "boolean" in r.error

    def test_fail_integer_for_string(self):
        plan = make_plan(("send_email", {"to": 12345, "subject": "hi", "body": "text"}))
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "fail"
        assert "to" in r.error

    def test_fail_step_index_reported(self):
        plan = make_plan(
            ("get_weather", {"location": "NYC"}),     # ok
            ("send_email",  {"to": "a@b.com"}),       # missing subject, body
        )
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "fail"
        assert r.step_index == 1


# ── ArgumentValueCritic ─────────────────────────────────────────────────────────

class TestArgumentValueCritic:
    critic = ArgumentValueCritic()

    # ── PASS cases ──────────────────────────────────────────────────────────────

    def test_pass_valid_enum_value(self):
        plan = make_plan(("get_weather", {"location": "NYC", "units": "celsius"}))
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "pass"

    def test_pass_valid_email_format(self):
        plan = make_plan(("send_email", {"to": "user@example.com", "subject": "Hi", "body": "Hey"}))
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "pass"

    def test_pass_valid_iso_date(self):
        plan = make_plan(("send_email", {
            "to": "a@b.com", "subject": "x", "body": "y", "date": "2025-07-14",
        }))
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "pass"

    def test_pass_numeric_at_minimum_boundary(self):
        plan = make_plan(("get_weather", {"location": "NYC", "days": 1}))
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "pass"

    def test_pass_numeric_at_maximum_boundary(self):
        plan = make_plan(("get_weather", {"location": "NYC", "days": 14}))
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "pass"

    def test_pass_pattern_matches(self):
        plan = make_plan(("search", {"query": "weather today in NYC"}))
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "pass"

    def test_pass_var_ref_skipped(self):
        plan = make_plan(("get_weather", {"location": "NYC", "units": "$unit_var"}))
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "pass"

    # ── FAIL cases ──────────────────────────────────────────────────────────────

    def test_fail_invalid_enum_value(self):
        plan = make_plan(("get_weather", {"location": "NYC", "units": "kelvin"}))
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "fail"
        assert "kelvin" in r.error
        assert r.step_index == 0

    def test_fail_invalid_email_format(self):
        plan = make_plan(("send_email", {"to": "not-an-email", "subject": "Hi", "body": "Hey"}))
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "fail"
        assert "not-an-email" in r.error

    def test_fail_invalid_date_wrong_separator(self):
        plan = make_plan(("send_email", {
            "to": "a@b.com", "subject": "x", "body": "y", "date": "07/14/2025",
        }))
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "fail"
        assert "date" in r.error.lower()

    def test_fail_numeric_below_minimum(self):
        plan = make_plan(("get_weather", {"location": "NYC", "days": 0}))
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "fail"
        assert "minimum" in r.error or "below" in r.error

    def test_fail_numeric_above_maximum(self):
        plan = make_plan(("get_weather", {"location": "NYC", "days": 99}))
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "fail"
        assert "maximum" in r.error or "above" in r.error

    def test_fail_pattern_too_short(self):
        plan = make_plan(("search", {"query": "ab"}))  # pattern requires 3+ chars
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "fail"
        assert "pattern" in r.error


# ── DependencyCritic ────────────────────────────────────────────────────────────

class TestDependencyCritic:
    critic = DependencyCritic()

    # ── PASS cases ──────────────────────────────────────────────────────────────

    def test_pass_no_var_refs(self):
        plan = make_plan(("get_weather", {"location": "NYC"}))
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "pass"

    def test_pass_valid_chain(self):
        plan = make_plan(
            ("get_weather", {"location": "NYC"}, "wx"),
            ("send_email",  {"to": "a@b.com", "subject": "$wx", "body": "FYI"}),
        )
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "pass"

    def test_pass_metadata_lists_defined_vars(self):
        plan = make_plan(
            ("get_weather", {"location": "NYC"}, "wx"),
            ("search",      {"query": "news"},   "results"),
        )
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "pass"
        assert "wx" in r.metadata["defined_vars"]
        assert "results" in r.metadata["defined_vars"]

    def test_pass_inline_string_interpolation(self):
        plan = make_plan(
            ("get_weather", {"location": "NYC"}, "temp"),
            ("search",      {"query": "NYC forecast $temp"}),
        )
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "pass"

    def test_pass_empty_plan_no_defined_vars(self):
        r = self.critic.evaluate(Plan(), ctx())
        assert r.status == "pass"
        assert r.metadata["defined_vars"] == []

    # ── FAIL cases ──────────────────────────────────────────────────────────────

    def test_fail_undefined_var_ref(self):
        plan = make_plan(("search", {"query": "$ghost_var"}))
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "fail"
        assert "ghost_var" in r.error

    def test_fail_forward_ref_not_yet_defined(self):
        # Step 0 uses $result which is only produced by step 1
        plan = make_plan(
            ("send_email", {"to": "a@b.com", "subject": "$result", "body": "x"}),
            ("get_weather", {"location": "NYC"}, "result"),
        )
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "fail"
        assert r.step_index == 0

    def test_fail_duplicate_output_var(self):
        plan = make_plan(
            ("get_weather", {"location": "NYC"}, "w"),
            ("get_weather", {"location": "LA"},  "w"),  # redefines 'w'
        )
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "fail"
        assert "w" in r.error

    def test_fail_ref_in_nested_list(self):
        # refs inside list arguments must also be validated
        plan = make_plan(("search", {"query": ["$missing"]}))
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "fail"
        assert "missing" in r.error


# ── ExecutionCritic ─────────────────────────────────────────────────────────────

class TestExecutionCritic:
    def _make_critic(self, handlers=None) -> ExecutionCritic:
        return ExecutionCritic(ToolExecutor(handlers or {}))

    # ── PASS cases ──────────────────────────────────────────────────────────────

    def test_pass_single_step_succeeds(self):
        critic = self._make_critic({"get_weather": lambda args: {"temp": 72}})
        plan = make_plan(("get_weather", {"location": "NYC"}))
        r = critic.evaluate(plan, ctx())
        assert r.status == "pass"
        assert r.metadata["trace"][0]["status"] == "ok"

    def test_pass_chained_steps_with_var_resolution(self):
        critic = self._make_critic({
            "get_weather": lambda args: "72F sunny",
            "send_email":  lambda args: {"sent": True},
        })
        plan = make_plan(
            ("get_weather", {"location": "NYC"}, "wx"),
            ("send_email",  {"to": "a@b.com", "subject": "$wx", "body": "see above"}),
        )
        r = critic.evaluate(plan, ctx())
        assert r.status == "pass"

    def test_pass_empty_plan(self):
        critic = self._make_critic()
        r = critic.evaluate(Plan(), ctx())
        assert r.status == "pass"

    # ── FAIL cases ──────────────────────────────────────────────────────────────

    def test_fail_no_handler_registered(self):
        critic = self._make_critic({})
        plan = make_plan(("get_weather", {"location": "NYC"}))
        r = critic.evaluate(plan, ctx())
        assert r.status == "fail"
        assert "get_weather" in r.error

    def test_fail_handler_raises_exception(self):
        critic = self._make_critic({"get_weather": lambda args: (_ for _ in ()).throw(ValueError("API down"))})
        plan = make_plan(("get_weather", {"location": "NYC"}))
        r = critic.evaluate(plan, ctx())
        assert r.status == "fail"
        assert "API down" in r.error

    def test_fail_second_step_error_reports_correct_index(self):
        calls = {"n": 0}
        def flaky(args):
            calls["n"] += 1
            if calls["n"] > 1:
                raise RuntimeError("quota exceeded")
            return "ok"
        critic = self._make_critic({"get_weather": flaky, "search": flaky})
        plan = make_plan(
            ("get_weather", {"location": "NYC"}),
            ("search",      {"query": "news"}),
        )
        r = critic.evaluate(plan, ctx())
        assert r.status == "fail"
        assert r.step_index == 1

    def test_fail_trace_attached_to_metadata(self):
        critic = self._make_critic({})
        plan = make_plan(("get_weather", {"location": "NYC"}))
        r = critic.evaluate(plan, ctx())
        assert r.status == "fail"
        assert "trace" in r.metadata


# ── RedundancyCritic ────────────────────────────────────────────────────────────

class TestRedundancyCritic:
    critic = RedundancyCritic()

    # ── PASS cases ──────────────────────────────────────────────────────────────

    def test_pass_empty_plan(self):
        r = self.critic.evaluate(Plan(), ctx())
        assert r.status == "pass"

    def test_pass_single_step(self):
        plan = make_plan(("search", {"query": "hello world"}))
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "pass"

    def test_pass_same_function_different_args(self):
        plan = make_plan(
            ("get_weather", {"location": "NYC"}),
            ("get_weather", {"location": "LA"}),
        )
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "pass"

    def test_pass_three_unique_steps(self):
        plan = make_plan(
            ("get_weather", {"location": "NYC"}),
            ("send_email",  {"to": "a@b.com", "subject": "wx", "body": "ok"}),
            ("search",      {"query": "news today"}),
        )
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "pass"

    # ── FAIL cases ──────────────────────────────────────────────────────────────

    def test_fail_adjacent_duplicate(self):
        plan = make_plan(
            ("get_weather", {"location": "NYC"}),
            ("get_weather", {"location": "NYC"}),
        )
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "fail"
        assert r.step_index == 1

    def test_fail_non_adjacent_duplicate(self):
        plan = make_plan(
            ("search",      {"query": "news"}),
            ("get_weather", {"location": "NYC"}),
            ("search",      {"query": "news"}),
        )
        r = self.critic.evaluate(plan, ctx())
        assert r.status == "fail"
        assert r.step_index == 2

    def test_fail_error_mentions_original_step(self):
        plan = make_plan(
            ("get_weather", {"location": "NYC"}),
            ("get_weather", {"location": "NYC"}),
        )
        r = self.critic.evaluate(plan, ctx())
        assert "step 0" in r.error


# ── ProposalParser ──────────────────────────────────────────────────────────────

class TestProposalParser:
    parser = ProposalParser()

    # ── PASS cases ──────────────────────────────────────────────────────────────

    def test_pass_valid_json_single_step(self):
        raw = '{"steps": [{"function": "get_weather", "arguments": {"location": "NYC"}}]}'
        plan, err = self.parser.parse(raw)
        assert err is None and plan is not None
        assert len(plan.steps) == 1
        assert plan.steps[0].function == "get_weather"
        assert plan.steps[0].arguments == {"location": "NYC"}
        assert plan.steps[0].output_var is None

    def test_pass_json_in_markdown_fence(self):
        raw = '```json\n{"steps": [{"function": "search", "arguments": {"query": "news"}}]}\n```'
        plan, err = self.parser.parse(raw)
        assert err is None
        assert plan.steps[0].function == "search"

    def test_pass_output_var_preserved(self):
        raw = '{"steps": [{"function": "get_weather", "arguments": {}, "output_var": "wx"}]}'
        plan, err = self.parser.parse(raw)
        assert err is None
        assert plan.steps[0].output_var == "wx"

    def test_pass_empty_steps_list(self):
        plan, err = self.parser.parse('{"steps": []}')
        assert err is None
        assert len(plan.steps) == 0

    def test_pass_multi_step_plan(self):
        raw = '{"steps": [{"function": "a", "arguments": {}}, {"function": "b", "arguments": {}}]}'
        plan, err = self.parser.parse(raw)
        assert err is None
        assert len(plan.steps) == 2

    # ── FAIL cases ──────────────────────────────────────────────────────────────

    def test_fail_not_json(self):
        plan, err = self.parser.parse("definitely not json }{")
        assert plan is None and err is not None
        assert "JSON" in err

    def test_fail_top_level_is_list(self):
        plan, err = self.parser.parse('[{"function": "fn", "arguments": {}}]')
        assert plan is None
        assert "steps" in err

    def test_fail_steps_field_missing(self):
        plan, err = self.parser.parse('{"result": []}')
        assert plan is None
        assert "steps" in err

    def test_fail_steps_not_a_list(self):
        plan, err = self.parser.parse('{"steps": "oops"}')
        assert plan is None
        assert "steps" in err

    def test_fail_step_missing_function_field(self):
        plan, err = self.parser.parse('{"steps": [{"arguments": {"x": 1}}]}')
        assert plan is None
        assert "function" in err

    def test_fail_step_function_not_string(self):
        plan, err = self.parser.parse('{"steps": [{"function": 42, "arguments": {}}]}')
        assert plan is None
        assert "function" in err

    def test_fail_arguments_not_dict(self):
        plan, err = self.parser.parse('{"steps": [{"function": "fn", "arguments": ["a", "b"]}]}')
        assert plan is None
        assert "arguments" in err


# ── ToolExecutor ────────────────────────────────────────────────────────────────

class TestToolExecutor:

    def test_simple_execution_returns_output(self):
        ex = ToolExecutor({"double": lambda args: args["x"] * 2})
        plan = make_plan(("double", {"x": 5}))
        trace = ex.execute_plan(plan)
        assert trace[0]["status"] == "ok"
        assert trace[0]["output"] == 10

    def test_variable_full_ref_resolved(self):
        ex = ToolExecutor({
            "produce": lambda args: 42,
            "consume": lambda args: args["value"] + 1,
        })
        plan = make_plan(
            ("produce", {}, "result"),
            ("consume", {"value": "$result"}),
        )
        trace = ex.execute_plan(plan)
        assert trace[1]["status"] == "ok"
        assert trace[1]["output"] == 43

    def test_variable_inline_string_interpolation(self):
        ex = ToolExecutor({
            "greet":  lambda args: "Alice",
            "format": lambda args: args["msg"],
        })
        plan = make_plan(
            ("greet",  {}, "name"),
            ("format", {"msg": "Hello $name!"}),
        )
        trace = ex.execute_plan(plan)
        assert trace[1]["status"] == "ok"
        assert trace[1]["output"] == "Hello Alice!"

    def test_unresolved_variable_records_error_status(self):
        ex = ToolExecutor({"fn": lambda args: args})
        plan = make_plan(("fn", {"x": "$no_such_var"}))
        trace = ex.execute_plan(plan)
        assert trace[0]["status"] == "error"
        assert "no_such_var" in trace[0]["error"]

    def test_missing_handler_returns_error_status(self):
        ex = ToolExecutor({})
        trace = ex.execute_plan(make_plan(("unknown_fn", {})))
        assert trace[0]["status"] == "error"
        assert "unknown_fn" in trace[0]["error"]

    def test_handler_exception_captured_not_raised(self):
        ex = ToolExecutor({"boom": lambda args: 1 / 0})
        trace = ex.execute_plan(make_plan(("boom", {})))
        assert trace[0]["status"] == "error"
        assert "ZeroDivision" in trace[0]["error"]

    def test_list_argument_resolved(self):
        ex = ToolExecutor({
            "first":  lambda args: "item_A",
            "second": lambda args: args["items"],
        })
        plan = make_plan(
            ("first",  {}, "a"),
            ("second", {"items": ["$a", "item_B"]}),
        )
        trace = ex.execute_plan(plan)
        assert trace[1]["status"] == "ok"
        assert trace[1]["output"] == ["item_A", "item_B"]

    def test_trace_includes_function_and_arguments(self):
        ex = ToolExecutor({"fn": lambda args: None})
        trace = ex.execute_plan(make_plan(("fn", {"k": "v"})))
        assert trace[0]["function"] == "fn"
        assert trace[0]["arguments"] == {"k": "v"}


# ── MetaController ──────────────────────────────────────────────────────────────

class TestMetaController:
    mc = MetaController()

    def test_initial_prompt_contains_query(self):
        prompt = self.mc.build_initial_prompt("What is the weather?", SCHEMAS)
        assert "What is the weather?" in prompt

    def test_initial_prompt_contains_all_function_names(self):
        prompt = self.mc.build_initial_prompt("test", SCHEMAS)
        assert "get_weather" in prompt
        assert "send_email" in prompt
        assert "search" in prompt

    def test_aggregate_no_failures_empty_instructions(self):
        results = [CriticResult(critic_name="FunctionValidityCritic", status="pass")]
        fb = self.mc.aggregate(results)
        assert len(fb.failures) == 0
        assert "0 validation" in fb.summary

    def test_aggregate_hard_critics_sorted_before_soft(self):
        results = [
            CriticResult(critic_name="RedundancyCritic",       status="fail", step_index=0, error="dup"),
            CriticResult(critic_name="FunctionValidityCritic", status="fail", step_index=1, error="bad fn"),
        ]
        fb = self.mc.aggregate(results)
        assert fb.failures[0].critic_name == "FunctionValidityCritic"
        assert fb.failures[1].critic_name == "RedundancyCritic"

    def test_aggregate_summary_counts_hard_and_soft(self):
        results = [
            CriticResult(critic_name="FunctionValidityCritic", status="fail", error="bad fn"),
            CriticResult(critic_name="RedundancyCritic",       status="fail", error="dup"),
        ]
        fb = self.mc.aggregate(results)
        assert "1 hard" in fb.summary
        assert "1 soft" in fb.summary

    def test_aggregate_instructions_include_suggestion(self):
        results = [
            CriticResult(critic_name="FunctionValidityCritic", status="fail",
                         error="fn missing", suggestion="use get_weather"),
        ]
        fb = self.mc.aggregate(results)
        assert "use get_weather" in fb.instructions[0]

    def test_backprompt_contains_error_details(self):
        results = [CriticResult(
            critic_name="FunctionValidityCritic", status="fail", step_index=0,
            error="function 'fly' not in set", suggestion="use get_weather",
        )]
        fb = self.mc.aggregate(results)
        prompt = self.mc.build_backprompt("test", SCHEMAS, '{"steps":[]}', fb, [])
        assert "fly" in prompt
        assert "get_weather" in prompt

    def test_backprompt_includes_original_query(self):
        fb = self.mc.aggregate([])
        prompt = self.mc.build_backprompt("Find the weather please", SCHEMAS, "{}", fb, [])
        assert "Find the weather please" in prompt


# ── BFCLAdapter ─────────────────────────────────────────────────────────────────

class TestBFCLAdapter:
    def _schema(self, name="fn"):
        return {"name": name, "description": "", "parameters": {"type": "object", "properties": {}, "required": []}}

    def test_query_from_question_list_of_lists(self):
        task = {
            "question": [[{"role": "user", "content": "What is the weather?"}]],
            "function": [self._schema("get_weather")],
        }
        query, schemas = load_bfcl_task(task)
        assert "weather" in query
        assert schemas[0]["name"] == "get_weather"

    def test_query_from_prompt_string(self):
        task = {"prompt": "Search for news", "function": [self._schema()]}
        query, _ = load_bfcl_task(task)
        assert query == "Search for news"

    def test_query_from_query_string(self):
        task = {"query": "Direct query", "function": [self._schema()]}
        query, _ = load_bfcl_task(task)
        assert query == "Direct query"

    def test_schemas_from_function_key(self):
        task = {"query": "test", "function": [self._schema("fn_a")]}
        _, schemas = load_bfcl_task(task)
        assert schemas[0]["name"] == "fn_a"

    def test_schemas_from_functions_key(self):
        task = {"query": "test", "functions": [self._schema("fn_b")]}
        _, schemas = load_bfcl_task(task)
        assert schemas[0]["name"] == "fn_b"

    def test_schemas_from_tools_openai_wrapper(self):
        task = {
            "query": "test",
            "tools": [{"type": "function", "function": {"name": "fn_c", "description": "", "parameters": {}}}],
        }
        _, schemas = load_bfcl_task(task)
        assert schemas[0]["name"] == "fn_c"

    def test_raises_on_missing_function_schemas(self):
        with pytest.raises(ValueError, match="function schemas"):
            load_bfcl_task({"query": "test"})

    def test_raises_on_missing_query(self):
        with pytest.raises(ValueError, match="query"):
            load_bfcl_task({"function": [self._schema()]})

    def test_multi_turn_question_concatenates_content(self):
        task = {
            "question": [[
                {"role": "user",      "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user",      "content": "What is 2+2?"},
            ]],
            "function": [self._schema()],
        }
        query, _ = load_bfcl_task(task)
        assert "Hello" in query
        assert "What is 2+2?" in query
