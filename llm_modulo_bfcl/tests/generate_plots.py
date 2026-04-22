#!/usr/bin/env python3
"""Generate diagnostic plots for the LLM-Modulo critic bank.

Run from repo root:
    python llm_modulo_bfcl/tests/generate_plots.py

Produces three PNGs in llm_modulo_bfcl/tests/plots/:
  1. critic_test_outcomes.png  — pass/fail matrix per critic per scenario
  2. critic_summary.png        — total pass vs. fail counts per critic
  3. bfcl_categories.png       — BFCL test category breakdown by group
"""

import sys
import pathlib

ROOT = pathlib.Path(__file__).parent.parent   # llm_modulo_bfcl/
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")  # headless — save to files, never open a window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from schemas import FunctionCall, Plan, CriticResult
from critics.function_validity import FunctionValidityCritic
from critics.argument_schema import ArgumentSchemaCritic
from critics.argument_value import ArgumentValueCritic
from critics.dependency import DependencyCritic
from critics.execution import ExecutionCritic
from critics.redundancy import RedundancyCritic
from execution.executor import ToolExecutor

PLOTS_DIR = pathlib.Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── BFCL category taxonomy ───────────────────────────────────────────────────────

CATEGORIES = {
    "Non-Live (single-turn)": [
        "simple_python",
        "simple_java",
        "simple_javascript",
        "multiple",
        "parallel",
        "parallel_multiple",
        "irrelevance",
    ],
    "Live (single-turn)": [
        "live_simple",
        "live_multiple",
        "live_parallel",
        "live_parallel_multiple",
        "live_irrelevance",
        "live_relevance",
    ],
    "Multi-Turn": [
        "multi_turn_base",
        "multi_turn_miss_func",
        "multi_turn_miss_param",
        "multi_turn_long_context",
    ],
    "Memory (Agentic)": [
        "memory_kv",
        "memory_vector",
        "memory_rec_sum",
    ],
    "Web Search (Agentic)": [
        "web_search_base",
        "web_search_no_snippet",
    ],
    "Non-Scoring": [
        "format_sensitivity",
    ],
}

GROUP_COLORS = {
    "Non-Live (single-turn)": "#4C72B0",
    "Live (single-turn)":     "#DD8452",
    "Multi-Turn":             "#55A868",
    "Memory (Agentic)":       "#C44E52",
    "Web Search (Agentic)":   "#8172B2",
    "Non-Scoring":            "#937860",
}

# ── Shared test fixtures ─────────────────────────────────────────────────────────

SCHEMAS = [
    {
        "name": "get_weather",
        "description": "Get weather for a location.",
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
        "description": "Send an email.",
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


def p(*steps) -> Plan:
    plan = Plan()
    for step in steps:
        fn, args = step[0], step[1]
        out = step[2] if len(step) > 2 else None
        plan.steps.append(FunctionCall(function=fn, arguments=args, output_var=out))
    return plan


def ctx() -> dict:
    return {"query": "unit-test", "function_schemas": SCHEMAS, "history": []}


# ── Test scenario definitions ────────────────────────────────────────────────────
# Each entry: (label, plan, context, expected_status)

def build_scenarios():
    # ExecutionCritic needs handlers
    ok_handlers = {
        "get_weather": lambda args: {"temp": 72},
        "send_email":  lambda args: {"sent": True},
        "search":      lambda args: {"hits": []},
    }
    bad_exec = ExecutionCritic(ToolExecutor({}))            # empty handlers → always fails
    good_exec = ExecutionCritic(ToolExecutor(ok_handlers))  # real handlers → succeeds

    fv  = FunctionValidityCritic()
    asc = ArgumentSchemaCritic()
    avc = ArgumentValueCritic()
    dep = DependencyCritic()
    red = RedundancyCritic()

    scenarios = {
        "FunctionValidity": [
            ("known fn",          fv, p(("get_weather", {"location": "NYC"})),        ctx(), "pass"),
            ("empty plan",        fv, Plan(),                                          ctx(), "pass"),
            ("multi-step valid",  fv, p(("get_weather",{}),("send_email",{})),        ctx(), "pass"),
            ("unknown fn@step0",  fv, p(("fly_to_moon", {})),                         ctx(), "fail"),
            ("unknown fn@step1",  fv, p(("get_weather",{}),("launch_rocket",{})),     ctx(), "fail"),
            ("empty schema set",  fv, p(("get_weather", {"location":"NYC"})),
             {"query":"t","function_schemas":[],"history":[]},                              "fail"),
        ],
        "ArgumentSchema": [
            ("required only",     asc, p(("get_weather", {"location":"NYC"})),        ctx(), "pass"),
            ("req+optional",      asc, p(("get_weather", {"location":"X","days":3})), ctx(), "pass"),
            ("$var skips type",   asc, p(("get_weather", {"location":"$c","days":"$n"})), ctx(), "pass"),
            ("missing required",  asc, p(("get_weather", {})),                        ctx(), "fail"),
            ("unknown arg",       asc, p(("get_weather", {"location":"X","bogus":1})),ctx(), "fail"),
            ("str for int",       asc, p(("get_weather", {"location":"X","days":"7"})),ctx(),"fail"),
            ("bool for int",      asc, p(("get_weather", {"location":"X","days":True})),ctx(),"fail"),
        ],
        "ArgumentValue": [
            ("valid enum",        avc, p(("get_weather",{"location":"X","units":"celsius"})),ctx(),"pass"),
            ("valid email",       avc, p(("send_email",{"to":"u@ex.com","subject":"s","body":"b"})),ctx(),"pass"),
            ("valid ISO date",    avc, p(("send_email",{"to":"a@b.com","subject":"s","body":"b","date":"2025-07-14"})),ctx(),"pass"),
            ("in-range int",      avc, p(("get_weather",{"location":"X","days":7})),  ctx(), "pass"),
            ("$var skipped",      avc, p(("get_weather",{"location":"X","units":"$u"})),ctx(),"pass"),
            ("invalid enum",      avc, p(("get_weather",{"location":"X","units":"kelvin"})),ctx(),"fail"),
            ("bad email",         avc, p(("send_email",{"to":"not-email","subject":"s","body":"b"})),ctx(),"fail"),
            ("bad date format",   avc, p(("send_email",{"to":"a@b.com","subject":"s","body":"b","date":"07/14/2025"})),ctx(),"fail"),
            ("below minimum",     avc, p(("get_weather",{"location":"X","days":0})),  ctx(), "fail"),
            ("above maximum",     avc, p(("get_weather",{"location":"X","days":99})), ctx(), "fail"),
            ("pattern fail",      avc, p(("search",{"query":"ab"})),                  ctx(), "fail"),
        ],
        "Dependency": [
            ("no refs",           dep, p(("get_weather",{"location":"NYC"})),          ctx(), "pass"),
            ("valid chain",       dep, p(("get_weather",{"location":"NYC"},"wx"),
                                        ("send_email",{"to":"a@b.com","subject":"$wx","body":"x"})), ctx(), "pass"),
            ("inline interp",     dep, p(("get_weather",{"location":"NYC"},"t"),
                                        ("search",{"query":"NYC wx $t"})),             ctx(), "pass"),
            ("empty plan",        dep, Plan(),                                          ctx(), "pass"),
            ("undefined ref",     dep, p(("search",{"query":"$ghost"})),               ctx(), "fail"),
            ("forward ref",       dep, p(("send_email",{"to":"a@b.com","subject":"$r","body":"x"}),
                                        ("get_weather",{"location":"NYC"},"r")),       ctx(), "fail"),
            ("duplicate output",  dep, p(("get_weather",{"location":"NYC"},"w"),
                                        ("get_weather",{"location":"LA"},"w")),        ctx(), "fail"),
        ],
        "Execution": [
            ("handler succeeds",  good_exec, p(("get_weather",{"location":"NYC"})),   ctx(), "pass"),
            ("chained success",   good_exec, p(("get_weather",{"location":"NYC"},"wx"),
                                              ("send_email",{"to":"a@b.com","subject":"$wx","body":"x"})), ctx(), "pass"),
            ("empty plan ok",     good_exec, Plan(),                                   ctx(), "pass"),
            ("no handler",        bad_exec,  p(("get_weather",{"location":"NYC"})),   ctx(), "fail"),
            ("handler raises",    ExecutionCritic(ToolExecutor({"get_weather": lambda a: 1/0})),
                                  p(("get_weather",{"location":"NYC"})),               ctx(), "fail"),
        ],
        "Redundancy": [
            ("empty plan",        red, Plan(),                                          ctx(), "pass"),
            ("unique steps",      red, p(("get_weather",{"location":"NYC"}),
                                        ("get_weather",{"location":"LA"})),            ctx(), "pass"),
            ("3 unique steps",    red, p(("get_weather",{"location":"NYC"}),
                                        ("send_email",{"to":"a@b.com","subject":"s","body":"b"}),
                                        ("search",{"query":"news"})),                  ctx(), "pass"),
            ("adjacent dup",      red, p(("get_weather",{"location":"NYC"}),
                                        ("get_weather",{"location":"NYC"})),           ctx(), "fail"),
            ("non-adjacent dup",  red, p(("search",{"query":"news"}),
                                        ("get_weather",{"location":"NYC"}),
                                        ("search",{"query":"news"})),                  ctx(), "fail"),
        ],
    }
    return scenarios


# ── Run scenarios and collect results ────────────────────────────────────────────

def run_scenarios(scenarios):
    """Return results dict: {critic_name: [(label, expected, actual, ok)]}"""
    results = {}
    for critic_name, cases in scenarios.items():
        rows = []
        for label, critic, plan, context, expected in cases:
            actual_result = critic.evaluate(plan, context)
            actual = actual_result.status
            ok = (actual == expected)
            rows.append((label, expected, actual, ok))
        results[critic_name] = rows
    return results


# ── Plot 1: Critic test outcome matrix ──────────────────────────────────────────

def plot_outcome_matrix(results):
    critic_names = list(results.keys())
    max_cases = max(len(v) for v in results.values())

    # Build 2D grid: rows=critics, cols=test_cases
    grid = np.full((len(critic_names), max_cases), np.nan)
    labels_grid = [[""] * max_cases for _ in critic_names]

    for r, cname in enumerate(critic_names):
        for c, (label, expected, actual, ok) in enumerate(results[cname]):
            grid[r, c] = 1.0 if ok else 0.0
            short = label[:18] + "…" if len(label) > 18 else label
            labels_grid[r][c] = short

    fig, ax = plt.subplots(figsize=(max(14, max_cases * 1.3), len(critic_names) * 0.85 + 1.5))

    cmap = plt.cm.RdYlGn
    cmap.set_bad(color="#f0f0f0")
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_yticks(range(len(critic_names)))
    ax.set_yticklabels(critic_names, fontsize=11)
    ax.set_xticks([])
    ax.set_xlabel("Test scenarios (left→right per critic)", fontsize=10)

    # Cell annotations
    for r in range(len(critic_names)):
        for c in range(max_cases):
            if not np.isnan(grid[r, c]):
                lbl = labels_grid[r][c]
                passed = grid[r, c] == 1.0
                col = "white" if not passed else "black"
                ax.text(c, r, lbl, ha="center", va="center",
                        fontsize=6.5, color=col, wrap=False)

    pass_patch = mpatches.Patch(color=cmap(1.0), label="Test passed (actual == expected)")
    fail_patch = mpatches.Patch(color=cmap(0.0), label="Test failed (actual ≠ expected)")
    grey_patch  = mpatches.Patch(color="#f0f0f0",  label="No test case")
    ax.legend(handles=[pass_patch, fail_patch, grey_patch],
              loc="upper right", bbox_to_anchor=(1.0, -0.05), ncol=3, fontsize=9)

    ax.set_title("LLM-Modulo Critic Bank — Test Outcome Matrix", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    out = PLOTS_DIR / "critic_test_outcomes.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")


# ── Plot 2: Per-critic pass/fail bar chart ───────────────────────────────────────

def plot_critic_summary(results):
    critic_names = list(results.keys())
    pass_counts = [sum(1 for _, _, _, ok in v if ok)     for v in results.values()]
    fail_counts = [sum(1 for _, _, _, ok in v if not ok) for v in results.values()]

    x = np.arange(len(critic_names))
    w = 0.38

    fig, ax = plt.subplots(figsize=(11, 5))
    bars_p = ax.bar(x - w / 2, pass_counts, w, label="Tests passed", color="#4CAF50", edgecolor="white")
    bars_f = ax.bar(x + w / 2, fail_counts, w, label="Tests failed", color="#F44336", edgecolor="white")

    for bar in bars_p:
        h = bar.get_height()
        if h:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05, str(int(h)),
                    ha="center", va="bottom", fontsize=10, fontweight="bold", color="#2E7D32")
    for bar in bars_f:
        h = bar.get_height()
        if h:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05, str(int(h)),
                    ha="center", va="bottom", fontsize=10, fontweight="bold", color="#C62828")

    ax.set_xticks(x)
    ax.set_xticklabels(critic_names, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Number of test cases", fontsize=11)
    ax.set_title("Critic Bank — Test Pass / Fail Counts", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(max(pass_counts), max(fail_counts)) + 1.5)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = PLOTS_DIR / "critic_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")


# ── Plot 3: BFCL category breakdown ─────────────────────────────────────────────

def plot_bfcl_categories():
    groups = list(CATEGORIES.keys())
    counts = [len(v) for v in CATEGORIES.values()]
    colors = [GROUP_COLORS[g] for g in groups]

    # Horizontal bar chart
    fig, (ax_bar, ax_table) = plt.subplots(
        1, 2, figsize=(16, 6),
        gridspec_kw={"width_ratios": [1, 2]}
    )

    y = np.arange(len(groups))
    bars = ax_bar.barh(y, counts, color=colors, edgecolor="white", height=0.6)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(groups, fontsize=10)
    ax_bar.set_xlabel("Number of categories", fontsize=10)
    ax_bar.set_title("BFCL Category Groups", fontsize=12, fontweight="bold")
    ax_bar.invert_yaxis()
    for bar, cnt in zip(bars, counts):
        ax_bar.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                    str(cnt), va="center", fontsize=11, fontweight="bold")
    ax_bar.set_xlim(0, max(counts) + 1.5)
    ax_bar.grid(axis="x", alpha=0.3)

    # Right panel: category listing table
    ax_table.axis("off")
    col_x = {"Group": 0.0, "Category": 0.38, "Scoring": 0.82}
    non_scoring = {"format_sensitivity"}
    ax_table.text(col_x["Group"],    1.02, "Group",    transform=ax_table.transAxes,
                  fontsize=10, fontweight="bold", va="top")
    ax_table.text(col_x["Category"], 1.02, "Category", transform=ax_table.transAxes,
                  fontsize=10, fontweight="bold", va="top")
    ax_table.text(col_x["Scoring"],  1.02, "Scoring",  transform=ax_table.transAxes,
                  fontsize=10, fontweight="bold", va="top")

    all_cats = [(g, c) for g, cats in CATEGORIES.items() for c in cats]
    total = len(all_cats)
    for i, (group, cat) in enumerate(all_cats):
        y_pos = 1.0 - (i + 1) / (total + 1)
        scoring = "N" if cat in non_scoring else "Y"
        color = GROUP_COLORS[group]
        ax_table.text(col_x["Group"],    y_pos, group,   transform=ax_table.transAxes,
                      fontsize=8, va="center", color=color)
        ax_table.text(col_x["Category"], y_pos, cat,     transform=ax_table.transAxes,
                      fontsize=8, va="center")
        ax_table.text(col_x["Scoring"],  y_pos, scoring, transform=ax_table.transAxes,
                      fontsize=8, va="center",
                      color="#2E7D32" if scoring == "Y" else "#C62828")

    ax_table.set_title(f"All {total} BFCL Categories", fontsize=12, fontweight="bold")
    fig.suptitle("BFCL Test Category Distribution", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = PLOTS_DIR / "bfcl_categories.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out}")


# ── Print category listing ───────────────────────────────────────────────────────

def print_category_listing():
    print("\n" + "=" * 65)
    print("  BFCL FULL CATEGORY LISTING (for LLM-Modulo critic design)")
    print("=" * 65)
    scoring_count = 0
    for group, cats in CATEGORIES.items():
        is_scoring = group != "Non-Scoring"
        tag = "[SCORING]" if is_scoring else "[non-scoring]"
        print(f"\n  {group}  {tag}  ({len(cats)} categories)")
        print("  " + "-" * 55)
        for cat in cats:
            scoring_count += 1 if is_scoring else 0
            print(f"    • {cat}")
    total = sum(len(v) for v in CATEGORIES.values())
    print(f"\n  Total: {total} categories  ({scoring_count} scoring, {total - scoring_count} non-scoring)")
    print("=" * 65)

    print("""
  CRITIC DESIGN NOTES BY CATEGORY GROUP
  ----------------------------------------
  Non-Live / Live (single-turn):
    - FunctionValidity, ArgumentSchema, ArgumentValue, Dependency,
      Execution, Redundancy  (all 6 current critics apply)
    - Live categories hit real endpoints; Execution critic needs
      real handlers or a sandboxed mock that mirrors the API

  Multi-Turn (4 categories):
    - Stateful conversation; need critics for:
        * TurnOrderCritic   -- each turn addresses the right context
        * StateConsistency  -- function outputs passed correctly between turns
        * MissFuncCritic    -- detects when a required fn call is omitted
        * MissParamCritic   -- detects when a required arg is left out
        * LongContextCritic -- plan stays coherent over many turns

  Memory/Agentic (3 categories):
    - memory_kv:      KV store correctness (set / get / overwrite)
    - memory_vector:  embedding recall precision (right doc retrieved)
    - memory_rec_sum: summarization coherence across recursive turns
    Critics to add: MemoryConsistencyCritic, RecallRelevanceCritic

  Web Search (2 categories -- currently SKIPPED, paid SerpAPI):
    - GroundingCritic  -- response grounded in search results
    - SnippetUseCritic -- correct use of snippet vs no-snippet mode
    (Skip until SerpAPI budget is available)

  Format Sensitivity (non-scoring):
    - Tests whether model is fragile to prompt phrasing
    - Useful for a RobustnessCritic if added later
""")


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    print("Running critic scenarios...")
    scenarios = build_scenarios()
    results   = run_scenarios(scenarios)

    total_cases = sum(len(v) for v in results.values())
    total_ok    = sum(sum(1 for _, _, _, ok in v if ok) for v in results.values())
    print(f"  {total_ok}/{total_cases} scenarios matched expected output\n")

    if total_ok < total_cases:
        print("  MISMATCHES:")
        for cname, rows in results.items():
            for label, expected, actual, ok in rows:
                if not ok:
                    print(f"    [{cname}] '{label}': expected={expected}, actual={actual}")
        print()

    print("Generating plots...")
    plot_outcome_matrix(results)
    plot_critic_summary(results)
    plot_bfcl_categories()

    print_category_listing()
    print(f"\nAll plots saved to: {PLOTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
