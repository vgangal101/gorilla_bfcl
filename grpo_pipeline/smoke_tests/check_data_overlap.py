"""
check_data_overlap.py — Thorough overlap analysis between training (non-live)
and evaluation (live) BFCL categories.

Checks:
  1. Question text — exact string match across all category pairs
  2. Function names — same function name appearing in both splits
  3. Full function schemas — same complete schema (name + description + params)
  4. Parameter structure — same function name + same parameter names/types

Run from repo root:
    python grpo_pipeline/smoke_tests/check_data_overlap.py
"""

import json
from pathlib import Path
from itertools import product

BFCL_DATA = Path("berkeley-function-call-leaderboard/bfcl_eval/data")

TRAIN_CATEGORIES = [
    "simple_python",
    "simple_java",
    "simple_javascript",
    "parallel",
    "multiple",
]

EVAL_CATEGORIES = [
    "live_simple",
    "live_parallel",
    "live_multiple",
    "live_parallel_multiple",
]


def load_category(category: str) -> list[dict]:
    path = BFCL_DATA / f"BFCL_v4_{category}.json"
    if not path.exists():
        print(f"  [SKIP] {category} — file not found")
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def get_questions(entries: list[dict]) -> set[str]:
    return {e["question"][0][0]["content"] for e in entries}


def get_function_names(entries: list[dict]) -> set[str]:
    names = set()
    for e in entries:
        for fn in e.get("function", []):
            names.add(fn["name"])
    return names


def normalize_schema(fn: dict) -> str:
    """Canonical JSON string of a function schema for exact comparison."""
    return json.dumps(fn, sort_keys=True)


def get_schemas(entries: list[dict]) -> set[str]:
    schemas = set()
    for e in entries:
        for fn in e.get("function", []):
            schemas.add(normalize_schema(fn))
    return schemas


def get_param_fingerprints(entries: list[dict]) -> set[str]:
    """Function name + sorted parameter names + types — catches same API with
    different description wording."""
    fps = set()
    for e in entries:
        for fn in e.get("function", []):
            params = fn.get("parameters", {}).get("properties", {})
            param_sig = sorted(
                f"{k}:{v.get('type','?')}" for k, v in params.items()
            )
            fps.add(f"{fn['name']}({','.join(param_sig)})")
    return fps


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def report_overlap(label: str, set_a: set, set_b: set, cat_a: str, cat_b: str):
    overlap = set_a & set_b
    print(f"  {cat_a} vs {cat_b}:  {len(overlap)} overlapping {label}")
    if overlap:
        for item in sorted(overlap)[:5]:
            print(f"    • {item[:120]}")
        if len(overlap) > 5:
            print(f"    ... and {len(overlap)-5} more")


def main():
    print("Loading categories...")
    train_data = {cat: load_category(cat) for cat in TRAIN_CATEGORIES}
    eval_data  = {cat: load_category(cat) for cat in EVAL_CATEGORIES}

    train_data = {k: v for k, v in train_data.items() if v}
    eval_data  = {k: v for k, v in eval_data.items() if v}

    print(f"  Train: {', '.join(f'{k}({len(v)})' for k,v in train_data.items())}")
    print(f"  Eval:  {', '.join(f'{k}({len(v)})' for k,v in eval_data.items())}")

    # ── 1. Question text overlap ─────────────────────────────────────────
    section("1. Question Text Overlap (exact string match)")
    any_overlap = False
    for tc, ec in product(train_data, eval_data):
        tq = get_questions(train_data[tc])
        eq = get_questions(eval_data[ec])
        overlap = tq & eq
        if overlap:
            report_overlap("questions", tq, eq, tc, ec)
            any_overlap = True
    if not any_overlap:
        print("  CLEAN — no question text appears in both train and eval splits")

    # ── 2. Function name overlap ─────────────────────────────────────────
    section("2. Function Name Overlap")
    all_train_names = set()
    all_eval_names  = set()
    for v in train_data.values():
        all_train_names |= get_function_names(v)
    for v in eval_data.values():
        all_eval_names |= get_function_names(v)

    name_overlap = all_train_names & all_eval_names
    print(f"  Unique function names in train: {len(all_train_names)}")
    print(f"  Unique function names in eval:  {len(all_eval_names)}")
    print(f"  Overlapping names:              {len(name_overlap)}")
    if name_overlap:
        print("  Overlapping names (first 10):")
        for n in sorted(name_overlap)[:10]:
            print(f"    • {n}")
        if len(name_overlap) > 10:
            print(f"    ... and {len(name_overlap)-10} more")
    else:
        print("  CLEAN — no function names shared between train and eval")

    # ── 3. Full schema overlap (exact) ───────────────────────────────────
    section("3. Full Schema Overlap (exact: name + description + parameters)")
    all_train_schemas = set()
    all_eval_schemas  = set()
    for v in train_data.values():
        all_train_schemas |= get_schemas(v)
    for v in eval_data.values():
        all_eval_schemas |= get_schemas(v)

    schema_overlap = all_train_schemas & all_eval_schemas
    print(f"  Unique schemas in train: {len(all_train_schemas)}")
    print(f"  Unique schemas in eval:  {len(all_eval_schemas)}")
    print(f"  Exact schema overlap:    {len(schema_overlap)}")
    if schema_overlap:
        print("  Overlapping schemas (first 3):")
        for s in sorted(schema_overlap)[:3]:
            fn = json.loads(s)
            print(f"    • {fn['name']} — {fn.get('description','')[:80]}")
    else:
        print("  CLEAN — no exact schema match between train and eval")

    # ── 4. Parameter fingerprint overlap ─────────────────────────────────
    section("4. Parameter Structure Overlap (name + param names/types)")
    all_train_fps = set()
    all_eval_fps  = set()
    for v in train_data.values():
        all_train_fps |= get_param_fingerprints(v)
    for v in eval_data.values():
        all_eval_fps |= get_param_fingerprints(v)

    fp_overlap = all_train_fps & all_eval_fps
    print(f"  Unique param fingerprints in train: {len(all_train_fps)}")
    print(f"  Unique param fingerprints in eval:  {len(all_eval_fps)}")
    print(f"  Structural overlap:                 {len(fp_overlap)}")
    if fp_overlap:
        print("  Overlapping fingerprints (first 10):")
        for fp in sorted(fp_overlap)[:10]:
            print(f"    • {fp}")
        if len(fp_overlap) > 10:
            print(f"    ... and {len(fp_overlap)-10} more")
    else:
        print("  CLEAN — no function structures shared between train and eval")

    # ── Summary ──────────────────────────────────────────────────────────
    section("SUMMARY")
    results = {
        "Question text":       len(
            set.union(*(get_questions(v) for v in train_data.values())) &
            set.union(*(get_questions(v) for v in eval_data.values()))
        ),
        "Function names":      len(name_overlap),
        "Exact schemas":       len(schema_overlap),
        "Param fingerprints":  len(fp_overlap),
    }
    for check, count in results.items():
        status = "CLEAN ✓" if count == 0 else f"OVERLAP — {count} shared"
        print(f"  {check:<25} {status}")

    if all(v == 0 for v in results.values()):
        print("\n  Train/eval split is clean on all checks.")
        print("  Safe to use non-live categories for training and live for evaluation.")
    else:
        print("\n  Overlap detected — review before using this split.")


if __name__ == "__main__":
    main()
