"""
data_prep.py — Extract (prompt, ground_truth) pairs from BFCL JSONL files
for SFT and GRPO training.

Run standalone to sanity-check:
    python grpo_pipeline/data_prep.py
"""

import json
from pathlib import Path

BFCL_DATA = Path("berkeley-function-call-leaderboard/bfcl_eval/data")

TRAIN_CATEGORIES = [
    "simple_python",
    "simple_java",
    "simple_javascript",
    "parallel",
    "multiple",
]

# Questions confirmed to appear in both training and evaluation splits.
# Identified by check_data_overlap.py — filtered from training to prevent
# contamination. The question remains in the eval set (live_multiple).
EXCLUDED_QUESTIONS: set[str] = {
    "Find out the rewards for playing Fortnite on Playstation platform with different missions and trophies",
}

SYSTEM_PROMPT = (
    "You are a function calling assistant. Given a user question and available "
    "functions, output the correct function call(s).\n\n"
    "Output format (strictly follow this):\n"
    "[func_name(param1=value1, param2=value2)]\n\n"
    "For multiple calls:\n"
    "[func1(a=1), func2(b=2)]\n\n"
    "If no function call is needed: [answer]\n"
    "No explanation, no markdown, just the bracketed call."
)


def load_category(category: str):
    data_path = BFCL_DATA / f"BFCL_v4_{category}.json"
    gt_path = BFCL_DATA / f"possible_answer/BFCL_v4_{category}.json"

    entries = [
        json.loads(line)
        for line in data_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gts = {
        json.loads(line)["id"]: json.loads(line)["ground_truth"]
        for line in gt_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    return entries, gts


def ground_truth_to_string(gt: list[dict]) -> str:
    """Convert ground truth dict format → '[func(a=1, b=2)]' string."""
    calls = []
    for call_dict in gt:
        for func_name, params in call_dict.items():
            args = []
            for k, v_list in params.items():
                # Pick first non-empty acceptable value
                val = next((v for v in v_list if v != ""), None)
                if val is not None:
                    args.append(f"{k}={repr(val)}")
            calls.append(f"{func_name}({', '.join(args)})")
    return "[" + ", ".join(calls) + "]"


def build_prompt(entry: dict) -> str:
    schemas = json.dumps(entry["function"], indent=2)
    user_msg = entry["question"][0][0]["content"]
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Available functions:\n{schemas}\n\n"
        f"Question: {user_msg}"
    )


def build_dataset(split: float = 0.9) -> tuple[list[dict], list[dict]]:
    """
    Returns (sft_data, grpo_data).
    - sft_data: first `split` fraction of each category
    - grpo_data: remaining fraction, used as GRPO rollout prompts
    """
    sft_data, grpo_data = [], []

    for category in TRAIN_CATEGORIES:
        try:
            entries, gts = load_category(category)
        except FileNotFoundError as e:
            print(f"[WARN] skipping {category}: {e}")
            continue

        n_train = int(len(entries) * split)

        for i, entry in enumerate(entries):
            if entry["id"] not in gts:
                continue
            if entry["question"][0][0]["content"] in EXCLUDED_QUESTIONS:
                continue

            record = {
                "id": entry["id"],
                "category": category,
                "prompt": build_prompt(entry),
                "completion": ground_truth_to_string(gts[entry["id"]]),
                "function": entry["function"],
                "ground_truth": gts[entry["id"]],
            }

            if i < n_train:
                sft_data.append(record)
            else:
                grpo_data.append(record)

    return sft_data, grpo_data


if __name__ == "__main__":
    sft, grpo = build_dataset()
    print(f"SFT examples:  {len(sft)}")
    print(f"GRPO examples: {len(grpo)}")
    print("\n--- Sample SFT prompt ---")
    print(sft[0]["prompt"][:400])
    print("\n--- Sample completion ---")
    print(sft[0]["completion"])
