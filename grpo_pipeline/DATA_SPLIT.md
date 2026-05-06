# Training / Evaluation Data Split

Documents how training and evaluation data was chosen, what overlap
analysis was done, and what was done to ensure a clean split.

---

## The Problem

The naive approach — training on a 90/10 split of the same BFCL categories
used for evaluation — is train/test contamination. The model sees 90% of the
benchmark questions during SFT and is evaluated on 100% of them. Any score
improvement could reflect memorisation rather than genuine learning.

---

## The Split

**Train on non-live categories** (synthetically generated questions):

| Category | Language | Examples |
|---|---|---|
| `simple_python` | Python | 400 |
| `simple_java` | Java | 100 |
| `simple_javascript` | JavaScript | 50 |
| `parallel` | Python | 200 |
| `multiple` | Python | 200 |
| **Total** | | **~950** |

**Evaluate on live categories** (real human-submitted queries):

| Category | Examples |
|---|---|
| `live_simple` | 258 |
| `live_parallel` | 16 |
| `live_multiple` | 1053 |
| `live_parallel_multiple` | 24 |
| **Total** | **~1351** |

The live categories were introduced in BFCL v4 as a separately collected set
of real-world user queries. They are not synthetically generated and test
the same function-calling capabilities as the non-live categories but with
entirely different questions.

---

## Overlap Analysis

Before finalising this split, `check_data_overlap.py` was run to verify
there is no meaningful leakage between the two sets. Four checks were performed
across all train/eval category combinations:

### Check 1 — Question text (exact string match)

| Result | Count |
|---|---|
| Overlapping questions | **1** |

One question appeared verbatim in both `multiple` (train) and `live_multiple`
(eval):

> *"Find out the rewards for playing Fortnite on Playstation platform with
> different missions and trophies"*

**Fix:** This question is filtered from the training set in `data_prep.py`
via `EXCLUDED_QUESTIONS`. It remains in the eval set. The model never trains
on it.

### Check 2 — Function names

| Result | Count |
|---|---|
| Shared function names | **9** |

The 9 shared names are: `book_flight`, `filter_list`, `game_missions.list`,
`game_rewards.get`, `game_scores.get`, `get_current_weather`, `send_email`,
`sort_list`, `sum_elements`.

These are generic API names that independently appear in any function-calling
dataset. Check 3 (exact schemas) confirms they are independently authored
definitions — the parameter descriptions and wording differ. Not a
contamination concern.

### Check 3 — Full schema (name + description + parameters, exact)

| Result | Count |
|---|---|
| Exact schema overlap | **0** ✓ |

No function schema in the training set is an exact copy of any schema in the
evaluation set. This is the most important check — it means the model is not
memorising specific API definitions word-for-word.

### Check 4 — Parameter structure (function name + param names/types)

| Result | Count |
|---|---|
| Structural overlap | **4** |

Four function signatures share both name and parameter structure between
splits: `game_rewards.get`, `game_scores.get`, `sort_list`, `sum_elements`.
Since Check 3 is clean (descriptions differ), these are independently authored
definitions that happen to share the same interface. This is standard in API
design — many APIs share the same parameter shape. Noted in results but not
considered disqualifying.

### Summary

| Check | Result |
|---|---|
| Question text | 1 overlap — **filtered from training** |
| Function names | 9 shared — coincidental, different schemas |
| Exact schemas | **CLEAN** ✓ |
| Param fingerprints | 4 shared — different descriptions, acceptable |

---

## Justification for the Split

Training on synthetic non-live data and evaluating on live human-submitted
queries is a stronger experimental setup than training and evaluating on the
same pool:

1. **No question overlap** (after filtering the 1 duplicate)
2. **No schema overlap** — the model cannot memorise specific API definitions
3. **Different generation methodology** — synthetic (train) vs human-annotated
   (eval) tests generalisation, not recall
4. **More meaningful result** — improvement on live queries from real users is
   a stronger claim than improvement on the same synthetic benchmark used for
   training

---

## Implementation

- **`data_prep.py`** — `TRAIN_CATEGORIES` uses non-live categories.
  `EXCLUDED_QUESTIONS` filters the 1 duplicate question.
- **`slurm/run_pipeline.slurm`** — `TEST_CATEGORIES` uses live categories.
- **`smoke_tests/check_data_overlap.py`** — run this any time categories
  change to re-verify the split is clean.

---

## Future Work — Stronger Split (not yet implemented)

The academically strongest split would be to **not use BFCL data for training
at all**, and train on a completely independent function-calling dataset.
Evaluating on the full BFCL benchmark with zero training/eval dataset
connection is the cleanest possible experimental setup.

**Candidate training datasets:**

| Dataset | Description | Why useful |
|---|---|---|
| **Gorilla APIBench** | 16,450 (instruction, API) pairs for HuggingFace, TorchHub, TensorFlow Hub | Original dataset from this repo's paper; same spirit as BFCL |
| **ToolBench** | 126k real-world tool-use instructions across 16k APIs | Large, diverse, real APIs |
| **ToolAlpaca** | 3.9k tool-use instances across 400 real APIs | Smaller, easier to work with |
| **API-Bank** | 73 API tools, 314 dialogues | Multi-turn tool use |

**What needs to be done:**

1. Pick a dataset (ToolBench is the most comprehensive; APIBench is already
   in this repo under `gorilla/`)
2. Write a new `data_prep_external.py` that converts the chosen dataset into
   the same `(prompt, completion, function, ground_truth)` format that
   `sft_train.py` and `grpo_train.py` expect
3. The reward function (`reward.py`) stays unchanged — it uses BFCL's
   `ast_checker` which only runs during evaluation, not training
4. Update `TRAIN_CATEGORIES` / dataset path in `data_prep_external.py`
5. Evaluate on the full BFCL benchmark (`simple_*`, `parallel`, `multiple`,
   `live_*`) — no exclusions needed since training data comes from elsewhere
6. Re-run `check_data_overlap.py` adapted for the external dataset to confirm

**Key challenge:** The external dataset's function schemas will use a different
format than BFCL's. `data_prep.py`'s `ground_truth_to_string()` and
`build_prompt()` assume BFCL's JSON structure. A new data prep script will
need to normalise the external format into the same shape.

**When implementing:** start with APIBench (already in `gorilla/` in this
repo) since it requires no download and shares the same origin as BFCL.
