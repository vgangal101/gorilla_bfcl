# BFCL GRPO Training Pipeline

Trains Qwen3-8B to call functions correctly on the BFCL benchmark using two phases:
1. **SFT** — supervised warm-start on ground-truth BFCL examples
2. **GRPO** — reinforcement learning using the BFCL AST checker as the reward signal

The trained model is evaluated with the same `bfcl evaluate` command used for baselines, so results are directly comparable to Gorilla, LLM Modulo, and the vanilla Qwen3 runs.

---

## Pipeline Structure

```
grpo_pipeline/
├── data_prep.py       Extract (prompt, ground_truth) pairs from BFCL JSONL files
├── reward.py          Wrap bfcl_eval ast_checker as a 0/1 scalar reward for GRPOTrainer
├── sft_train.py       Phase 1 — LoRA SFT warm-start on Qwen3-8B
├── grpo_train.py      Phase 2 — GRPO fine-tuning with K=4 rollouts per prompt
├── hf_job.py          Submit the full pipeline as a HuggingFace Job on A100
└── slurm/
    ├── setup_env.sh   One-time conda env setup (run once on login node)
    ├── sft_train.slurm    SBATCH script for SFT phase
    ├── grpo_train.slurm   SBATCH script for GRPO phase
    └── logs/              Job stdout/stderr land here (auto-created)
```

All scripts are run from the **repo root** (`gorilla_bfcl/`), not from inside `grpo_pipeline/`.

---

## How It Works

### Why two phases?

GRPO requires the model to produce some correct outputs early in training so that reward variance exists within each group of K rollouts. If all K responses score 0, the group-relative advantage is undefined and the gradient vanishes. SFT on correct BFCL examples first ensures the model can at least parse the output format, giving GRPO something to work with.

### What is the reward?

For each rollout the model generates a function call string like `[func(a=1, b=2)]`. The reward function in `reward.py`:
1. Parses that string into a list of dicts using BFCL's own AST parser
2. Passes it to `ast_checker` alongside the ground-truth acceptable values
3. Returns `1.0` if the checker says valid, `0.0` otherwise

This is the exact same checker `bfcl evaluate` uses, so GRPO is directly optimizing the benchmark metric.

### Training categories

| Category | Language | # Examples (approx) |
|---|---|---|
| `simple_python` | Python | ~400 |
| `simple_java` | Java | ~400 |
| `simple_javascript` | JavaScript | ~400 |
| `parallel_function` | Python | ~200 |
| `multiple_function` | Python | ~200 |

90% of each category goes to SFT, 10% to GRPO. Multi-turn and live categories are excluded — multi-turn requires stateful conversation handling that GRPO's single-turn rollouts do not support.

---

## Prerequisites

### Python environment

Install from `berkeley-function-call-leaderboard/` (the BFCL harness must be importable):

```bash
cd berkeley-function-call-leaderboard
pip install -e .
cd ..
```

Then install training dependencies:

```bash
pip install "transformers>=4.45" "trl>=0.12" "datasets>=2.20" "peft>=0.12" "accelerate>=0.34"
```

### Hardware

| Option | Notes |
|---|---|
| **SOL `public` partition — A100 (configured)** | Account `class_cse59818158spring2026`, QoS `class`. Scripts are pre-configured for this. |
| **HuggingFace Jobs** | Single A100-80G on cloud; no queue wait; use `hf_job.py`. |
| **HuggingFace Jobs** | Single A100-80G on cloud; no queue wait; use `hf_job.py`. |
| **SOL Gaudi partition** | Not supported — TRL's GRPOTrainer uses CUDA-specific kernels. Gaudi is inference-only in this project. |

Minimum GPU memory for Qwen3-8B with LoRA (rank 16): **~40 GB** (fits on one H100-80G or A100-80G).

### HuggingFace token (for HF Jobs path)

```bash
huggingface-cli login
# or: export HF_TOKEN=hf_...
```

---

## Step-by-Step Instructions

### Step 0 — Verify you are on the right branch

```bash
git status
# Should show: On branch bfcl_rl
```

If not:

```bash
git switch bfcl_rl
```

---

### Step 1 — Sanity-check data prep

**Run this before anything else.** It verifies that the BFCL data files exist, parses them, and prints sample prompts so you can confirm the format looks right.

```bash
python grpo_pipeline/data_prep.py
```

Expected output:
```
SFT examples:  ~1530
GRPO examples: ~170

--- Sample SFT prompt ---
You are a function calling assistant...
Available functions:
[{"name": "calculate_triangle_area", ...}]
Question: Find the area of a triangle...

--- Sample completion ---
[calculate_triangle_area(base=10, height=5)]
```

**If you see `[WARN] skipping <category>`** it means that BFCL data file does not exist at the expected path. Check that `berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_<category>.json` exists. The `v4` prefix matters — older checkouts used `v3`.

---

### Step 2 — Smoke-test the reward function

Verifies that `ast_checker` is importable and scores correctly before wasting training time.

```bash
python grpo_pipeline/reward.py
```

Expected output:
```
correct call → reward=1.0
wrong value  → reward=0.0
unparseable  → reward=0.0
```

If you get an `ImportError` for `bfcl_eval`, the `pip install -e .` from Step 0 prerequisites was not run.

---

### Step 3 — Run SFT (Phase 1)

**Local / SOL NVIDIA:**

```bash
python grpo_pipeline/sft_train.py
```

**Multi-GPU (2× A100-40G):**

```bash
accelerate launch --num_processes 2 grpo_pipeline/sft_train.py
```

What to expect:
- Runtime: ~1–2 hours on a single A100-80G
- Checkpoint saved to: `checkpoints/sft_final/`
- Watch `train/loss` in logs — should drop from ~2.5 to ~0.5 over 2 epochs
- If loss does not move at all, the learning rate (2e-5) is too low or the LoRA rank is too small

Do not proceed to GRPO until SFT loss has visibly converged.

---

### Step 4 — Run GRPO (Phase 2)

**Local / SOL NVIDIA:**

```bash
python grpo_pipeline/grpo_train.py
```

**Multi-GPU:**

```bash
accelerate launch --num_processes 2 grpo_pipeline/grpo_train.py
```

What to expect:
- Runtime: ~4–8 hours on a single A100-80G (3 epochs, K=4 rollouts)
- Checkpoint saved to: `checkpoints/grpo_final/`
- Key metrics to watch in logs:

| Metric | Healthy range | Problem if... |
|---|---|---|
| `rewards/mean` | Rising from ~0.3 toward ~0.7+ | Stuck near 0 → reward fn broken |
| `rewards/std` | 0.2–0.5 | Near 0 for >50 steps → all rollouts same score, increase `temperature` |
| `kl` | Slowly rising, stays <1.0 | Spikes >2.0 → increase `beta` in `grpo_train.py` |
| `train/loss` | Slowly decreasing | Erratic spikes → lower `learning_rate` to 1e-6 |

**If `rewards/std` collapses to 0:** open `grpo_train.py` and raise `temperature` from `0.8` to `1.0`. This increases diversity across the K rollouts so the group has variance to learn from.

**If `rewards/mean` plateaus below 0.4:** the model is stuck on hard categories. Open `data_prep.py` and temporarily remove `simple_java` and `simple_javascript` from `TRAIN_CATEGORIES` — cross-language generalisation is harder and can stall early training.

---

### Step 5 — Run on ASU SOL (SLURM)

This is the recommended path if you have access to SOL's `general` partition.
All commands run from the repo root on a **SOL login node**.

**One-time setup (do this before your first job):**

```bash
cd /scratch/$USER/gorilla_bfcl   # or wherever the repo lives on SOL
bash grpo_pipeline/slurm/setup_env.sh
```

This creates the `bfcl_rl` conda env and installs all training dependencies.
Takes ~5 minutes. Only needs to be run once.

**Submit SFT:**

```bash
sbatch grpo_pipeline/slurm/sft_train.slurm
```

Note the job ID printed (e.g. `Submitted batch job 12345678`).

**Submit GRPO — chained to start automatically after SFT finishes:**

```bash
# Replace 12345678 with your actual SFT job ID
sbatch --dependency=afterok:12345678 grpo_pipeline/slurm/grpo_train.slurm
```

Or submit independently if SFT is already done:

```bash
sbatch grpo_pipeline/slurm/grpo_train.slurm
```

**Monitor jobs:**

```bash
squeue -u $USER                          # see queue status
tail -f grpo_pipeline/slurm/logs/sft_<JOBID>.out   # live SFT log
tail -f grpo_pipeline/slurm/logs/grpo_<JOBID>.out  # live GRPO log
```

**If H100s are unavailable** (queue full), switch to A100 on the `public` partition.
Edit the top of the `.slurm` file and change:
```bash
#SBATCH --partition=general
#SBATCH --gres=gpu:h100:1
```
to:
```bash
#SBATCH --partition=public
#SBATCH --gres=gpu:a100:1
```

**Expected runtimes on H100:**

| Phase | Time |
|---|---|
| SFT (2 epochs, ~1500 examples) | ~1–2 hours |
| GRPO (3 epochs, ~170 examples, K=4) | ~4–6 hours |

---

### Step 5b (Alternative) — Submit as HuggingFace Job

If you do not have local A100 access, use `hf_job.py` to run the full pipeline on HF's infrastructure.

**Before submitting, edit these two lines in `hf_job.py`:**

```python
HF_REPO_ID = "SaiGanesh314/bfcl-grpo-qwen3-8b"   # your HF username/repo
```

And inside `INLINE_JOB_SCRIPT`, update the git clone URL to wherever your `bfcl_rl` branch is accessible (a public GitHub URL or a HuggingFace dataset repo mirror).

Then submit:

```bash
python grpo_pipeline/hf_job.py
```

The job runs SFT → GRPO → pushes the final checkpoint to your HF Hub repo. Monitor progress at `https://huggingface.co/jobs/<job_id>`.

---

### Step 6 — Evaluate with BFCL

Once training is done, run the standard BFCL evaluation against the saved checkpoint. The trained model is served via vLLM just like the baseline runs.

```bash
cd berkeley-function-call-leaderboard

# Start vLLM server pointing at the GRPO checkpoint
python -m vllm.entrypoints.openai.api_server \
    --model ../checkpoints/grpo_final \
    --port 8001 &

# Generate with skip-server-setup (server already running)
REMOTE_OPENAI_BASE_URL=http://localhost:8001/v1 \
bfcl generate \
    --model grpo_qwen3_8b \
    --test-category simple_python,simple_java,simple_javascript,parallel_function,multiple_function \
    --skip-server-setup

# Score
bfcl evaluate \
    --model grpo_qwen3_8b \
    --test-category simple_python,simple_java,simple_javascript,parallel_function,multiple_function
```

Results land in `score/grpo_qwen3_8b/` — compare directly against `score/Qwen_Qwen3-8B/` (baseline) and `score/` from your LLM Modulo runs.

---

## Key Configuration Knobs

All knobs are at the top of each script — no need to dig into the training loop.

| Knob | File | Default | When to change |
|---|---|---|---|
| `MODEL` | `sft_train.py` | `Qwen/Qwen3-8B` | Use `Qwen3-14B` for better ceiling if you have 2×A100 |
| `SFT_CHECKPOINT` | `grpo_train.py` | `checkpoints/sft_final` | Change if you saved SFT elsewhere |
| `TRAIN_CATEGORIES` | `data_prep.py` | 5 categories | Remove hard categories if GRPO stalls early |
| `split` | `data_prep.py` | `0.9` | Lower to `0.8` for more GRPO examples |
| `num_generations` (K) | `grpo_train.py` | `4` | Increase to `8` for better gradient estimates; doubles memory |
| `temperature` | `grpo_train.py` | `0.8` | Raise to `1.0` if reward std collapses |
| `beta` | `grpo_train.py` | `0.01` | Raise to `0.05` if KL divergence spikes |
| `lora_r` | both train scripts | `16` | Raise to `32` for more capacity; doubles LoRA memory |

---

## Comparing Results

After running `bfcl evaluate`, the aggregate CSV is at:

```
berkeley-function-call-leaderboard/score/data_non_live.csv
```

The columns you care about for comparison:

| Model | Source |
|---|---|
| Qwen3-8B baseline | `score/Qwen_Qwen3-8B/` |
| Qwen3-8B + LLM Modulo | `result_modulo/Qwen3-8B/` scored via `bfcl evaluate --partial-eval` |
| Qwen3-8B GRPO (this pipeline) | `score/grpo_qwen3_8b/` |

A successful GRPO run should close most of the gap between baseline and LLM Modulo on `simple_*` categories, while matching or exceeding LLM Modulo on `parallel_function` and `multiple_function` (where argument chaining matters most).
