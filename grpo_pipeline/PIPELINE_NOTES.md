# Pipeline Notes

Technical reference for understanding how the GRPO pipeline works under the hood.
Complements the step-by-step instructions in `README.md`.

---

## The Three Phases

### Phase 1 — SFT

Supervised fine-tuning on BFCL ground-truth function call examples (non-live
categories only). The model learns to produce valid function call syntax.

Without SFT, GRPO starts blind — the model outputs random text, the AST
checker always returns 0, there is no reward variance, and the policy gradient
is zero. SFT ensures the model starts with at least partially parseable outputs
so GRPO has a signal to learn from.

### Phase 2 — GRPO

RL fine-tuning using the BFCL AST checker as the reward signal. For each
prompt, the trainer generates K=4 rollouts, scores each with `ast_checker`
(1.0 if correct, 0.0 otherwise), computes group-relative advantages, and
updates the policy via policy gradient.

No learned reward model is needed — the benchmark's own scorer IS the reward
function. The model is trained to internalise what LLM Modulo enforced at
inference time (correct function name, arguments, types), but at 1× inference
cost instead of up to 5× for LLM Modulo.

### Phase 3 — Evaluation

Standard BFCL benchmark run on the **live** (held-out) test categories. This
is not a verifier — it is a measurement of how well the trained model
generalises to out-of-distribution queries that were never seen during training.

Steps:
1. **Merge** — `merge_lora.py` fuses the GRPO LoRA adapter into the base model
   weights. vLLM cannot serve a bare PEFT adapter directory.
2. **Serve** — vLLM loads the merged model and exposes an OpenAI-compatible
   HTTP endpoint at `localhost:8001`.
3. **Generate** — `bfcl generate --skip-server-setup` sends each live test
   prompt to vLLM and writes the raw model responses to `result/`.
4. **Score** — `bfcl evaluate` runs the AST checker offline on those responses
   and writes per-category scores to `score/`.

vLLM is the inference engine, not the verifier. The verifier is `bfcl evaluate`.
The live categories are used specifically because the model was trained on
non-live — this is the proper out-of-distribution test of generalisation.

---

## How vLLM Works in Phase 3

Phase 3 does not run inference directly. It uses vLLM as a local
**OpenAI-compatible HTTP server** that `bfcl generate` talks to over localhost.

```
merge_lora.py
    Merges the GRPO LoRA adapter weights into the base model.
    Output: checkpoints/<model_key>/merged/
    (Full model weights — no adapter needed at serve time.)

vllm.entrypoints.openai.api_server  [background process]
    Loads the merged model from checkpoints/<model_key>/merged/.
    --served-model-name tells vLLM to answer requests for the base
    model name (e.g. Qwen/Qwen3-8B) even though the files are local.
    Listens on http://localhost:8001/v1
    Exposes POST /v1/chat/completions  (OpenAI-compatible format)

bfcl generate --skip-server-setup
    Reads REMOTE_OPENAI_BASE_URL=http://localhost:8001/v1
    Sends one HTTP request per BFCL test case.
    Writes raw model outputs → result/Qwen_Qwen3-8B/

[vLLM server killed]

bfcl evaluate
    Offline AST matching against ground truth — no GPU, no server.
    Writes scores → score/Qwen_Qwen3-8B/
```

### Why `--skip-server-setup`?

Without this flag, `bfcl generate` would try to launch its own vLLM server
from scratch. With it, `bfcl generate` skips that and just points to the
server we already started via `REMOTE_OPENAI_BASE_URL`. This is the same
pattern used by the baseline Gaudi inference pipeline.

### Why merge before serving?

The GRPO training saves a PEFT LoRA adapter (small delta weights), not a
full model. Serving options:

| Approach | Problem |
|---|---|
| `--lora-modules alias=path` | alias name must differ from base model name — if they match (our case), vLLM has ambiguous routing |
| Merge + serve directly | Clean — single full model, no LoRA alias confusion, `--served-model-name` handles the naming |

We use the merge approach. `merge_and_unload()` is in-place and memory-safe
— peak usage is ~2× model size briefly, well within the A100 allocation.

### Why `--served-model-name`?

`bfcl generate --model Qwen/Qwen3-8B` sends requests with `model=Qwen/Qwen3-8B`
in the JSON body. vLLM loads from a local path (`checkpoints/.../merged/`) but
`--served-model-name Qwen/Qwen3-8B` tells it to answer to that name.
Without this flag, the model name in the API would be the local path string,
and `bfcl generate` would reject it as an unrecognised model.

---

## Hardware: Gaudi vs NVIDIA

### What the other branches used (and still use)

The `bfcl_gaudi` and `LLM-Modulo_gaudi` branches run BFCL **inference** on
**Intel Gaudi2 HPUs** (`hl225` devices) on ASU SOL's `gaudi` partition.
The infrastructure for this lives in `sol_gaudi/` on those branches:

- vLLM is Habana's custom fork, delivered via Apptainer SIF (`vllm_gaudi.sif`)
- Jobs run under `class_gaudi` QoS on the `gaudi` partition
- `hl-smi` not `nvidia-smi`, `--gres=gpu:hl225:N` not `--gres=gpu:a100:N`
- 96 GB HBM per Gaudi2 chip vs 80 GB per A100 — Gaudi wins on memory for large models

### What this branch uses

`bfcl_rl` deleted `sol_gaudi/` entirely. Everything runs on **NVIDIA A100s**
on SOL's `public` partition. There is no Gaudi anywhere in this pipeline.

| Phase | Hardware | Reason |
|---|---|---|
| SFT training | NVIDIA A100 (CUDA) | TRL, PEFT, bitsandbytes all require CUDA |
| GRPO training | NVIDIA A100 (CUDA) | Same — no Gaudi support in TRL's GRPOTrainer |
| LoRA merge | NVIDIA A100 (CUDA) | Runs in the same job, GPUs free after training |
| vLLM serving | NVIDIA A100 (CUDA) | Standard `vllm==0.8.5` is CUDA-only |
| `bfcl evaluate` | CPU (login node ok) | Pure AST matching, no GPU needed |

### Does hardware affect scores?

`bfcl evaluate` is deterministic AST matching — scores are identical regardless
of whether generation ran on Gaudi or A100, as long as the same outputs were
produced. At `temperature=0` (greedy decoding), the generated tokens are
deterministic on both hardware types for the same model weights.

At `temperature > 0`, sampling randomness differs between hardware due to
floating-point implementation differences. For fair comparison against the
Gaudi baseline, keep temperature at `0.001` (BFCL default) during evaluation.

### If you want to evaluate the GRPO model on Gaudi

The merged checkpoint (`checkpoints/<model>/merged/`) is a standard
HuggingFace model directory. You can copy it to SOL scratch and serve it
via the Gaudi vLLM SIF exactly like the baseline models, then run
`bfcl generate --skip-server-setup` against it. Results would be directly
comparable to the baseline Gaudi runs.

---

## Why GRPO — Theoretical Basis

BFCL scores function calls with deterministic AST matching. This gives us a
**verifiable reward signal** — exactly the setup GRPO works best on (same
principle as DeepSeek-R1 using math/code verification).

The pipeline exploits this:
- **SFT** teaches the model the output format so GRPO starts with non-zero reward variance
- **GRPO** generates K=4 rollouts per question, scores each with `ast_checker`,
  and computes group-relative advantages: `A_i = (r_i - mean(r)) / std(r)`
- No learned reward model needed — the benchmark's own scorer IS the reward function
- The trained model internalises what LLM Modulo enforced at inference time
  (correct function name, correct arguments, correct types) — but at 1× inference
  cost instead of up to 5× (max_iters=5)

### Key numbers to watch during GRPO

| Metric | Healthy | Problem |
|---|---|---|
| `rewards/mean` | Rising from ~0.3 toward 0.7+ | Stuck near 0 → reward fn broken |
| `rewards/std` | 0.2–0.5 | Near 0 for >50 steps → all rollouts same, raise `temperature` |
| `kl` | Slowly rising, stays <1.0 | Spikes >2.0 → raise `beta` in `grpo_train.py` |
| `train/loss` | Slowly decreasing | Erratic spikes → lower `learning_rate` |

---

## Checkpoint Layout

After a full pipeline run for `qwen3_8b`:

```
checkpoints/
└── qwen3_8b/
    ├── sft/              Epoch checkpoints from SFT (intermediate, safe to delete)
    ├── sft_final/        Final SFT LoRA adapter — input to GRPO
    ├── grpo/             Epoch checkpoints from GRPO (intermediate, safe to delete)
    ├── grpo_final/       Final GRPO LoRA adapter — input to merge
    └── merged/           Full merged model — served by vLLM for evaluation
```

`checkpoints/*/` is gitignored (model weights are too large to commit).
The `.gitkeep` file keeps the `checkpoints/` directory tracked.

---

## Comparing Results

After `bfcl evaluate` completes, scores land in:

```
berkeley-function-call-leaderboard/score/Qwen_Qwen3-8B/
```

This is the same directory as the vanilla Qwen3-8B baseline (BFCL uses the
model name to key the score directory). To keep results separate, copy or
rename before running evaluation for a different variant:

```bash
cp -r score/Qwen_Qwen3-8B score/Qwen_Qwen3-8B_grpo_backup
```

The aggregate CSV is at:
```
berkeley-function-call-leaderboard/score/data_non_live.csv
```

Compare columns: baseline (vanilla bfcl generate) vs GRPO-trained vs
LLM Modulo (result_modulo/ scored with --partial-eval).
