# Bug Report — grpo_pipeline

All bugs found and fixed during code review of the initial implementation.
Recorded here so future contributors know what to watch for if the pipeline
is extended or refactored.

---

## Critical Bugs (would have caused silent failures or wrong results)

**Bug 3 — Unregistered model name in `bfcl generate`**
- File: `slurm/run_pipeline.slurm`
- Problem: `bfcl generate --model grpo_qwen3_8b` — BFCL validates model names against a registered list and rejects unknown names.
- Fix: Use `$BFCL_HF_MODEL` (the registered base model name) throughout Phase 3. vLLM's `--served-model-name` maps the registered name to the local merged weights.

**Bug 4 — Reward function signature mismatched TRL**
- File: `reward.py`
- Problem: `bfcl_reward_fn(prompts, completions, ...)` — TRL 0.12's GRPOTrainer calls `reward_fn(completions, **dataset_columns)` with completions as the only positional arg. `prompts` silently received the completions list; everything was misaligned.
- Fix: Changed to `bfcl_reward_fn(completions, **kwargs)` with explicit extraction from kwargs.

**Bug 6 — GRPOTrainer can't load a PEFT adapter directory**
- File: `grpo_train.py`
- Problem: `GRPOTrainer(model=SFT_CHECKPOINT)` — the SFT checkpoint is a LoRA adapter directory (no full model weights). `AutoModelForCausalLM.from_pretrained` fails on it.
- Fix: Explicitly load base model with `AutoModelForCausalLM.from_pretrained(HF_MODEL)`, then apply adapter with `PeftModel.from_pretrained(base, SFT_CHECKPOINT, is_trainable=True)`. Pass the model object (not a path string) to GRPOTrainer.

**Bug 8 — `device_map="auto"` breaks multi-GPU DDP**
- File: `grpo_train.py`
- Problem: With `accelerate launch --num_processes N`, all GPUs are visible to each process. `device_map="auto"` spreads the model across all visible GPUs (model parallelism) inside each DDP worker, conflicting with DDP which requires each process to hold the full model replica.
- Fix: Removed `device_map` argument entirely — accelerate/DDP handles device placement.

**Bug 10 — vLLM LoRA alias same as base model name**
- File: `slurm/run_pipeline.slurm`
- Problem: `--lora-modules "Qwen/Qwen3-8B=/path"` — alias equals base model name, causing ambiguous routing or startup errors in vLLM.
- Fix: Switched to merge approach — `merge_lora.py` merges LoRA into base model, vLLM serves the merged model with `--served-model-name` for name aliasing. No LoRA modules needed at serve time.

**Bug 11 — `bfcl generate` missing `--allow-overwrite`**
- File: `slurm/run_pipeline.slurm`
- Problem: Without `--allow-overwrite`, `bfcl generate` silently skips test IDs where result files already exist (e.g. from a prior baseline run), loading the old outputs as "done". GRPO results would be silently mixed with baseline outputs in the same result directory — wrong scores with no error.
- Fix: Added `--allow-overwrite` to the `bfcl generate` call.

**Bug 13 — `import sys` accidentally removed from `merge_lora.py`**
- File: `merge_lora.py`
- Problem: When cleaning up an unnecessary `sys.path.insert`, `import sys` was removed along with it. `sys.argv` and `sys.exit` in `__main__` would throw `NameError` at runtime — script always crashes.
- Fix: Restored `import sys`.

---

## Moderate Bugs (would have caused job failures)

**Bug 1 — `--export` stripped job environment**
- File: `slurm/submit.sh`
- Problem: `--export="VAR=val"` without `ALL` tells SLURM to export ONLY those named variables, stripping `PATH`, `USER`, `HOME`, etc. `module load` and `conda activate` both depend on `PATH` and fail silently.
- Fix: Changed to `--export="ALL,VAR=val"` to inherit the full environment plus the new variables.

**Bug 5 — vLLM not installed**
- File: `slurm/setup_env.sh`
- Problem: `pip install -e berkeley-function-call-leaderboard/` — `vllm==0.8.5` is an optional extra in BFCL's `pyproject.toml`, not a base dependency. Phase 3 fails immediately with `No module named vllm`.
- Fix: Changed to `pip install -e "berkeley-function-call-leaderboard/[oss_eval_vllm]"`.

**Bug 7 — vLLM orphaned on Phase 3 failure**
- File: `slurm/run_pipeline.slurm`
- Problem: With `set -euo pipefail`, if `bfcl generate` errors, the script exits immediately before `kill $VLLM_PID` runs. vLLM continues consuming GPUs for the rest of the job's walltime.
- Fix: Added `trap '[[ -n "$VLLM_PID" ]] && kill "$VLLM_PID" 2>/dev/null || true' EXIT` immediately after starting vLLM.

**Bug 9 — Separate pip calls risked vllm dependency conflicts**
- File: `slurm/setup_env.sh`
- Problem: Two separate `pip install` calls — first installed vllm==0.8.5 with its pinned deps, second could upgrade those deps breaking vllm at runtime.
- Fix: Combined into a single `pip install` call so the resolver sees all constraints together.

---

## Minor Bugs (cosmetic or low-impact)

**Bug 2 — `dict_keys` instead of `list()` for `remove_columns`**
- File: `sft_train.py`
- Problem: `remove_columns=sft_data[0].keys()` passes a `dict_keys` object. Some versions of the `datasets` library require an explicit `list`.
- Fix: Changed to `remove_columns=list(sft_data[0].keys())`.

**Bug 12 — Unnecessary `sys.path.insert` in `merge_lora.py`**
- File: `merge_lora.py`
- Problem: `sys.path.insert(0, ".../berkeley-function-call-leaderboard")` present but `merge_lora.py` imports nothing from `bfcl_eval`. Dead code.
- Fix: Removed. (Note: this removal inadvertently caused Bug 13 above.)

---

## Patterns to Watch For

If the pipeline is extended, these categories kept causing bugs:

1. **SLURM `--export`**: Always use `ALL,...` prefix when passing custom env vars. Without it, the job environment is stripped.
2. **PEFT adapter paths vs full model paths**: `AutoModelForCausalLM.from_pretrained` cannot load adapter directories. Always load base model first, then apply adapter via `PeftModel.from_pretrained`.
3. **`device_map="auto"` in DDP context**: Fine for single-GPU or inference. Never use during `accelerate launch` multi-GPU training — let accelerate assign devices.
4. **`bfcl generate` skips existing results silently**: Always pass `--allow-overwrite` when you intend a fresh generation run.
5. **vLLM model name aliasing**: The `--lora-modules alias=path` alias must differ from the base model name. When they are the same, use the merge + `--served-model-name` approach instead.
6. **Optional pip extras**: Check `pyproject.toml` before assuming a package is installed by the base `pip install`. vLLM is under `[oss_eval_vllm]`.
