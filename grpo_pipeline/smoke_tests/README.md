# Smoke Tests

Run these before submitting a full pipeline job to verify each phase works.
All smoke test files live here — nothing touches the main pipeline code.

---

## Step 1 — Login node checks (no GPU, instant)

Run from the repo root with `bfcl_rl` env active:

```bash
conda activate bfcl_rl
bash grpo_pipeline/smoke_tests/smoke_login.sh
```

Checks:
- All required packages importable (torch, trl, peft, vllm, bfcl_eval, etc.)
- `data_prep.py` can load and parse BFCL data files
- `reward.py` reward function scores correctly (correct=1.0, wrong=0.0, unparseable=0.0)
- All 5 BFCL data files + ground truth files exist

**Fix failures here before going further** — no point burning GPU time if imports are broken.

---

## Step 2 — Full pipeline smoke test (GPU required, ~1–2 hours)

Submit from the repo root:

```bash
sbatch grpo_pipeline/smoke_tests/smoke_test.slurm
```

Monitor:
```bash
tail -f grpo_pipeline/slurm/logs/smoke_<JOBID>.out
```

What it runs vs the full pipeline:

| | Full pipeline | Smoke test |
|---|---|---|
| Model | Any | Qwen3-8B only |
| SFT examples | ~1500 | 4 |
| SFT steps | ~full epochs | 3 steps |
| GRPO examples | ~170 | 4 |
| GRPO steps | ~full epochs | 2 steps |
| GRPO K rollouts | 4 | 2 |
| Eval categories | 5 categories | `simple_python` only |
| Eval test cases | all | 5 |
| Walltime | 24h | 2h |

**If the smoke test completes without error, the full pipeline is safe to run.**

---

## What each phase confirms

**Phase 1 (SFT):**
- Base model downloads correctly from HuggingFace
- LoRA training runs without OOM or import errors
- Checkpoint saves to `checkpoints/qwen3_8b/smoke_sft_final/`

**Phase 2 (GRPO):**
- SFT checkpoint loads correctly via `PeftModel.from_pretrained`
- Reward function integrates with GRPOTrainer (correct kwargs)
- Group-relative advantage calculation works with K=2
- Checkpoint saves to `checkpoints/qwen3_8b/smoke_grpo_final/`

**Phase 3 (Evaluate):**
- `merge_lora.py` merges adapter into base model correctly
- vLLM starts and serves the merged model on port 8002
- `bfcl generate --run-ids` runs on 5 specific test cases
- `bfcl evaluate --partial-eval` scores the 5 results

---

## Smoke test checkpoints

Smoke test writes to separate directories so it never pollutes full-run checkpoints:

```
checkpoints/qwen3_8b/
├── smoke_sft_final/      SFT smoke checkpoint (safe to delete after)
├── smoke_grpo_final/     GRPO smoke checkpoint (safe to delete after)
└── smoke_merged/         Merged smoke model (safe to delete after)
```

These are gitignored. Delete them freely after the smoke test passes.

---

## If a phase fails

**Phase 1 fails (SFT):**
- Check the `.err` log for import errors or OOM
- If OOM: the A100 may have other jobs — try resubmitting
- If import error: re-run `bash grpo_pipeline/slurm/setup_env.sh` and resubmit

**Phase 2 fails (GRPO):**
- `ERROR: SFT checkpoint not found` → Phase 1 failed silently; check `.err`
- `rewards/std near 0` warning in logs is OK for a 2-step smoke test — not enough steps to see variance

**Phase 3 fails (vLLM):**
- `vLLM server did not start within 300s` → check `smoke_vllm_<JOBID>.log` in `grpo_pipeline/slurm/logs/`
- Common cause: model loading takes longer than expected — resubmit
- `bfcl generate` error → check that `test_case_ids_to_generate.json` was created correctly
