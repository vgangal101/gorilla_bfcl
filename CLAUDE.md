# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Layout

This is a monorepo hosting multiple Gorilla projects. Active work on this branch (`bfcl_inference`) is focused on **BFCL** (Berkeley Function Calling Leaderboard) evaluation, especially running local/self-hosted models (e.g. Qwen3-32B) via vLLM on a SLURM cluster ("SOL").

Top-level sub-projects:
- `berkeley-function-call-leaderboard/` — main BFCL evaluation harness (primary working area)
- `gorilla/` — original Gorilla model + APIBench inference/eval code
- `openfunctions/` — Gorilla OpenFunctions-V2
- `goex/` — Gorilla Execution Engine runtime
- `raft/` — Retrieval-Augmented Fine-Tuning
- `agent-arena/`, `data/` — supporting datasets and arena infra

Root-level helper scripts (custom to this clone, not upstream) drive BFCL runs on SLURM:
- `run_vllm_qwen32b_SOL.sh` — SBATCH script that runs `bfcl generate` + `bfcl evaluate` for Qwen3-32B on 2×A100s with the vllm backend. Note the hard-coded `cd /Users/vgangal/...` path from a prior machine — update it before submitting.
- `sol_gaudi/` — **Intel Gaudi2 sweep on SOL** (its own README, `quickstart.sh`, per-model SLURM generator, Apptainer SIF builder, lifecycle wrapper). Covers Qwen3-{4B,8B,14B,32B} + Gemma-4-31B. See `sol_gaudi/README.md` to run, `sol_gaudi/GAUDI_EVAL_GUIDE.md` for hardware reference.
- `manage_qwen_eval.sh` — submit / status / logs / cancel / results / clean wrapper around the SLURM job (looks for `evaluate_qwen3_32b_vllm.slurm`).
- `setup_qwen_eval.sh`, `setup_api_key.sh` — one-time env setup.
- `analyze_qwen_results.py` — post-hoc score parsing.

Reference docs in the root (`BFCL_QUICK_START.md`, `BFCL_ARCHITECTURE.md`, `BFCL_EVAL_ANALYSIS.md`, `QWEN3_32B_EVAL_GUIDE.md`, `SLURM_GUIDE.md`, `VLLM_FIXES_SUMMARY.md`) are the authoritative guides for workflows; read them before changing the SLURM/vLLM pipeline.

## Common Commands

All BFCL commands run from `berkeley-function-call-leaderboard/` after `pip install -e .` (Python 3.10+, preferably in a conda env named `bfcl`).

```bash
# Install (editable, from berkeley-function-call-leaderboard/)
pip install -e .
# Optional extras:  .[oss_eval_vllm]  or  .[oss_eval_sglang]

# Entry point installed as `bfcl`; also runnable as `python -m bfcl_eval`
bfcl --version
bfcl models            # list supported models
bfcl test-categories   # list test categories

# Two-phase eval: generate responses, then score them
bfcl generate --model <MODEL> --test-category <CAT> [--num-threads N]
bfcl evaluate --model <MODEL> --test-category <CAT>

# Self-hosted model via vLLM (multi-GPU)
bfcl generate --model Qwen/Qwen3-32B --test-category multi_turn,agentic \
  --backend vllm --num-gpus 2

# Use a pre-existing OpenAI-compatible server (skip auto-spawn)
bfcl generate --model <name> --test-category <cat> --skip-server-setup
# relies on LOCAL_SERVER_ENDPOINT/PORT or REMOTE_OPENAI_BASE_URL envs

# Target specific test IDs: populate test_case_ids_to_generate.json then
bfcl generate --model <MODEL> --run-ids
bfcl evaluate --model <MODEL> --test-category <CATS> --partial-eval
```

Outputs: generation → `result/<model>/BFCL_v3_<cat>_result.json`; scoring → `score/<model>/...` plus aggregate CSVs `score/data_overall.csv`, `data_non_live.csv`, `data_live.csv`, `data_multi_turn.csv`.

Credentials live in `berkeley-function-call-leaderboard/.env` (copy from `bfcl_eval/.env.example`). There are no unit tests in this repo — "testing" a change means running `bfcl generate`/`evaluate` on a small category (e.g. `simple_python`) and inspecting the score JSON/CSV.

## BFCL Architecture (big picture)

The pipeline is two-phase and cleanly separated:

1. **Generation** (`bfcl_eval/_llm_response_generation.py`) loads test cases from `bfcl_eval/data/`, builds prompts or native function-calling tool specs, routes to a handler, and writes raw model outputs to `result/`.
2. **Evaluation** (`bfcl_eval/eval_checker/`) parses generated calls, builds ASTs for generated + ground-truth calls, compares them semantically (formatting-agnostic), and writes per-category scores + aggregated CSVs to `score/`.

Model integration is organized around a handler abstraction in `bfcl_eval/model_handler/`:
- `base_handler.py` — abstract `ModelHandler` interface every provider implements.
- `api_inference/` — OpenAI, Anthropic, Google, Cohere, Mistral, Writer, Bedrock, etc.
- `local_inference/` — self-hosted backends (vLLM, sglang); these launch a local OpenAI-compatible server and then reuse the API path, unless `--skip-server-setup` is passed.
- `parser/` — response extractors (JSON, XML, tool-use blocks) shared across handlers.

Each model appears in `SUPPORTED_MODELS.md` with a suffix convention: `-FC` for native function-calling mode, plain name for prompt mode. Handlers dispatch on this suffix. Multi-turn categories drive a stateful conversation loop inside the handler rather than a single call.

Test categories are declared in `bfcl_eval/constants/` and in `TEST_CATEGORIES.md`; aliases like `all`, `all_scoring`, `multi_turn`, `agentic`, `live` expand to sets of JSONL files under `bfcl_eval/data/`.

## Working notes specific to this clone

- The Qwen/SLURM scripts contain absolute paths from the original author's machine (`/Users/vgangal/...`). Fix paths before running on SOL.
- `transformers==4.51.3`, `numpy==1.26.4`, `tree_sitter==0.21.3` are pinned in `pyproject.toml`; don't bump casually — past commits (`778c148`, `4d989a9`) specifically adjusted these to get the vLLM Qwen runner working.
- `TORCHDYNAMO_DISABLE=1` is set in the SLURM script and is required for the current vLLM/Qwen combination; see `VLLM_FIXES_SUMMARY.md` for context.

## Gaudi2 path (ASU SOL)

All Gaudi work lives under `sol_gaudi/` and has its own docs. Do **not** replicate the NVIDIA root-level pattern for Gaudi — the infrastructure there is different in every dimension (Apptainer SIF instead of mamba+pip, `class_gaudi` QoS on `class_cse59827694spring2026` account, per-model SLURM scripts generated from a Python template, `hl-smi` not `nvidia-smi`, `--gres=gpu:hl225:N` not `--gres=gpu:N`).

Entry points, in priority order:
- `sol_gaudi/README.md` — the operational guide: `quickstart.sh`, `manage_bfcl_gaudi.sh {submit|status|logs|results|cancel}`, sweep targets (Qwen3-{4B,8B,14B,32B}, Gemma-4-31B).
- `sol_gaudi/config.env.example` — single source of truth for account, QoS, partition, gres prefix, Habana runtime envs (`PT_HPU_LAZY_MODE=0`, `VLLM_SKIP_WARMUP=True`, `VLLM_DELAYED_SAMPLING=True`, `TORCHDYNAMO_DISABLE=1`), SIF path, HF cache. Every `.sh` / generated `.slurm` in the folder sources this.
- `sol_gaudi/generate_bfcl_scripts.py` — per-model SBATCH generator (holds the (hpus, tp, cpus, mem, time) table for each model). Change allocations here, not in hand-edited `.slurm` files.
- `sol_gaudi/GAUDI_EVAL_GUIDE.md` — reference-only: SOL hardware facts, `/data/sse/gaudi` assets, PyTorch-on-HPU code snippets, authoritative SLURM values.

Key operational constraints:
- Upstream `vllm==0.8.5` pinned in BFCL's `pyproject.toml` is CUDA-only; the Gaudi runtime comes from Habana's vLLM fork delivered via Apptainer SIF (`vllm_gaudi.sif`, built from `vault.habana.ai/gaudi-docker/1.23.0/...` by `build_vllm_gaudi_sif.sh`). Don't `pip install vllm` on the `gaudi` partition.
- BFCL code is **unchanged** for Qwen3 on Gaudi — the SLURM script launches the Apptainer vLLM server and uses `bfcl generate --skip-server-setup` with `REMOTE_OPENAI_BASE_URL`. Gemma-4-31B is the only model that requires BFCL source edits (`supported_models.py` + `model_config.py`).
- The workshop deck only explicitly listed DeepSeek-R1, Llama 3.x, Qwen 2.5, and Qwen2.5-VL-7B as HPU-vLLM-supported; Qwen3-* and Gemma 4 are not on that list. Always run `manage_bfcl_gaudi.sh submit qwen3_4b` as a smoke test before queuing the full sweep.

Coverage scope for this sweep:
- Default `BFCL_TEST_CATEGORY` in the generated SLURM scripts is `single_turn,multi_turn` — 17 of BFCL's 22 scoring categories.
- `web_search_*` is **skipped**: requires a paid SerpAPI key ($75/mo+; free tier of 100 searches/month is well below what one run needs). Out of scope for a class budget.
- `memory_*` is **skipped**: 465 cases × 3 backends adds multi-hour walltime per model without changing the hardware-comparison story. Re-enable for full leaderboard parity.
- See `sol_gaudi/README.md` "Test coverage" for the full rationale and the override recipe.

KV-cache tuning (if a run OOMs or stalls — e.g. long multi-turn / agentic trajectories):
- `VLLM_MAX_MODEL_LEN` in `sol_gaudi/config.env` — context budget per sequence; KV cost scales linearly with it. Drop from the default 40960 first.
- `VLLM_GPU_MEMORY_UTILIZATION` — default 0.85; push to 0.92 to squeeze more KV out of HBM before adding HPUs.
- `BFCL_NUM_THREADS` — lowering concurrency reduces in-flight sequences, which reduces KV pressure.
- HPU count / TP — bump `hpus` + `tp` for the affected model in `generate_bfcl_scripts.py` (not the `.slurm` — it's regenerated) and re-run `python3 sol_gaudi/generate_bfcl_scripts.py`. Per-HPU HBM is 96 GB; qwen3_32b at TP=4 already gets 384 GB total (~260 GB KV budget after weights), ~3× what 2×A100-80G offers.
