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
