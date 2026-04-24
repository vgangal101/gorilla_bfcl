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
- Default `BFCL_TEST_CATEGORY` in the generated SLURM scripts is `single_turn,multi_turn,memory` — 20 of BFCL's 22 scoring categories.
- `web_search_*` is **skipped**: requires a paid SerpAPI key ($75/mo+; free tier of 100 searches/month is well below what one run needs). Out of scope for a class budget.
- `memory_*` is included — `sentence-transformers` + `faiss-cpu` are already in `pyproject.toml`, the `all-MiniLM-L6-v2` encoder auto-downloads to `HF_HOME` on first run.
- See `sol_gaudi/README.md` "Test coverage" for the full rationale and the override recipe.

KV-cache tuning (if a run OOMs or stalls — e.g. long multi-turn / agentic trajectories):
- `VLLM_MAX_MODEL_LEN` in `sol_gaudi/config.env` — context budget per sequence; KV cost scales linearly with it. Drop from the default 40960 first.
- `VLLM_GPU_MEMORY_UTILIZATION` — default 0.85; push to 0.92 to squeeze more KV out of HBM before adding HPUs.
- `BFCL_NUM_THREADS` — lowering concurrency reduces in-flight sequences, which reduces KV pressure.
- HPU count / TP — bump `hpus` + `tp` for the affected model in `generate_bfcl_scripts.py` (not the `.slurm` — it's regenerated) and re-run `python3 sol_gaudi/generate_bfcl_scripts.py`. Per-HPU HBM is 96 GB; qwen3_32b at TP=4 already gets 384 GB total (~260 GB KV budget after weights), ~3× what 2×A100-80G offers.

## LLM-Modulo path (branch: LLM-Modulo_gaudi)

Merges `main` (Gaudi plumbing) + `LLM-Modulo_agent` (the Generate-Test-Critique runner in `llm_modulo_bfcl/`). Runner entry: `berkeley-function-call-leaderboard/run_bfcl_llm_modulo.py`.

Lifecycle mirrors baseline — same `sol_gaudi/manage_bfcl_gaudi.sh` wrapper, same per-model SBATCH generator — plus:
- `sol_gaudi/submit_modulo_sweep.sh` — fires 8B/14B/32B jobs in one go (defaults to `configs/modulo_full.yaml`; override with `MODELS=...` or `MODULO_CONFIG=...`).
- `sol_gaudi/quickstart.sh modulo` — smoke-test entrypoint (submits `qwen3_4b_modulo` on `configs/smoketest.yaml`).
- Model targets live in `MODULO_MODELS` in `sol_gaudi/generate_bfcl_scripts.py`. **4B is excluded** from the modulo sweep: the only Qwen3-4B variant BFCL registers is `Qwen3-4B-Instruct-2507`, which is the non-thinking specialist; thinking-mode prompts regress hard on it. Use 8B/14B/32B.

Gotchas burned through so far (don't repeat them):

1. **`<think>` in Qwen3 output** — hybrid 8B/14B/32B prepend a `<think>…</think>` block before the JSON plan. The modulo `ProposalParser` in `llm_modulo_bfcl/parser.py` strips it via `_THINK_RE` before `json.loads`. If you ever see every sample scoring 0% with `model_result_raw: "[]"`, that regex regressed — fix `parser.py:14` before blaming the model.
2. **Category name vs. data-file mismatch** — `ALLOWED_CATEGORIES` in `run_bfcl_llm_modulo.py` is hand-maintained. BFCL v4 has no `BFCL_v4_simple.json` (split into `simple_python` / `_java` / `_javascript`), so listing `simple` anywhere — in the yaml **or** in `ALLOWED_CATEGORIES` — crashes on the first category with `FileNotFoundError`, after vLLM has already loaded (wastes ~2 min of HPU time per job). The failure goes to `.err`; `.out` just shows "Stopping vLLM server" which looks like a clean exit. Cross-check any new category against `ls berkeley-function-call-leaderboard/bfcl_eval/data/`.
3. **`allow_overwrite: false`** (default in both configs) — if `result_modulo/<Model>/non_live/*.json` exists from a previous run, the runner raises `FileExistsError` right after the config echo. Either `rm -rf result_modulo/<Model>` before resubmitting, or flip `allow_overwrite: true` in the yaml.
4. **Score-dir contamination** — `bfcl evaluate` writes `score/<Model>/` keyed on model name, not run type. A modulo partial-eval over `simple_python` coexists with baseline scores from `agentic/`, `live/`, `multi_turn/`, other `non_live/` categories in the same dir. When bundling modulo results to share, tar only `result_modulo/<Model>` + the specific `score/<Model>/<group>/BFCL_v4_<category>_score.json` files the modulo run produced. Don't send the whole `score/<Model>/` dir.
5. **Aggregator `KeyError` on stale score dirs** — `generate_leaderboard_csv` iterates over everything under `score/` and explodes on orphan dirs from prior experiments (e.g. `Qwen_Qwen3-*.pre_nothink/`). The eval files land fine, but the aggregation step dies. Clean up stale `score/<Model>.*` dirs before running eval.

Config files:
- `berkeley-function-call-leaderboard/configs/smoketest.yaml` — 10 samples on `simple_python`, `max_iters=2`, `allow_overwrite: true`, `result_dir: result_modulo_smoke` (note: the runner currently ignores this override and writes to `result_modulo/` — minor bug, not blocking).
- `berkeley-function-call-leaderboard/configs/modulo_full.yaml` — 10 AST categories (`simple` dropped in commit `ed8bc9a`), `max_iters=5`, all 5 critics, `temperature: 0.001` (was 0.6 — lowered 2026-04-23 to match baseline).

### Current sweep state (as of 2026-04-23 evening)

Re-submitted 8B/14B/32B on `configs/modulo_full.yaml` after fixing two issues that made the Apr 23 morning run diverge from baseline:

- `8fb0541 llm_modulo: drop temperature 0.6 -> 0.001 to match BFCL baseline` — the 0.6 "Qwen thinking-mode recommendation" was the main driver of the previous ~3.55pp regression vs baseline on 32B. Baseline ran at BFCL's CLI default of 0.001 with thinking ON and got 86-96% on single-turn AST without collapse, so the comment warning about 0.001 repetition collapse did not apply to this workload.
- `6e1c8de llm_modulo: set 30-min OpenAI client timeout to survive 32B generation` — previous 32B job hit one `APITimeoutError` on a single sample (default httpx timeout too short). Impact was ~0.04% of samples, not the main accuracy issue, but worth fixing.

Modulo jobs submitted 2026-04-23 evening (via `submit_modulo_sweep.sh`, auto-clean of `result_modulo/<Model>/` worked on submit):
- qwen3_8b_modulo  → job **51803356**
- qwen3_14b_modulo → job **51803357**
- qwen3_32b_modulo → job **51803358**

(Superseded prior jobs 51800071/73/75 — those were submitted earlier the same evening before the `submit_modulo_sweep.sh` resubmit. Cancel those if they're still in the queue to avoid running twice.)

Baseline jobs (on `bfcl_gaudi` branch) are run separately by the coworker — see `sol_gaudi/tar_results.sh` for bundling both sweeps' artifacts when done.

Logs at `sol_gaudi/logs/bfcl_qwen3_{8b,14b,32b}_modulo_<jobid>.{out,err}`. 24h walltimes.

Previous Apr 23 morning run (temperature=0.6, pre-fix) for reference — `simple_java` dropped −22pp on 14B and −15pp on 32B vs baseline; 32B aggregate was 82.37% vs baseline 85.92%. If this re-run lands within ±2pp of baseline, the code-path difference between modulo's `OpenAICompatLLM` (→ vLLM /v1/chat/completions, native Qwen3 template) and baseline's `QwenHandler._format_prompt` (→ vLLM /v1/completions, BFCL-built template) is immaterial and no refactor is needed. If 3pp+ gap persists, next step is rewriting `OpenAICompatLLM` in `run_bfcl_llm_modulo.py:187-210` to route through `QwenHandler`.

Branch settings audit (2026-04-23) — these **cannot** be cleanly unified without rework:
- `bfcl_gaudi` HEAD has `0340576` which adds empty `<think></think>` prefill → thinking OFF. The Apr 19 baseline tarball (scores in `~/Downloads/bfcl_qwen3_14b_32b/`) was generated *before* `0340576` → thinking ON. Re-running `bfcl generate` on `bfcl_gaudi` HEAD today will NOT reproduce those baseline numbers.
- `LLM-Modulo_gaudi` has `_THINKING_TEMPERATURE = 0.6` hardcoded in `QwenHandler`/`QwenFCHandler` (commit `5e6b9f6`). This override only bites if anyone runs `bfcl generate` from the modulo branch — not modulo runs themselves (which go through `OpenAICompatLLM`). Don't remove it without confirming no baseline `bfcl generate` runs are happening on this branch.
