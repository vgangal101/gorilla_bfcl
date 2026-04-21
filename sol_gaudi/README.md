# BFCL on ASU SOL with Intel Gaudi2

Run Berkeley Function-Calling Leaderboard evaluations on SOL's Gaudi partition via the Habana vLLM fork (delivered as an Apptainer container).

Target models (full sweep): `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B`, `Qwen/Qwen3-14B`, `Qwen/Qwen3-32B`, `google/gemma-4-31B-it`.

---

## 0. Prerequisites (one-time, per teammate)

- ASU SOL account with access to `class_gaudi` QoS and `class_cse59827694spring2026` account
- SSH into a login node: `ssh <asurite>@login.sol.rc.asu.edu`

## 1. Clone & configure (~2 minutes)

```bash
cd /scratch/$USER
git clone https://github.com/HectorHernandez1/gorilla_bfcl.git   # or your fork
cd gorilla_bfcl
git checkout bfcl_gaudi

cp sol_gaudi/config.env.example sol_gaudi/config.env
# Edit sol_gaudi/config.env only if your account/QoS/SIF path differs from the defaults.
# Most teammates can leave it untouched for the class account.
```

## 2. One-command bring-up (~3 min interactive + ~20 min queue wait)

```bash
./sol_gaudi/quickstart.sh
```

This runs:

1. `gaudi_setup_check.sh` — probes `hl-smi`, `/dev/accel*`, apptainer, container, mamba env. Aborts with a clear message if anything is missing.
2. `setup_gaudi_env.sh` — creates the `bfcl_gaudi` mamba env, installs BFCL in editable mode, resolves/builds `vllm_gaudi.sif`.
3. `generate_bfcl_scripts.py` — emits five `.slurm` files into `sol_gaudi/slurm/`.
4. Submits the smoke test: `qwen3_4b` on `simple_python`. Prints the job ID and the commands to watch it.

On first run, if `config.env` doesn't exist, `quickstart.sh` copies the example and asks you to re-run.

## 3. Watching a job

```bash
./sol_gaudi/manage_bfcl_gaudi.sh status            # squeue for your bfcl_*_gaudi jobs
./sol_gaudi/manage_bfcl_gaudi.sh status <JOB_ID>   # detailed squeue + sacct for one job
./sol_gaudi/manage_bfcl_gaudi.sh logs <JOB_ID>     # last 100 lines of the .out file
./sol_gaudi/manage_bfcl_gaudi.sh results qwen3_4b  # score row once job finishes
```

## 4. Running the full sweep

After the smoke test passes:

```bash
./sol_gaudi/manage_bfcl_gaudi.sh submit-all        # queues all 5 models
# or pick one:
./sol_gaudi/manage_bfcl_gaudi.sh submit qwen3_32b
./sol_gaudi/manage_bfcl_gaudi.sh submit gemma4_31b
```

Override the test category for a given submission:

```bash
BFCL_TEST_CATEGORY=all_scoring ./sol_gaudi/manage_bfcl_gaudi.sh submit qwen3_8b
```

Available model keys: `qwen3_4b`, `qwen3_8b`, `qwen3_14b`, `qwen3_32b`, `gemma4_31b`.

## Test coverage (what this sweep does and does not run)

By default, the generated SLURM scripts pass `--test-category single_turn,multi_turn,memory` — **20 of BFCL's 22 scoring categories**:

- `non_live` (7): `simple_python`, `simple_java`, `simple_javascript`, `multiple`, `parallel`, `parallel_multiple`, `irrelevance`
- `live` (6): `live_simple`, `live_multiple`, `live_parallel`, `live_parallel_multiple`, `live_irrelevance`, `live_relevance`
- `multi_turn` (4): `multi_turn_base`, `multi_turn_miss_func`, `multi_turn_miss_param`, `multi_turn_long_context`
- `memory` (3): `memory_kv`, `memory_vector`, `memory_rec_sum` — uses `sentence-transformers` + `faiss-cpu` (already installed by `pip install -e .`); `all-MiniLM-L6-v2` encoder auto-downloads to `HF_HOME` on first run (~90 MB)

**Skipped on purpose:**

| skipped | count | reason |
| --- | --- | --- |
| `web_search_base`, `web_search_no_snippet` | 2 | Requires a paid [SerpAPI](https://serpapi.com/) key. Free tier (100 searches/mo) is nowhere near the ~300–600 searches a full run consumes; paid starts at $75/mo. Out of scope for a class budget. |

To override and run something different for a single submission:

```bash
BFCL_TEST_CATEGORY=all_scoring ./sol_gaudi/manage_bfcl_gaudi.sh submit qwen3_8b
# (requires SERPAPI_API_KEY in berkeley-function-call-leaderboard/.env for web_search_*)
```

## 5. Reading results

Aggregate CSVs live in `berkeley-function-call-leaderboard/score/`:

- `data_overall.csv` — headline accuracy per model
- `data_non_live.csv`, `data_live.csv`, `data_multi_turn.csv` — category breakdowns

Quick numeric summary:

```bash
./sol_gaudi/manage_bfcl_gaudi.sh results            # all models
./sol_gaudi/manage_bfcl_gaudi.sh results qwen3_32b  # one model
```

## 6. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `sbatch: error: Invalid account` | Confirm `SLURM_ACCOUNT` in `config.env`; verify with `sacctmgr show user $USER` |
| `quickstart.sh` says container not found | Run `bash sol_gaudi/build_vllm_gaudi_sif.sh` (pulls ~20 GB from Habana vault) |
| Job stuck `PD` with reason `QOSGrpGRES` | Class HPU quota is saturated; wait or downgrade to a smaller model |
| vLLM server never becomes healthy | `logs/bfcl_<model>_<JOB>.err` — look for OOM / tokenizer issues; reduce `VLLM_MAX_MODEL_LEN` in `config.env` |
| `bfcl generate` says model unsupported | For Gemma 4 you need the BFCL patch (see `bfcl_eval/constants/supported_models.py` + `model_config.py`); re-install with `pip install -e berkeley-function-call-leaderboard` |
| Cancel everything | `./sol_gaudi/manage_bfcl_gaudi.sh cancel` |

## 7. A100 parity check (optional)

Once `qwen3_32b` completes on Gaudi, compare `data_overall.csv` against the existing A100 run:

- Delta within ~1 point: expected hardware/kernel variance
- Delta > ~5 points on `multi_turn` or `agentic`: likely an inference-path bug — open an issue before continuing the sweep

---

## Layout

```
sol_gaudi/
  README.md                    # this file
  quickstart.sh                # one-command entry
  gaudi_setup_check.sh         # hardware/software diagnostic
  setup_gaudi_env.sh           # create bfcl_gaudi env + resolve SIF
  build_vllm_gaudi_sif.sh      # fallback: build SIF from Habana vault
  generate_bfcl_scripts.py     # emit per-model .slurm files
  manage_bfcl_gaudi.sh         # submit | submit-all | status | logs | results | cancel | clean
  config.env.example           # copy to config.env
  slurm/                       # generated .slurm files (gitignored)
  logs/                        # SLURM stdout/stderr (gitignored)
  containers/                  # vllm_gaudi.sif if built locally (gitignored)
```

## Design notes

- **No BFCL code changes for Qwen3 runs.** The SLURM script launches the Apptainer-packaged vLLM server itself, then invokes `bfcl generate --skip-server-setup` with `REMOTE_OPENAI_BASE_URL` pointing at the local port. This is wired at `bfcl_eval/model_handler/local_inference/base_oss_handler.py:42-47,161`.
- **Gemma 4 is the only BFCL source change.** We add the `google/gemma-4-31B-it` entry to `supported_models.py` + `model_config.py` and reuse the existing `GemmaHandler`. If Gemma 4's chat template diverges from Gemma 3, fork a `gemma4.py` handler.
- **All tunables in `config.env`.** Scripts never hardcode paths. Models, HPU counts, memory, time, and test categories can all be adjusted in one place.
