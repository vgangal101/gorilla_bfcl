#!/usr/bin/env python3
"""Emit per-model SBATCH scripts for running BFCL on SOL Gaudi.

Reads knobs from sol_gaudi/config.env (via shell; here we only need the
model table and the template). Writes .slurm files into sol_gaudi/slurm/.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path
from textwrap import dedent

SCRIPT_DIR = Path(__file__).resolve().parent
SLURM_DIR = SCRIPT_DIR / "slurm"

# model_key -> (hf_id, hpus, tp, cpus, mem, time)
# Wall clocks are sized for Qwen3 thinking mode (much higher output-token budget
# than non-thinking) — qwen3_8b previously hit the 8h limit in thinking mode, so
# all Qwen models are given 1.5-2x headroom. gemma4 is unaffected by thinking
# mode but kept at 24h to match the 32B class.
MODELS: dict[str, tuple[str, int, int, int, str, str]] = {
    "qwen3_4b":    ("Qwen/Qwen3-4B-Instruct-2507", 1, 1, 24, "160G", "10:00:00"),
    "qwen3_8b":    ("Qwen/Qwen3-8B",       1, 1, 24, "160G", "16:00:00"),
    "qwen3_14b":   ("Qwen/Qwen3-14B",      2, 2, 32, "200G", "20:00:00"),
    "qwen3_32b":   ("Qwen/Qwen3-32B",      4, 4, 60, "384G", "24:00:00"),
    "gemma4_31b":  ("google/gemma-4-31B-it", 4, 4, 60, "384G", "24:00:00"),
}

# LLM-Modulo runs: same vLLM server boot, but the work step is the iterative
# Generate-Test-Critique loop in `run_bfcl_llm_modulo.py` instead of `bfcl
# generate`. Each entry is processed by up to `max_iters` (5) loop iterations,
# so wall-clock requirements are roughly 3-5x the baseline. Allocations are
# capped at 24h (assumed QoS limit). Categories supported by LLM-Modulo v1 are
# single-turn AST only (see ALLOWED_CATEGORIES in run_bfcl_llm_modulo.py).
# Gemma is excluded — Qwen3 is the only family the modulo runner has been
# exercised against to date.
# model_key -> (hf_id, hpus, tp, cpus, mem, time, default_modulo_config)
MODULO_MODELS: dict[str, tuple[str, int, int, int, str, str, str]] = {
    "qwen3_4b_modulo":   ("Qwen/Qwen3-4B-Instruct-2507", 1, 1, 24, "160G", "24:00:00", "configs/modulo_example.yaml"),
    "qwen3_8b_modulo":   ("Qwen/Qwen3-8B",       1, 1, 24, "160G", "24:00:00", "configs/modulo_example.yaml"),
    "qwen3_14b_modulo":  ("Qwen/Qwen3-14B",      2, 2, 32, "200G", "24:00:00", "configs/modulo_example.yaml"),
    "qwen3_32b_modulo":  ("Qwen/Qwen3-32B",      4, 4, 60, "384G", "24:00:00", "configs/modulo_example.yaml"),
}

TEMPLATE = r"""#!/bin/bash
#SBATCH --job-name=bfcl_{model_key}_gaudi
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --account={account}
#SBATCH --gres={gres_prefix}:{hpus}
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output=sol_gaudi/logs/bfcl_{model_key}_%j.out
#SBATCH --error=sol_gaudi/logs/bfcl_{model_key}_%j.err
{mail_line}
set -euo pipefail

# --- Load config (paths derive from SLURM_SUBMIT_DIR so the file is portable) ---
SCRIPT_DIR="${{SLURM_SUBMIT_DIR}}/sol_gaudi"
source "${{SCRIPT_DIR}}/config.env" 2>/dev/null || source "${{SCRIPT_DIR}}/config.env.example"

# --- Load mamba / activate env ---
module load "${{MAMBA_MODULE}}"
source activate "${{BFCL_GAUDI_ENV}}"

# --- Dirs we bind into the container ---
HF_CACHE_HOST="${{HF_HOME}}"
mkdir -p "${{HF_CACHE_HOST}}"

HABANA_LOGS="${{LOGS_DIR}}/habana_${{SLURM_JOB_ID}}"
mkdir -p "${{HABANA_LOGS}}"

# Port 8000 — --exclusive owns the node so no conflict possible.
PORT=8000
echo "vLLM server will bind to 127.0.0.1:${{PORT}}"

# --- Launch vLLM server inside Apptainer ---
# Binds are per /data/sse/gaudi/guides/vllm-gaudi-apptainer-guide.md: they
# expose the host's Habana runtime & a writable log dir, which is what
# actually fixes HPU init (synStatus=26 without them).
# We invoke vLLM's own OpenAI entrypoint rather than ASU's
# `entrypoints.entrypoint_main server` wrapper, because that wrapper has
# an internal model allowlist (server/server_defaults.yaml) covering only
# Llama-3.1-class models.
apptainer exec \
    --bind /usr/lib64:/host-lib64 \
    --bind /usr/lib/habanalabs:/usr/lib/habanalabs \
    --bind /opt/habanalabs:/opt/habanalabs \
    --bind /usr/bin/shim_ctl:/usr/bin/shim_ctl \
    --bind "${{HF_CACHE_HOST}}:/mnt/hf_cache" \
    --bind "${{HABANA_LOGS}}:/var/log/habana_logs" \
    --env HF_HOME=/mnt/hf_cache \
    --env HABANA_VISIBLE_DEVICES=all \
    --env HPU_MEM_GAUDI2=96 \
    --env PT_HPU_LAZY_MODE=0 \
    --env PT_HPU_ENABLE_LAZY_COLLECTIVES=0 \
    --env VLLM_SKIP_WARMUP=True \
    --env VLLM_GRAPH_RESERVED_MEM=0.10 \
    --env PYTHONUNBUFFERED=1 \
    --writable-tmpfs \
    "${{VLLM_GAUDI_SIF}}" \
    python3 -m vllm.entrypoints.openai.api_server \
    --model "{hf_id}" \
    --tensor-parallel-size {tp} \
    --gpu-memory-utilization "${{VLLM_GPU_MEMORY_UTILIZATION}}" \
    --max-model-len "${{VLLM_MAX_MODEL_LEN}}" \
    --max-num-seqs 16 \
    --trust-remote-code \
    --port "${{PORT}}" &
SERVER_PID=$!
echo "vLLM PID: ${{SERVER_PID}}"

cleanup() {{
    echo "Stopping vLLM server (PID ${{SERVER_PID}})..."
    kill "${{SERVER_PID}}" 2>/dev/null || true
    wait "${{SERVER_PID}}" 2>/dev/null || true
}}
trap cleanup EXIT INT TERM

# --- Wait for server health (up to 10 min) ---
echo "Waiting for /v1/models to respond..."
for i in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:${{PORT}}/v1/models" >/dev/null 2>&1; then
        echo "Server healthy after ${{i}}0s"
        break
    fi
    if ! kill -0 "${{SERVER_PID}}" 2>/dev/null; then
        echo "ERROR: vLLM server exited before becoming healthy" >&2
        exit 1
    fi
    sleep 10
done
if ! curl -fsS "http://127.0.0.1:${{PORT}}/v1/models" >/dev/null 2>&1; then
    echo "ERROR: vLLM server never became healthy" >&2
    exit 1
fi

# --- Point BFCL at the running server ---
export REMOTE_OPENAI_BASE_URL="http://127.0.0.1:${{PORT}}/v1"
export REMOTE_OPENAI_API_KEY="EMPTY"
export LOCAL_SERVER_ENDPOINT="127.0.0.1"
export LOCAL_SERVER_PORT="${{PORT}}"

# --- Run BFCL generate + evaluate ---
cd "${{BFCL_ROOT}}"
# Default coverage: single_turn (non_live + live) + multi_turn + memory.
# 20 of 22 scoring categories. web_search_* is skipped because it requires a
# paid SerpAPI key (~$75/mo). See sol_gaudi/README.md "Test coverage".
TEST_CAT="${{BFCL_TEST_CATEGORY:-single_turn,multi_turn,memory}}"
echo "Running bfcl generate on categories: ${{TEST_CAT}}"
bfcl generate --model "{hf_id}" --test-category "${{TEST_CAT}}" --skip-server-setup --num-threads "${{BFCL_NUM_THREADS}}"

echo "Running bfcl evaluate"
bfcl evaluate --model "{hf_id}" --test-category "${{TEST_CAT}}"

echo "DONE model={hf_id}"
"""

# Same boot sequence as TEMPLATE (Apptainer + vLLM server + health check + env
# exports), but the work step is the LLM-Modulo iterative runner instead of
# `bfcl generate`. The runner writes results into result_modulo/ (relative to
# BFCL_ROOT); `bfcl evaluate --result-dir result_modulo` then scores them.
# Override knobs at submit time:
#   MODULO_CONFIG       path to a YAML/JSON config (default: per-model from MODULO_MODELS)
#   MODULO_RESULT_DIR   output dir for modulo runs (default: result_modulo)
MODULO_TEMPLATE = r"""#!/bin/bash
#SBATCH --job-name=bfcl_{model_key}_gaudi
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --account={account}
#SBATCH --gres={gres_prefix}:{hpus}
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output=sol_gaudi/logs/bfcl_{model_key}_%j.out
#SBATCH --error=sol_gaudi/logs/bfcl_{model_key}_%j.err
{mail_line}
set -euo pipefail

# --- Load config (paths derive from SLURM_SUBMIT_DIR so the file is portable) ---
SCRIPT_DIR="${{SLURM_SUBMIT_DIR}}/sol_gaudi"
source "${{SCRIPT_DIR}}/config.env" 2>/dev/null || source "${{SCRIPT_DIR}}/config.env.example"

# --- Load mamba / activate env ---
module load "${{MAMBA_MODULE}}"
source activate "${{BFCL_GAUDI_ENV}}"

# --- Dirs we bind into the container ---
HF_CACHE_HOST="${{HF_HOME}}"
mkdir -p "${{HF_CACHE_HOST}}"

HABANA_LOGS="${{LOGS_DIR}}/habana_${{SLURM_JOB_ID}}"
mkdir -p "${{HABANA_LOGS}}"

PORT=8000
echo "vLLM server will bind to 127.0.0.1:${{PORT}}"

apptainer exec \
    --bind /usr/lib64:/host-lib64 \
    --bind /usr/lib/habanalabs:/usr/lib/habanalabs \
    --bind /opt/habanalabs:/opt/habanalabs \
    --bind /usr/bin/shim_ctl:/usr/bin/shim_ctl \
    --bind "${{HF_CACHE_HOST}}:/mnt/hf_cache" \
    --bind "${{HABANA_LOGS}}:/var/log/habana_logs" \
    --env HF_HOME=/mnt/hf_cache \
    --env HABANA_VISIBLE_DEVICES=all \
    --env HPU_MEM_GAUDI2=96 \
    --env PT_HPU_LAZY_MODE=0 \
    --env PT_HPU_ENABLE_LAZY_COLLECTIVES=0 \
    --env VLLM_SKIP_WARMUP=True \
    --env VLLM_GRAPH_RESERVED_MEM=0.10 \
    --env PYTHONUNBUFFERED=1 \
    --writable-tmpfs \
    "${{VLLM_GAUDI_SIF}}" \
    python3 -m vllm.entrypoints.openai.api_server \
    --model "{hf_id}" \
    --tensor-parallel-size {tp} \
    --gpu-memory-utilization "${{VLLM_GPU_MEMORY_UTILIZATION}}" \
    --max-model-len "${{VLLM_MAX_MODEL_LEN}}" \
    --max-num-seqs 16 \
    --trust-remote-code \
    --port "${{PORT}}" &
SERVER_PID=$!
echo "vLLM PID: ${{SERVER_PID}}"

cleanup() {{
    echo "Stopping vLLM server (PID ${{SERVER_PID}})..."
    kill "${{SERVER_PID}}" 2>/dev/null || true
    wait "${{SERVER_PID}}" 2>/dev/null || true
}}
trap cleanup EXIT INT TERM

echo "Waiting for /v1/models to respond..."
for i in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:${{PORT}}/v1/models" >/dev/null 2>&1; then
        echo "Server healthy after ${{i}}0s"
        break
    fi
    if ! kill -0 "${{SERVER_PID}}" 2>/dev/null; then
        echo "ERROR: vLLM server exited before becoming healthy" >&2
        exit 1
    fi
    sleep 10
done
if ! curl -fsS "http://127.0.0.1:${{PORT}}/v1/models" >/dev/null 2>&1; then
    echo "ERROR: vLLM server never became healthy" >&2
    exit 1
fi

export REMOTE_OPENAI_BASE_URL="http://127.0.0.1:${{PORT}}/v1"
export REMOTE_OPENAI_API_KEY="EMPTY"
export LOCAL_SERVER_ENDPOINT="127.0.0.1"
export LOCAL_SERVER_PORT="${{PORT}}"

# --- Run LLM-Modulo + evaluate ---
cd "${{BFCL_ROOT}}"
MODULO_CONFIG_PATH="${{MODULO_CONFIG:-{config}}}"
MODULO_RESULT_DIR="${{MODULO_RESULT_DIR:-result_modulo}}"
echo "LLM-Modulo: config=${{MODULO_CONFIG_PATH}} model={hf_id} result_dir=${{MODULO_RESULT_DIR}}"

# Pull category list from the YAML so we can pass --test-category to evaluate.
TEST_CAT=$(python3 -c "import yaml,sys; print(','.join(yaml.safe_load(open(sys.argv[1]))['categories']))" "${{MODULO_CONFIG_PATH}}")
echo "Categories from config: ${{TEST_CAT}}"

python3 run_bfcl_llm_modulo.py \
    --config "${{MODULO_CONFIG_PATH}}" \
    --model "{hf_id}" \
    --endpoint "${{REMOTE_OPENAI_BASE_URL}}" \
    --result-dir "${{MODULO_RESULT_DIR}}"

echo "Running bfcl evaluate"
# --partial-eval so a num_samples-capped smoke run still scores cleanly.
bfcl evaluate --model "{hf_id}" --test-category "${{TEST_CAT}}" --result-dir "${{MODULO_RESULT_DIR}}" --partial-eval

echo "DONE model={hf_id} (LLM-Modulo)"
"""


def main() -> int:
    # Surface the config so humans reading the script know where to look
    config_env = SCRIPT_DIR / "config.env"
    if not config_env.exists():
        print(
            f"warning: {config_env} does not exist. Copy config.env.example -> config.env "
            "or the emitted scripts will source the example file at run time.",
            file=sys.stderr,
        )

    # Defaults pulled from the example (parsed minimally — scripts source config.env at run time)
    defaults = _read_config_defaults(SCRIPT_DIR / "config.env.example")
    account = defaults.get("SLURM_ACCOUNT", "class_cse59827694spring2026")
    qos = defaults.get("SLURM_QOS", "class_gaudi")
    partition = defaults.get("SLURM_PARTITION", "gaudi")
    gres_prefix = defaults.get("SLURM_GRES_PREFIX", "gpu:hl225")
    mail_user = defaults.get("SLURM_MAIL_USER", "").strip()
    mail_line = f"#SBATCH --mail-type=END\n#SBATCH --mail-user={mail_user}" if mail_user else ""

    SLURM_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for model_key, (hf_id, hpus, tp, cpus, mem, time_limit) in MODELS.items():
        content = TEMPLATE.format(
            model_key=model_key,
            hf_id=hf_id,
            hpus=hpus,
            tp=tp,
            cpus=cpus,
            mem=mem,
            time=time_limit,
            partition=partition,
            qos=qos,
            account=account,
            gres_prefix=gres_prefix,
            mail_line=mail_line,
        )
        out = SLURM_DIR / f"bfcl_{model_key}_gaudi.slurm"
        out.write_text(content)
        out.chmod(out.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        written.append(out)

    for model_key, (hf_id, hpus, tp, cpus, mem, time_limit, default_config) in MODULO_MODELS.items():
        content = MODULO_TEMPLATE.format(
            model_key=model_key,
            hf_id=hf_id,
            hpus=hpus,
            tp=tp,
            cpus=cpus,
            mem=mem,
            time=time_limit,
            partition=partition,
            qos=qos,
            account=account,
            gres_prefix=gres_prefix,
            mail_line=mail_line,
            config=default_config,
        )
        out = SLURM_DIR / f"bfcl_{model_key}_gaudi.slurm"
        out.write_text(content)
        out.chmod(out.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        written.append(out)

    print(f"Emitted {len(written)} SBATCH scripts into {SLURM_DIR}:")
    for p in written:
        print(f"  {p.relative_to(SCRIPT_DIR.parent)}")
    return 0


def _read_config_defaults(path: Path) -> dict[str, str]:
    """Parse KEY=VALUE lines from a shell env file (best-effort; ignores complex shell).

    Handles `KEY="value"   # comment` by stripping the inline comment that sits
    outside the quotes.
    """
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, rest = line.partition("=")
        key = key.strip()
        rest = rest.strip()
        if rest.startswith('"'):
            end = rest.find('"', 1)
            value = rest[1:end] if end != -1 else rest[1:]
        elif rest.startswith("'"):
            end = rest.find("'", 1)
            value = rest[1:end] if end != -1 else rest[1:]
        else:
            value = rest.split("#", 1)[0].strip()
        if "$" in value or "`" in value:
            continue
        out[key] = value
    return out


if __name__ == "__main__":
    sys.exit(main())
