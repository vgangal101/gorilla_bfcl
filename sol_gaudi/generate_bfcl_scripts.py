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
MODELS: dict[str, tuple[str, int, int, int, str, str]] = {
    "qwen3_4b":    ("Qwen/Qwen3-4B",       1, 1, 24, "160G", "06:00:00"),
    "qwen3_8b":    ("Qwen/Qwen3-8B",       1, 1, 24, "160G", "08:00:00"),
    "qwen3_32b":   ("Qwen/Qwen3-32B",      4, 4, 60, "384G", "12:00:00"),
    "gemma4_31b":  ("google/gemma-4-31B-it", 4, 4, 60, "384G", "12:00:00"),
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
#SBATCH --output={logs_dir}/bfcl_{model_key}_%j.out
#SBATCH --error={logs_dir}/bfcl_{model_key}_%j.err
{mail_line}
set -euo pipefail

# --- Load config ---
SCRIPT_DIR="{sol_gaudi_root}"
source "${{SCRIPT_DIR}}/config.env"

# --- Load mamba / activate env ---
module load "${{MAMBA_MODULE}}"
source activate "${{BFCL_GAUDI_ENV}}"

# --- Habana / vLLM runtime ---
export HABANA_VISIBLE_DEVICES=all
export HF_HOME="${{HF_HOME}}"
export APPTAINERENV_PT_HPU_LAZY_MODE="${{APPTAINERENV_PT_HPU_LAZY_MODE}}"
export APPTAINERENV_VLLM_SKIP_WARMUP="${{APPTAINERENV_VLLM_SKIP_WARMUP}}"
export APPTAINERENV_VLLM_DELAYED_SAMPLING="${{APPTAINERENV_VLLM_DELAYED_SAMPLING}}"
export APPTAINERENV_TORCHDYNAMO_DISABLE="${{APPTAINERENV_TORCHDYNAMO_DISABLE}}"
export APPTAINERENV_HF_HOME="${{HF_HOME}}"

# --- Pick a free TCP port for the vLLM server ---
PORT=$(python - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
)
echo "vLLM server will bind to 127.0.0.1:${{PORT}}"

# --- Launch vLLM OpenAI-compatible server inside Apptainer ---
apptainer run --bind "/scratch/${{USER}}:/scratch/${{USER}}" "${{VLLM_GAUDI_SIF}}" \
    python -m vllm.entrypoints.openai.api_server \
    --model "{hf_id}" \
    --tensor-parallel-size {tp} \
    --gpu-memory-utilization "${{VLLM_GPU_MEMORY_UTILIZATION}}" \
    --max-model-len "${{VLLM_MAX_MODEL_LEN}}" \
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
TEST_CAT="${{BFCL_TEST_CATEGORY:-simple_python,multi_turn_base,live}}"
echo "Running bfcl generate on categories: ${{TEST_CAT}}"
bfcl generate --model "{hf_id}" --test-category "${{TEST_CAT}}" --skip-server-setup --num-threads "${{BFCL_NUM_THREADS}}"

echo "Running bfcl evaluate"
bfcl evaluate --model "{hf_id}" --test-category "${{TEST_CAT}}"

echo "DONE model={hf_id}"
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
    logs_dir = defaults.get("LOGS_DIR", str(SCRIPT_DIR / "logs"))

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
            logs_dir=logs_dir,
            sol_gaudi_root=str(SCRIPT_DIR),
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
