#!/bin/bash
#SBATCH --job-name=bfcl_qwen3_8b_gaudi
#SBATCH --partition=gaudi
#SBATCH --qos=class_gaudi
#SBATCH --account=class_cse59827694spring2026
#SBATCH --gres=gpu:hl225:1
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=sol_gaudi/logs/bfcl_qwen3_8b_%j.out
#SBATCH --error=sol_gaudi/logs/bfcl_qwen3_8b_%j.err

set -euo pipefail

# --- Load config (paths derive from SLURM_SUBMIT_DIR so the file is portable) ---
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/sol_gaudi"
source "${SCRIPT_DIR}/config.env" 2>/dev/null || source "${SCRIPT_DIR}/config.env.example"

# --- Load mamba / activate env ---
module load "${MAMBA_MODULE}"
source activate "${BFCL_GAUDI_ENV}"

# --- Dirs we bind into the container ---
HF_CACHE_HOST="${HF_HOME}"
mkdir -p "${HF_CACHE_HOST}"

HABANA_LOGS="${LOGS_DIR}/habana_${SLURM_JOB_ID}"
mkdir -p "${HABANA_LOGS}"

# Port 8000 — --exclusive owns the node so no conflict possible.
PORT=8000
echo "vLLM server will bind to 127.0.0.1:${PORT}"

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
    --bind "${HF_CACHE_HOST}:/mnt/hf_cache" \
    --bind "${HABANA_LOGS}:/var/log/habana_logs" \
    --env HF_HOME=/mnt/hf_cache \
    --env HABANA_VISIBLE_DEVICES=all \
    --env HPU_MEM_GAUDI2=96 \
    --env PT_HPU_LAZY_MODE=0 \
    --env PT_HPU_ENABLE_LAZY_COLLECTIVES=0 \
    --env VLLM_SKIP_WARMUP=True \
    --env VLLM_GRAPH_RESERVED_MEM=0.10 \
    --env PYTHONUNBUFFERED=1 \
    --writable-tmpfs \
    "${VLLM_GAUDI_SIF}" \
    python3 -m vllm.entrypoints.openai.api_server \
    --model "Qwen/Qwen3-8B" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --max-num-seqs 16 \
    --trust-remote-code \
    --port "${PORT}" &
SERVER_PID=$!
echo "vLLM PID: ${SERVER_PID}"

cleanup() {
    echo "Stopping vLLM server (PID ${SERVER_PID})..."
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --- Wait for server health (up to 10 min) ---
echo "Waiting for /v1/models to respond..."
for i in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "Server healthy after ${i}0s"
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: vLLM server exited before becoming healthy" >&2
        exit 1
    fi
    sleep 10
done
if ! curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "ERROR: vLLM server never became healthy" >&2
    exit 1
fi

# --- Point BFCL at the running server ---
export REMOTE_OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
export REMOTE_OPENAI_API_KEY="EMPTY"
export LOCAL_SERVER_ENDPOINT="127.0.0.1"
export LOCAL_SERVER_PORT="${PORT}"

# --- Run BFCL generate + evaluate ---
cd "${BFCL_ROOT}"
TEST_CAT="${BFCL_TEST_CATEGORY:-simple_python,multi_turn_base,live}"
echo "Running bfcl generate on categories: ${TEST_CAT}"
bfcl generate --model "Qwen/Qwen3-8B" --test-category "${TEST_CAT}" --skip-server-setup --num-threads "${BFCL_NUM_THREADS}"

echo "Running bfcl evaluate"
bfcl evaluate --model "Qwen/Qwen3-8B" --test-category "${TEST_CAT}"

echo "DONE model=Qwen/Qwen3-8B"
