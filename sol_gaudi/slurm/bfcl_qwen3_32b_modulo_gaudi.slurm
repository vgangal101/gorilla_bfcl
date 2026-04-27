#!/bin/bash
#SBATCH --job-name=bfcl_qwen3_32b_modulo_gaudi
#SBATCH --partition=gaudi
#SBATCH --qos=class_gaudi
#SBATCH --account=class_cse59827694spring2026
#SBATCH --gres=gpu:hl225:4
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=384G
#SBATCH --time=24:00:00
#SBATCH --output=sol_gaudi/logs/bfcl_qwen3_32b_modulo_%j.out
#SBATCH --error=sol_gaudi/logs/bfcl_qwen3_32b_modulo_%j.err

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

PORT=8000
echo "vLLM server will bind to 127.0.0.1:${PORT}"

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
    --model "Qwen/Qwen3-32B" \
    --tensor-parallel-size 4 \
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

export REMOTE_OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
export REMOTE_OPENAI_API_KEY="EMPTY"
export LOCAL_SERVER_ENDPOINT="127.0.0.1"
export LOCAL_SERVER_PORT="${PORT}"

# --- Run LLM-Modulo + evaluate ---
cd "${BFCL_ROOT}"
MODULO_CONFIG_PATH="${MODULO_CONFIG:-configs/modulo_example.yaml}"
MODULO_RESULT_DIR="${MODULO_RESULT_DIR:-result_modulo}"
echo "LLM-Modulo: config=${MODULO_CONFIG_PATH} model=Qwen/Qwen3-32B result_dir=${MODULO_RESULT_DIR}"

# Pull category list from the YAML so we can pass --test-category to evaluate.
TEST_CAT=$(python3 -c "import yaml,sys; print(','.join(yaml.safe_load(open(sys.argv[1]))['categories']))" "${MODULO_CONFIG_PATH}")
echo "Categories from config: ${TEST_CAT}"

python3 run_bfcl_llm_modulo.py \
    --config "${MODULO_CONFIG_PATH}" \
    --model "Qwen/Qwen3-32B" \
    --endpoint "${REMOTE_OPENAI_BASE_URL}" \
    --result-dir "${MODULO_RESULT_DIR}"

echo "Running bfcl evaluate"
# --partial-eval so a num_samples-capped smoke run still scores cleanly.
bfcl evaluate --model "Qwen/Qwen3-32B" --test-category "${TEST_CAT}" --result-dir "${MODULO_RESULT_DIR}" --partial-eval

echo "DONE model=Qwen/Qwen3-32B (LLM-Modulo)"
