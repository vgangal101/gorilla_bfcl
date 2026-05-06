#!/usr/bin/env bash
#SBATCH --job-name=bfcl_eval
#SBATCH --account=class_cse59818158spring2026
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=grpo_pipeline/slurm/logs/eval_%j.out
#SBATCH --error=grpo_pipeline/slurm/logs/eval_%j.err
#
# Evaluation — serves the GRPO checkpoint via vLLM, runs bfcl generate,
# then scores with bfcl evaluate. Results land in:
#   result/grpo_qwen3_8b/   (raw generations)
#   score/grpo_qwen3_8b/    (scores, comparable to baseline)
#
# Submit from repo root:
#   sbatch grpo_pipeline/slurm/evaluate.slurm
#
# Or chain automatically after GRPO training:
#   sbatch --dependency=afterok:<GRPO_JOB_ID> grpo_pipeline/slurm/evaluate.slurm
# -----------------------------------------------------------------------

set -euo pipefail

echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Started:   $(date)"
echo "=========================================="

# ── Environment ────────────────────────────────────────────────────────
module load mamba/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bfcl_rl

export HF_HOME="/scratch/${USER}/hf_cache"
export TOKENIZERS_PARALLELISM=false

# ── Guard: GRPO checkpoint must exist ───────────────────────────────────
if [ ! -d "checkpoints/grpo_final" ]; then
    echo "ERROR: checkpoints/grpo_final/ not found."
    echo "Run grpo_train.slurm first, or chain with --dependency=afterok:<GRPO_JOB_ID>"
    exit 1
fi

# ── Config ───────────────────────────────────────────────────────────────
BASE_MODEL="Qwen/Qwen3-8B"
LORA_CHECKPOINT="checkpoints/grpo_final"
MODEL_NAME="grpo_qwen3_8b"           # used as result/score directory name
VLLM_PORT=8001
VLLM_LOG="grpo_pipeline/slurm/logs/vllm_${SLURM_JOB_ID}.log"

TEST_CATEGORIES="simple_python,simple_java,simple_javascript,parallel_function,multiple_function"

# ── Sanity check ─────────────────────────────────────────────────────────
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Base model:      $BASE_MODEL"
echo "LoRA checkpoint: $LORA_CHECKPOINT"
echo "Model name:      $MODEL_NAME"

# ── Step 1: Start vLLM server ────────────────────────────────────────────
echo ""
echo "Starting vLLM server..."

python -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --enable-lora \
    --lora-modules "${MODEL_NAME}=${LORA_CHECKPOINT}" \
    --port "$VLLM_PORT" \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID (log: $VLLM_LOG)"

# ── Step 2: Wait for server to be ready ──────────────────────────────────
echo "Waiting for vLLM server to be ready..."
MAX_WAIT=300   # 5 minutes
ELAPSED=0
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "ERROR: vLLM server did not start within ${MAX_WAIT}s."
        echo "Last vLLM log lines:"
        tail -20 "$VLLM_LOG"
        kill "$VLLM_PID" 2>/dev/null || true
        exit 1
    fi
    sleep 10
    ELAPSED=$((ELAPSED + 10))
    echo "  ...waiting (${ELAPSED}s elapsed)"
done
echo "vLLM server is ready."

# ── Step 3: bfcl generate ────────────────────────────────────────────────
echo ""
echo "Running bfcl generate..."
cd berkeley-function-call-leaderboard

REMOTE_OPENAI_BASE_URL="http://localhost:${VLLM_PORT}/v1" \
bfcl generate \
    --model "$MODEL_NAME" \
    --test-category "$TEST_CATEGORIES" \
    --skip-server-setup \
    --num-threads 4

cd ..

# ── Step 4: Stop vLLM server ─────────────────────────────────────────────
echo ""
echo "Stopping vLLM server..."
kill "$VLLM_PID" 2>/dev/null || true
wait "$VLLM_PID" 2>/dev/null || true
echo "vLLM server stopped."

# ── Step 5: bfcl evaluate ────────────────────────────────────────────────
echo ""
echo "Running bfcl evaluate..."
cd berkeley-function-call-leaderboard

bfcl evaluate \
    --model "$MODEL_NAME" \
    --test-category "$TEST_CATEGORIES"

cd ..

echo ""
echo "=========================================="
echo "Evaluation complete: $(date)"
echo "Generations: berkeley-function-call-leaderboard/result/${MODEL_NAME}/"
echo "Scores:      berkeley-function-call-leaderboard/score/${MODEL_NAME}/"
echo "Compare against baseline in score/Qwen_Qwen3-8B/"
echo "=========================================="
