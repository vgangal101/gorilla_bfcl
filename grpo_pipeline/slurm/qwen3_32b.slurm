#!/usr/bin/env bash
#SBATCH --job-name=bfcl_rl_qwen3_32b
#SBATCH --account=class_cse59818158spring2026
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=320G
#SBATCH --cpus-per-task=64
#SBATCH --time=120:00:00
#SBATCH --output=grpo_pipeline/slurm/logs/%x_%j.out
#SBATCH --error=grpo_pipeline/slurm/logs/%x_%j.err
#
# Full RL pipeline for Qwen/Qwen3-32B: SFT → GRPO → Evaluate
#
# Submit:
#   sbatch grpo_pipeline/slurm/qwen3_32b.slurm
# -----------------------------------------------------------------------

set -euo pipefail

export BFCL_MODEL_KEY="qwen3_32b"
export BFCL_HF_MODEL="Qwen/Qwen3-32B"
export BFCL_NUM_GPUS=4

BFCL_SFT_CHECKPOINT="checkpoints/${BFCL_MODEL_KEY}/sft_final"
BFCL_GRPO_CHECKPOINT="checkpoints/${BFCL_MODEL_KEY}/grpo_final"

echo "=========================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Model:      $BFCL_HF_MODEL"
echo "Started:    $(date)"
echo "=========================================="

module load mamba/latest
source activate bfcl_rl

export HF_HOME="/scratch/${USER}/hf_cache"
export TOKENIZERS_PARALLELISM=false

mkdir -p "checkpoints/${BFCL_MODEL_KEY}"

LOG_PREFIX="grpo_pipeline/slurm/logs"
SFT_LOG="${LOG_PREFIX}/sft_${BFCL_MODEL_KEY}_${SLURM_JOB_ID}.out"
GRPO_LOG="${LOG_PREFIX}/grpo_${BFCL_MODEL_KEY}_${SLURM_JOB_ID}.out"
VLLM_LOG="${LOG_PREFIX}/eval_vllm_${BFCL_MODEL_KEY}_${SLURM_JOB_ID}.out"

echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Python:          $(which python)"
echo "PyTorch:         $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available:  $(python -c 'import torch; print(torch.cuda.is_available())')"

# ════════════════════════════════════════════════════════════════════════
# PHASE 1 — SFT
# ════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════ PHASE 1: SFT ══════════════"
echo "Started: $(date)"

python grpo_pipeline/sft_train.py 2>&1 | tee "$SFT_LOG"

echo "SFT complete: $(date)"
echo "Checkpoint: $BFCL_SFT_CHECKPOINT"
echo "Log: $SFT_LOG"

# ════════════════════════════════════════════════════════════════════════
# PHASE 2 — GRPO
# ════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════ PHASE 2: GRPO ═════════════"
echo "Started: $(date)"

if [ ! -d "$BFCL_SFT_CHECKPOINT" ]; then
    echo "ERROR: SFT checkpoint not found at $BFCL_SFT_CHECKPOINT"
    exit 1
fi

python grpo_pipeline/grpo_train.py 2>&1 | tee "$GRPO_LOG"

echo "GRPO complete: $(date)"
echo "Checkpoint: $BFCL_GRPO_CHECKPOINT"
echo "Log: $GRPO_LOG"

# ════════════════════════════════════════════════════════════════════════
# PHASE 3 — EVALUATE
# ════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════ PHASE 3: EVALUATE ═════════"
echo "Started: $(date)"

if [ ! -d "$BFCL_GRPO_CHECKPOINT" ]; then
    echo "ERROR: GRPO checkpoint not found at $BFCL_GRPO_CHECKPOINT"
    exit 1
fi

VLLM_PORT=8001
MERGED_DIR="checkpoints/${BFCL_MODEL_KEY}/merged"
TEST_CATEGORIES="live_simple,live_parallel,live_multiple,live_parallel_multiple"
VLLM_PID=""

trap '[[ -n "$VLLM_PID" ]] && kill "$VLLM_PID" 2>/dev/null || true' EXIT

echo "Merging GRPO LoRA into base model..."
python grpo_pipeline/merge_lora.py \
    "$BFCL_HF_MODEL" \
    "$BFCL_GRPO_CHECKPOINT" \
    "$MERGED_DIR"
echo "Merge complete: $MERGED_DIR"

echo "Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MERGED_DIR" \
    --served-model-name "$BFCL_HF_MODEL" \
    --port "$VLLM_PORT" \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size "$BFCL_NUM_GPUS" \
    > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!

MAX_WAIT=900
ELAPSED=0
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "ERROR: vLLM server did not start within ${MAX_WAIT}s."
        tail -20 "$VLLM_LOG"
        exit 1
    fi
    sleep 10
    ELAPSED=$((ELAPSED + 10))
    echo "  ...waiting (${ELAPSED}s elapsed)"
done
echo "vLLM server ready."

cd berkeley-function-call-leaderboard
REMOTE_OPENAI_BASE_URL="http://localhost:${VLLM_PORT}/v1" \
bfcl generate \
    --model "$BFCL_HF_MODEL" \
    --test-category "$TEST_CATEGORIES" \
    --skip-server-setup \
    --allow-overwrite \
    --num-threads 4
cd ..

kill "$VLLM_PID" 2>/dev/null || true
wait "$VLLM_PID" 2>/dev/null || true
echo "vLLM server stopped."

cd berkeley-function-call-leaderboard
bfcl evaluate \
    --model "$BFCL_HF_MODEL" \
    --test-category "$TEST_CATEGORIES"
cd ..

echo ""
echo "=========================================="
echo "Pipeline complete: $(date)"
echo ""
echo "Phase logs:"
echo "  SFT:      $SFT_LOG"
echo "  GRPO:     $GRPO_LOG"
echo "  vLLM:     $VLLM_LOG"
echo "Scores:     berkeley-function-call-leaderboard/score/${BFCL_HF_MODEL//\//_}/"
echo "Checkpoint: $BFCL_GRPO_CHECKPOINT"
echo "=========================================="
