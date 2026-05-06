#!/usr/bin/env bash
#SBATCH --job-name=bfcl_rl
#SBATCH --account=class_cse59818158spring2026
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=grpo_pipeline/slurm/logs/%x_%j.out
#SBATCH --error=grpo_pipeline/slurm/logs/%x_%j.err
#
# Unified RL pipeline: SFT → GRPO → Evaluate (all phases in one job).
# Resource flags (--gres, --mem, --time, --cpus-per-task) are NOT set here —
# they are injected by submit.sh based on the chosen model.
#
# Do NOT submit this script directly. Use:
#   bash grpo_pipeline/slurm/submit.sh <model_key>
#
# Model keys: qwen3_8b | qwen3_14b | qwen3_32b | gemma4_31b
# -----------------------------------------------------------------------

set -euo pipefail

# ── Validate required env vars (set by submit.sh via --export) ──────────
: "${BFCL_MODEL_KEY:?  ERROR: BFCL_MODEL_KEY not set. Use submit.sh.}"
: "${BFCL_HF_MODEL:?   ERROR: BFCL_HF_MODEL not set. Use submit.sh.}"
: "${BFCL_NUM_GPUS:?   ERROR: BFCL_NUM_GPUS not set. Use submit.sh.}"

BFCL_SFT_CHECKPOINT="checkpoints/${BFCL_MODEL_KEY}/sft_final"
BFCL_GRPO_CHECKPOINT="checkpoints/${BFCL_MODEL_KEY}/grpo_final"
export BFCL_MODEL_KEY BFCL_HF_MODEL BFCL_SFT_CHECKPOINT

echo "=========================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Model key:  $BFCL_MODEL_KEY"
echo "HF model:   $BFCL_HF_MODEL"
echo "GPUs:       $BFCL_NUM_GPUS"
echo "Started:    $(date)"
echo "=========================================="

# ── Environment ─────────────────────────────────────────────────────────
module load mamba/latest
source activate bfcl_rl

export HF_HOME="/scratch/${USER}/hf_cache"
export TOKENIZERS_PARALLELISM=false

mkdir -p "checkpoints/${BFCL_MODEL_KEY}"

LOG_PREFIX="grpo_pipeline/slurm/logs"
SFT_LOG="${LOG_PREFIX}/sft_${BFCL_MODEL_KEY}_${SLURM_JOB_ID}.out"
GRPO_LOG="${LOG_PREFIX}/grpo_${BFCL_MODEL_KEY}_${SLURM_JOB_ID}.out"
VLLM_LOG="${LOG_PREFIX}/eval_vllm_${BFCL_MODEL_KEY}_${SLURM_JOB_ID}.out"

# ── Sanity check ─────────────────────────────────────────────────────────
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Python:          $(which python)"
echo "PyTorch:         $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available:  $(python -c 'import torch; print(torch.cuda.is_available())')"

# ── Helper: launch with accelerate for multi-GPU, plain python for single ─
run_training() {
    local script="$1"
    if [ "$BFCL_NUM_GPUS" -gt 1 ]; then
        accelerate launch --num_processes "$BFCL_NUM_GPUS" "$script"
    else
        python "$script"
    fi
}

# ════════════════════════════════════════════════════════════════════════
# PHASE 1 — SFT
# ════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════ PHASE 1: SFT ══════════════"
echo "Started: $(date)"

run_training grpo_pipeline/sft_train.py 2>&1 | tee "$SFT_LOG"

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

run_training grpo_pipeline/grpo_train.py 2>&1 | tee "$GRPO_LOG"

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

# Ensure vLLM is killed on any exit (success, error, or signal).
trap '[[ -n "$VLLM_PID" ]] && kill "$VLLM_PID" 2>/dev/null || true' EXIT

# Merge LoRA adapter into base model weights before serving.
# Avoids a vLLM name collision bug where --lora-modules alias equals the
# base model name, causing ambiguous routing or startup errors.
echo "Merging GRPO LoRA into base model..."
python grpo_pipeline/merge_lora.py \
    "$BFCL_HF_MODEL" \
    "$BFCL_GRPO_CHECKPOINT" \
    "$MERGED_DIR"
echo "Merge complete: $MERGED_DIR"

# Start vLLM server on the merged model.
# --served-model-name makes vLLM answer to the registered BFCL model name
# so bfcl generate --model $BFCL_HF_MODEL routes to the GRPO-trained weights.
echo "Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MERGED_DIR" \
    --served-model-name "$BFCL_HF_MODEL" \
    --port "$VLLM_PORT" \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size "$BFCL_NUM_GPUS" \
    > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# Wait for server ready
echo "Waiting for vLLM server..."
MAX_WAIT=900
ELAPSED=0
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "ERROR: vLLM server did not start within ${MAX_WAIT}s."
        tail -20 "$VLLM_LOG"
        kill "$VLLM_PID" 2>/dev/null || true
        exit 1
    fi
    sleep 10
    ELAPSED=$((ELAPSED + 10))
    echo "  ...waiting (${ELAPSED}s elapsed)"
done
echo "vLLM server ready."

# bfcl generate — model name matches --served-model-name so vLLM routes
# requests to the merged GRPO checkpoint.
# --allow-overwrite is required: without it, bfcl generate silently skips
# any test IDs where result files already exist (e.g. from a baseline run)
# and mixes baseline outputs with GRPO outputs in the same result directory.
cd berkeley-function-call-leaderboard
REMOTE_OPENAI_BASE_URL="http://localhost:${VLLM_PORT}/v1" \
bfcl generate \
    --model "$BFCL_HF_MODEL" \
    --test-category "$TEST_CATEGORIES" \
    --skip-server-setup \
    --allow-overwrite \
    --num-threads 4
cd ..

# Stop vLLM
kill "$VLLM_PID" 2>/dev/null || true
wait "$VLLM_PID" 2>/dev/null || true
echo "vLLM server stopped."

# bfcl evaluate
cd berkeley-function-call-leaderboard
bfcl evaluate \
    --model "$BFCL_HF_MODEL" \
    --test-category "$TEST_CATEGORIES"
cd ..

echo ""
echo "=========================================="
echo "Pipeline complete: $(date)"
echo "Model:       $BFCL_MODEL_KEY"
echo ""
echo "Phase logs:"
echo "  SFT:      $SFT_LOG"
echo "  GRPO:     $GRPO_LOG"
echo "  vLLM:     $VLLM_LOG"
echo "Scores:     berkeley-function-call-leaderboard/score/${BFCL_HF_MODEL//\//_}/"
echo "Checkpoint: $BFCL_GRPO_CHECKPOINT"
echo "=========================================="
