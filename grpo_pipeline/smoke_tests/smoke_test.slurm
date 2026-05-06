#!/usr/bin/env bash
#SBATCH --job-name=bfcl_rl_smoke
#SBATCH --account=class_eee515598spring2026
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=grpo_pipeline/slurm/logs/smoke_%j.out
#SBATCH --error=grpo_pipeline/slurm/logs/smoke_%j.err
#
# Smoke test — runs the full pipeline (SFT → GRPO → merge → vLLM → eval)
# with minimal settings to verify each phase works before a full run.
#
#   Model:       Qwen/Qwen3-8B (smallest)
#   SFT:         4 examples, 3 steps
#   GRPO:        4 examples, 2 steps, K=2 rollouts
#   Evaluation:  simple_python only, 5 test cases
#   Walltime:    2 hours (vs 24h for a full run)
#
# Submit from repo root:
#   sbatch grpo_pipeline/smoke_tests/smoke_test.slurm
#
# Check results:
#   tail -f grpo_pipeline/slurm/logs/smoke_<JOBID>.out
# -----------------------------------------------------------------------

set -euo pipefail

export BFCL_MODEL_KEY="qwen3_8b"
export BFCL_HF_MODEL="Qwen/Qwen3-8B"
export BFCL_NUM_GPUS=1

echo "========================================"
echo "BFCL RL — SLURM Smoke Test"
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Started:  $(date)"
echo "========================================"

# ── Environment ─────────────────────────────────────────────────────────
module load mamba/latest
source activate bfcl_rl

export HF_HOME="/scratch/${USER}/hf_cache"
export TOKENIZERS_PARALLELISM=false

mkdir -p "checkpoints/${BFCL_MODEL_KEY}"

LOG_PREFIX="grpo_pipeline/slurm/logs"
SMOKE_SFT_LOG="${LOG_PREFIX}/smoke_sft_${SLURM_JOB_ID}.out"
SMOKE_GRPO_LOG="${LOG_PREFIX}/smoke_grpo_${SLURM_JOB_ID}.out"
VLLM_LOG="${LOG_PREFIX}/smoke_eval_vllm_${SLURM_JOB_ID}.out"

echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Python: $(which python)"

# ════════════════════════════════════════════════════════════════════════
# PHASE 1 — SFT (4 examples, 3 steps)
# ════════════════════════════════════════════════════════════════════════
echo ""
echo "══ SMOKE PHASE 1: SFT (3 steps) ══"
echo "Started: $(date)"

python grpo_pipeline/smoke_tests/smoke_sft.py 2>&1 | tee "$SMOKE_SFT_LOG"

echo "SFT smoke done: $(date)"
echo "Log: $SMOKE_SFT_LOG"

# ════════════════════════════════════════════════════════════════════════
# PHASE 2 — GRPO (4 examples, 2 steps, K=2)
# ════════════════════════════════════════════════════════════════════════
echo ""
echo "══ SMOKE PHASE 2: GRPO (2 steps, K=2) ══"
echo "Started: $(date)"

python grpo_pipeline/smoke_tests/smoke_grpo.py 2>&1 | tee "$SMOKE_GRPO_LOG"

echo "GRPO smoke done: $(date)"
echo "Log: $SMOKE_GRPO_LOG"

# ════════════════════════════════════════════════════════════════════════
# PHASE 3 — MERGE + vLLM + BFCL EVAL (5 test cases, simple_python only)
# ════════════════════════════════════════════════════════════════════════
echo ""
echo "══ SMOKE PHASE 3: EVALUATE ══"
echo "Started: $(date)"

SMOKE_GRPO_CHECKPOINT="checkpoints/${BFCL_MODEL_KEY}/smoke_grpo_final"
SMOKE_MERGED_DIR="checkpoints/${BFCL_MODEL_KEY}/smoke_merged"
VLLM_PORT=8002         # different port from full pipeline to avoid conflicts
VLLM_PID=""

trap '[[ -n "$VLLM_PID" ]] && kill "$VLLM_PID" 2>/dev/null || true' EXIT

# Merge
echo "Merging smoke GRPO checkpoint..."
python grpo_pipeline/merge_lora.py \
    "$BFCL_HF_MODEL" \
    "$SMOKE_GRPO_CHECKPOINT" \
    "$SMOKE_MERGED_DIR"

# Start vLLM
echo "Starting vLLM server on port $VLLM_PORT..."
python -m vllm.entrypoints.openai.api_server \
    --model "$SMOKE_MERGED_DIR" \
    --served-model-name "$BFCL_HF_MODEL" \
    --port "$VLLM_PORT" \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.85 \
    > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!

# Wait for server ready
MAX_WAIT=600
ELAPSED=0
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "ERROR: vLLM server did not start within ${MAX_WAIT}s."
        tail -20 "$VLLM_LOG"
        exit 1
    fi
    sleep 10
    ELAPSED=$((ELAPSED + 10))
    echo "  ...waiting (${ELAPSED}s)"
done
echo "vLLM server ready."

# Populate test_case_ids_to_generate.json with 5 simple_python IDs
cat > berkeley-function-call-leaderboard/test_case_ids_to_generate.json <<'EOF'
["simple_python_0", "simple_python_1", "simple_python_2", "simple_python_3", "simple_python_4"]
EOF

# bfcl generate — 5 test cases only
cd berkeley-function-call-leaderboard
REMOTE_OPENAI_BASE_URL="http://localhost:${VLLM_PORT}/v1" \
bfcl generate \
    --model "$BFCL_HF_MODEL" \
    --test-category simple_python \
    --skip-server-setup \
    --allow-overwrite \
    --run-ids \
    --num-threads 2
cd ..

# Stop vLLM
kill "$VLLM_PID" 2>/dev/null || true
wait "$VLLM_PID" 2>/dev/null || true
echo "vLLM stopped."

# Clean up test IDs file
rm -f berkeley-function-call-leaderboard/test_case_ids_to_generate.json

# bfcl evaluate
cd berkeley-function-call-leaderboard
bfcl evaluate \
    --model "$BFCL_HF_MODEL" \
    --test-category simple_python \
    --partial-eval
cd ..

echo ""
echo "========================================"
echo "Smoke test COMPLETE: $(date)"
echo "All 3 phases passed."
echo ""
echo "Phase logs:"
echo "  SFT:      $SMOKE_SFT_LOG"
echo "  GRPO:     $SMOKE_GRPO_LOG"
echo "  vLLM:     $VLLM_LOG"
echo "Scores:     berkeley-function-call-leaderboard/score/${BFCL_HF_MODEL//\//_}/"
echo "========================================"
echo ""
echo "If all phases completed without error, the full pipeline is safe to run:"
echo "  bash grpo_pipeline/slurm/submit.sh qwen3_8b"
