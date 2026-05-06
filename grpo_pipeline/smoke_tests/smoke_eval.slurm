#!/usr/bin/env bash
#SBATCH --job-name=bfcl_rl_smoke_eval
#SBATCH --account=class_eee515598spring2026
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=grpo_pipeline/slurm/logs/smoke_eval_%j.out
#SBATCH --error=grpo_pipeline/slurm/logs/smoke_eval_%j.err
#
# Smoke test — Phase 3 only: merge LoRA → serve with vLLM → bfcl generate → bfcl evaluate.
#
# Run this standalone when Phase 1 (SFT) and Phase 2 (GRPO) have already
# completed and checkpoints/qwen3_8b/smoke_grpo_final exists.
# Avoids re-running the 20+ minute training phases just to retry eval.
#
# Submit from repo root:
#   sbatch grpo_pipeline/smoke_tests/smoke_eval.slurm
#
# What to watch for:
#   1. Merge step  — should complete in ~2 min. If it OOMs, the GPU has
#                    residual memory from a prior run; wait or requeue.
#   2. vLLM start  — watch smoke_eval_vllm_<JOBID>.out. Healthy startup
#                    ends with "Application startup complete." Takes ~4 min
#                    (model load ~30s + torch.compile ~64s + CUDA warmup).
#   3. bfcl generate — 5 test cases. Should finish in under 1 min once
#                      vLLM is up. Check result/Qwen_Qwen3-8B/ for output files.
#   4. bfcl evaluate — pure CPU, fast. Check the score printed at the end.
#                      Any score > 0.0 means the model produced at least one
#                      valid function call — the pipeline is working.
#                      Don't expect high accuracy from a 2-step smoke GRPO run.
#
# Score file after completion:
#   berkeley-function-call-leaderboard/score/Qwen_Qwen3-8B/non_live/BFCL_v4_simple_python_score.json
# -----------------------------------------------------------------------

set -euo pipefail

export BFCL_MODEL_KEY="qwen3_8b"
export BFCL_HF_MODEL="Qwen/Qwen3-8B"

SMOKE_GRPO_CHECKPOINT="checkpoints/${BFCL_MODEL_KEY}/smoke_grpo_final"
SMOKE_MERGED_DIR="checkpoints/${BFCL_MODEL_KEY}/smoke_merged"
VLLM_PORT=8002
VLLM_PID=""
LOG_PREFIX="grpo_pipeline/slurm/logs"
VLLM_LOG="${LOG_PREFIX}/smoke_eval_vllm_${SLURM_JOB_ID}.out"

echo "========================================"
echo "BFCL RL — Smoke Eval (Phase 3 only)"
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Started:  $(date)"
echo "========================================"

# ── Environment ─────────────────────────────────────────────────────────
module load mamba/latest
source activate bfcl_rl

export HF_HOME="/scratch/${USER}/hf_cache"
export TOKENIZERS_PARALLELISM=false

echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Python: $(which python)"

# ── Verify checkpoint exists ─────────────────────────────────────────────
if [ ! -d "$SMOKE_GRPO_CHECKPOINT" ]; then
    echo "ERROR: GRPO checkpoint not found at $SMOKE_GRPO_CHECKPOINT"
    echo "  Run smoke_test.slurm (or smoke_sft.py + smoke_grpo.py) first."
    exit 1
fi

trap '[[ -n "$VLLM_PID" ]] && kill "$VLLM_PID" 2>/dev/null || true' EXIT

# ════════════════════════════════════════════════════════════════════════
# STEP 1 — MERGE LoRA into base model
# ════════════════════════════════════════════════════════════════════════
echo ""
echo "══ STEP 1: MERGE ══"
echo "Started: $(date)"

python grpo_pipeline/merge_lora.py \
    "$BFCL_HF_MODEL" \
    "$SMOKE_GRPO_CHECKPOINT" \
    "$SMOKE_MERGED_DIR"

echo "Merge complete: $(date)"

# ════════════════════════════════════════════════════════════════════════
# STEP 2 — START vLLM
# ════════════════════════════════════════════════════════════════════════
echo ""
echo "══ STEP 2: vLLM ══"
echo "Started: $(date)"
echo "vLLM log: $VLLM_LOG"

python -m vllm.entrypoints.openai.api_server \
    --model "$SMOKE_MERGED_DIR" \
    --served-model-name "$BFCL_HF_MODEL" \
    --port "$VLLM_PORT" \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.85 \
    > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!

MAX_WAIT=600
ELAPSED=0
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "ERROR: vLLM did not start within ${MAX_WAIT}s. Last 20 lines:"
        tail -20 "$VLLM_LOG"
        exit 1
    fi
    sleep 10
    ELAPSED=$((ELAPSED + 10))
    echo "  ...waiting (${ELAPSED}s)"
done
echo "vLLM ready: $(date)"

# ════════════════════════════════════════════════════════════════════════
# STEP 3 — BFCL GENERATE (5 test cases)
# ════════════════════════════════════════════════════════════════════════
echo ""
echo "══ STEP 3: BFCL GENERATE ══"
echo "Started: $(date)"

cat > berkeley-function-call-leaderboard/test_case_ids_to_generate.json <<'EOF'
{"simple_python": ["simple_python_0", "simple_python_1", "simple_python_2", "simple_python_3", "simple_python_4"]}
EOF

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

kill "$VLLM_PID" 2>/dev/null || true
wait "$VLLM_PID" 2>/dev/null || true
echo "vLLM stopped."

rm -f berkeley-function-call-leaderboard/test_case_ids_to_generate.json

# ════════════════════════════════════════════════════════════════════════
# STEP 4 — BFCL EVALUATE
# ════════════════════════════════════════════════════════════════════════
echo ""
echo "══ STEP 4: BFCL EVALUATE ══"
echo "Started: $(date)"

cd berkeley-function-call-leaderboard
bfcl evaluate \
    --model "$BFCL_HF_MODEL" \
    --test-category simple_python \
    --partial-eval
cd ..

echo ""
echo "========================================"
echo "Smoke eval COMPLETE: $(date)"
echo ""
echo "Logs:"
echo "  vLLM:   $VLLM_LOG"
echo "  Scores: berkeley-function-call-leaderboard/score/${BFCL_HF_MODEL//\//_}/non_live/BFCL_v4_simple_python_score.json"
echo ""
echo "What a passing result looks like:"
echo "  - No errors in this log"
echo "  - result/Qwen_Qwen3-8B/ contains BFCL_v3_simple_python_result.json"
echo "  - Score file exists and shows accuracy > 0.0"
echo "  - High accuracy is NOT expected from a 2-step GRPO smoke run"
echo "    (the point is the pipeline runs end-to-end, not that it learns well)"
echo "========================================"
