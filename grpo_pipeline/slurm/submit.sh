#!/usr/bin/env bash
# submit.sh — Submit the full RL pipeline for a given model.
#
# Usage:
#   bash grpo_pipeline/slurm/submit.sh <model_key>
#
# Model keys:
#   qwen3_8b    — Qwen/Qwen3-8B        (1×A100,  80G,  24h)
#   qwen3_14b   — Qwen/Qwen3-14B       (2×A100, 160G,  48h)
#   qwen3_32b   — Qwen/Qwen3-32B       (4×A100, 320G, 120h)
#   gemma4_31b  — google/gemma-4-31B-it (4×A100, 320G, 120h)
#
# Examples:
#   bash grpo_pipeline/slurm/submit.sh qwen3_8b
#   bash grpo_pipeline/slurm/submit.sh qwen3_14b

set -euo pipefail

MODEL_KEY="${1:-}"
if [ -z "$MODEL_KEY" ]; then
    echo "Usage: bash grpo_pipeline/slurm/submit.sh <model_key>"
    echo "Keys:  qwen3_8b | qwen3_14b | qwen3_32b | gemma4_31b"
    exit 1
fi

# ── Model config table ───────────────────────────────────────────────────
case "$MODEL_KEY" in
    qwen3_8b)
        HF_MODEL="Qwen/Qwen3-8B"
        NUM_GPUS=1
        GRES="gpu:a100:1"
        MEM="80G"
        CPUS=16
        TIME="24:00:00"
        ;;
    qwen3_14b)
        HF_MODEL="Qwen/Qwen3-14B"
        NUM_GPUS=2
        GRES="gpu:a100:2"
        MEM="160G"
        CPUS=32
        TIME="48:00:00"
        ;;
    qwen3_32b)
        HF_MODEL="Qwen/Qwen3-32B"
        NUM_GPUS=4
        GRES="gpu:a100:4"
        MEM="320G"
        CPUS=64
        TIME="120:00:00"
        ;;
    gemma4_31b)
        HF_MODEL="google/gemma-4-31B-it"
        NUM_GPUS=4
        GRES="gpu:a100:4"
        MEM="320G"
        CPUS=64
        TIME="120:00:00"
        ;;
    *)
        echo "ERROR: Unknown model key '$MODEL_KEY'"
        echo "Keys:  qwen3_8b | qwen3_14b | qwen3_32b | gemma4_31b"
        exit 1
        ;;
esac

# ── Confirm ──────────────────────────────────────────────────────────────
echo "Submitting pipeline for: $MODEL_KEY"
echo "  HF model:  $HF_MODEL"
echo "  GPUs:      $GRES"
echo "  Memory:    $MEM"
echo "  CPUs:      $CPUS"
echo "  Walltime:  $TIME"
echo ""

JOB_ID=$(sbatch \
    --job-name="bfcl_rl_${MODEL_KEY}" \
    --gres="$GRES" \
    --mem="$MEM" \
    --cpus-per-task="$CPUS" \
    --time="$TIME" \
    --export="ALL,BFCL_MODEL_KEY=${MODEL_KEY},BFCL_HF_MODEL=${HF_MODEL},BFCL_NUM_GPUS=${NUM_GPUS}" \
    --parsable \
    grpo_pipeline/slurm/run_pipeline.slurm)

echo "Submitted job $JOB_ID"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f grpo_pipeline/slurm/logs/bfcl_rl_${MODEL_KEY}_${JOB_ID}.out"
