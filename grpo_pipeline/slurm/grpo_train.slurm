#!/usr/bin/env bash
#SBATCH --job-name=bfcl_grpo
#SBATCH --account=class_cse59818158spring2026
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=grpo_pipeline/slurm/logs/grpo_%j.out
#SBATCH --error=grpo_pipeline/slurm/logs/grpo_%j.err
#
# GRPO Phase — reinforcement learning with BFCL AST checker as reward.
# Requires checkpoints/sft_final/ to exist (run sft_train.slurm first).
#
# Submit from repo root:
#   sbatch grpo_pipeline/slurm/grpo_train.slurm
#
# Or chain automatically after SFT:
#   sbatch --dependency=afterok:<SFT_JOB_ID> grpo_pipeline/slurm/grpo_train.slurm
# -----------------------------------------------------------------------

set -euo pipefail

echo "=========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Started:   $(date)"
echo "=========================================="

# ── Environment ────────────────────────────────────────────────────────
module load mamba/latest
conda activate bfcl_rl

export HF_HOME="/scratch/${USER}/hf_cache"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

# ── Paths ───────────────────────────────────────────────────────────────
LOG_DIR="grpo_pipeline/slurm/logs"
mkdir -p "$LOG_DIR" checkpoints/grpo

# ── Guard: SFT checkpoint must exist ────────────────────────────────────
if [ ! -d "checkpoints/sft_final" ]; then
    echo "ERROR: checkpoints/sft_final/ not found."
    echo "Run sft_train.slurm first, or chain with --dependency=afterok:<SFT_JOB_ID>"
    exit 1
fi

# ── Sanity check ────────────────────────────────────────────────────────
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "SFT checkpoint: $(ls checkpoints/sft_final/)"

# ── Run GRPO ─────────────────────────────────────────────────────────────
echo ""
echo "Starting GRPO..."
python grpo_pipeline/grpo_train.py

echo ""
echo "=========================================="
echo "GRPO complete: $(date)"
echo "Checkpoint:    checkpoints/grpo_final/"
echo "Next step:     run bfcl evaluate (see grpo_pipeline/README.md Step 6)"
echo "=========================================="
