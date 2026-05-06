#!/usr/bin/env bash
#SBATCH --job-name=bfcl_sft
#SBATCH --account=class_cse59818158spring2026
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=grpo_pipeline/slurm/logs/sft_%j.out
#SBATCH --error=grpo_pipeline/slurm/logs/sft_%j.err
#
# SFT Phase — supervised warm-start on BFCL ground-truth examples.
# Must complete before submitting grpo_train.slurm.
#
# Submit from repo root:
#   sbatch grpo_pipeline/slurm/sft_train.slurm
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
REPO_ROOT="$(pwd)"
LOG_DIR="grpo_pipeline/slurm/logs"
mkdir -p "$LOG_DIR" checkpoints/sft

# ── Sanity check ────────────────────────────────────────────────────────
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# ── Run SFT ─────────────────────────────────────────────────────────────
echo ""
echo "Starting SFT..."
python grpo_pipeline/sft_train.py

echo ""
echo "=========================================="
echo "SFT complete: $(date)"
echo "Checkpoint:   checkpoints/sft_final/"
echo "Next step:    sbatch grpo_pipeline/slurm/grpo_train.slurm"
echo "=========================================="
