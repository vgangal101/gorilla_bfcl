#!/usr/bin/env bash
# setup_env.sh — One-time environment setup for the GRPO training pipeline.
# Run this once on a login node before submitting any SLURM jobs.
#
# Usage:
#   bash grpo_pipeline/slurm/setup_env.sh

set -euo pipefail

module load mamba/latest

# Create dedicated conda env for RL training (separate from bfcl_gaudi)
if ! conda env list | grep -q "bfcl_rl"; then
    echo "Creating bfcl_rl conda environment..."
    mamba create -n bfcl_rl python=3.11 -y
fi

conda activate bfcl_rl

echo "Installing BFCL harness..."
pip install -e berkeley-function-call-leaderboard/

echo "Installing RL training dependencies..."
pip install \
    "transformers>=4.45" \
    "trl>=0.12" \
    "datasets>=2.20" \
    "peft>=0.12" \
    "accelerate>=0.34" \
    "bitsandbytes>=0.43" \
    "huggingface_hub>=0.24"

echo "Done. Activate with: conda activate bfcl_rl"
