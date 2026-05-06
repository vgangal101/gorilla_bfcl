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

source activate bfcl_rl

echo "Installing BFCL harness + vLLM + RL training dependencies..."
# Single pip call so the resolver sees all constraints together.
# Separate calls risk vllm==0.8.5's pinned deps being silently upgraded
# by the second install, breaking vllm at runtime.
pip install \
    -e "berkeley-function-call-leaderboard/[oss_eval_vllm]" \
    "trl>=0.12" \
    "datasets>=2.20" \
    "peft>=0.12" \
    "accelerate>=0.34" \
    "bitsandbytes>=0.43"

# Create log directory so SLURM can write output files on job start
mkdir -p grpo_pipeline/slurm/logs

echo "Done. Activate with: source activate bfcl_rl"
