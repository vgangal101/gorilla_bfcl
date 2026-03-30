#!/bin/bash
#SBATCH --job-name=bfcl_qwen3_32b
#SBATCH --output=logs/bfcl_qwen3_32b_%j.out
#SBATCH --error=logs/bfcl_qwen3_32b_%j.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint="gpu_type:a100"

# Submit to SOL: sbatch evaluate_qwen3_32b_vllm.slurm

./evaluate_qwen3_32b_vllm.sh
