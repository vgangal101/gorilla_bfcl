#!/bin/bash
#SBATCH --job-name=bfcl_qwen32b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:2
#SBATCH --time=08:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=vgangal3@asu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

module load mamba/latest
source activate bfcl
conda activate bfcl

cd /Users/vgangal/phd_research_workspace/gorilla_bfcl/berkeley-function-call-leaderboard

export TORCHDYNAMO_DISABLE=1

# Generate responses for multi-turn and agentic categories
bfcl generate --model Qwen/Qwen3-32B --test-category multi_turn,agentic --backend vllm --num-gpus 2

# Evaluate results
bfcl evaluate --model Qwen/Qwen3-32B --test-category multi_turn,agentic
