# Qwen3-32B Multi-Turn Evaluation on SOL

This guide explains how to evaluate the Qwen3-32B model on BFCL's multi-turn function calling tasks using vLLM with **prompt-based function calling**.

## Model Configuration

- **Model**: `Qwen/Qwen3-32B` (regular model, not FC variant)
- **Function Calling**: Enabled via prompting using Qwen's chat template
- **Handler**: `QwenHandler` (prompt-based function calling)
- **Test Category**: `multi_turn` (all multi-turn function calling tests)

### Customization

Edit `evaluate_qwen3_32b_vllm.sh` to modify:

```bash
# Change test category (line ~41)
TEST_CATEGORY="multi_turn_base"  # or: multi_turn_miss_func, multi_turn_miss_param, etc.

# Adjust GPU configuration (line ~29-30)
#SBATCH --gres=gpu:1  # Use single GPU
VLLM_TENSOR_PARALLEL_SIZE=1

# Increase memory for larger batch sizes (line ~30)
#SBATCH --mem=128G
NUM_THREADS=8
```

### Available Test Categories (Multi-Turn)

- `multi_turn` - All multi-turn tests
- `multi_turn_base` - Base multi-turn function calls
- `multi_turn_miss_func` - Multi-turn with missing functions
- `multi_turn_miss_param` - Multi-turn with missing parameters
- `multi_turn_long_context` - Long context multi-turn calls

## What the Script Does

### Phase 0: vLLM Server Setup
- Loads mamba/conda environment
- Installs BFCL with vLLM backend
- Starts vLLM server hosting Qwen3-32B-FC
- Waits for server health check

### Phase 1: Generation
- Sends BFCL test cases to vLLM server
- Generates function calls for each test
- Stores results in `result/Qwen/Qwen3-32B-FC/`

### Phase 2: Evaluation
- Evaluates generated function calls for correctness
- Compares against ground truth
- Stores scores in `score/`

### Phase 3: Results
- Displays overall accuracy
- Breaks down scores by category
- Generates summary report

## Output Files

After execution, check:

```bash
# Main evaluation results
cat logs/evaluation_*.log

# vLLM server logs (for debugging)
tail -50 logs/vllm_server_*.log

# Summary report
cat logs/evaluation_report_*.txt

# Detailed evaluation scores
head -20 score/data_overall.csv
head -20 score/data_multi_turn.csv

# Generated responses
ls -la result/Qwen/Qwen3-32B/
```

## Memory and GPU Requirements

- **GPU Memory**: ~32-40GB (with tensor parallelism on 2x GPUs)
- **System RAM**: 64GB minimum
- **Duration**: 30-120 minutes (depending on test size)

### Adjust for Different GPU Counts

**Single GPU (V100/A100 with 80GB):**
```bash
#SBATCH --gres=gpu:1
VLLM_TENSOR_PARALLEL_SIZE=1
NUM_THREADS=2
#SBATCH --mem=64G
```

**Multiple GPUs (2x A100):**
```bash
#SBATCH --gres=gpu:2
VLLM_TENSOR_PARALLEL_SIZE=2
NUM_THREADS=4
#SBATCH --mem=128G
```

## Troubleshooting

### vLLM Server Fails to Start
```bash
# Check GPU availability
nvidia-smi

# Check mamba environment
conda activate bfcl_vllm
pip list | grep vllm

# Check logs
tail logs/vllm_server_*.log
```

### Generation Phase Hangs
- Ensure vLLM server is running: `curl http://127.0.0.1:8000/v1/models`
- Check for port conflicts: `lsof -i :8000`
- Increase timeout in script if network is slow

### Out of Memory Errors
- Reduce `VLLM_GPU_MEMORY_UTILIZATION` to 0.8 or 0.7
- Reduce `NUM_THREADS` to 2
- Use single GPU with `VLLM_TENSOR_PARALLEL_SIZE=1`

### Import Errors
Ensure vLLM backend is installed:
```bash
conda activate bfcl_vllm
pip install -e .[oss_eval_vllm]
```

## Multi-Turn Evaluation Details

The multi-turn category tests:

1. **Base Multi-Turn** (`multi_turn_base`): Sequential function calls where later calls depend on earlier results
2. **Missing Functions** (`multi_turn_miss_func`): Evaluate handling of unavailable functions
3. **Missing Parameters** (`multi_turn_miss_param`): Evaluate parameter selection when some are unavailable
4. **Long Context** (`multi_turn_long_context`): Test performance with extended conversation history

## Tips for Better Results

1. **Warm up vLLM**: First generation takes longer as KV cache initializes
2. **Monitor GPU**: Use `watch nvidia-smi` in separate terminal
3. **Check logs frequently**: Don't wait for full completion to spot issues
4. **Use smaller category first**: Run `multi_turn_base` before full `multi_turn` suite

## SOL-Specific Notes

Replace `gpu_type` constraint based on available GPUs:
```bash
#SBATCH --constraint="gpu_type:a100"    # A100s
#SBATCH --constraint="gpu_type:v100"    # V100s
#SBATCH --constraint="gpu_type:h100"    # H100s
```

## Related Commands

```bash
# Check job status
squeue -j <job_id>

# Cancel running job
scancel <job_id>

# View resource usage after completion
sacct -j <job_id> --format=JobID,AllocCPUS,MaxRSS,Elapsed

# List completed jobs
sacct --starttime=2024-01-01 -u $USER
```

## Next Steps

1. Review evaluation results
2. Compare with other models in `score/data_overall.csv`
3. Analyze specific failure cases
4. Adjust prompts or model parameters if needed
5. Re-run with different test categories as needed

## Support

For BFCL-specific issues:
- Check [Berkeley Function Call Leaderboard README](../berkeley-function-call-leaderboard/README.md)
- Check [BFCL Quick Start](../BFCL_QUICK_START.md)

For vLLM issues:
- [vLLM Documentation](https://docs.vllm.ai/)
- Check vLLM logs at `logs/vllm_server_*.log`
