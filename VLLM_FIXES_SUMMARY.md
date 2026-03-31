# vLLM Server Setup - Critical Fixes Summary

## Issues Identified

Your report that "the vllm server never got setup" revealed three critical bugs in the evaluation script:

### 1. ❌ Missing vLLM Package Installation
**Problem**: The vLLM binary was never installed in the mamba environment
- Script created the mamba environment but didn't install `vllm` package
- This caused `mamba run -n bfcl_vllm vllm serve` to fail with "command not found"

**Fix Applied** (Line 164):
```bash
mamba run -n "$env_name" pip install vllm || error_exit "Failed to install vllm"
```

### 2. ❌ vLLM Server Running Outside Mamba Environment
**Problem**: The `vllm serve` command was executed in the host shell, not in the mamba environment
- Even if vllm was installed, it wouldn't be found
- PyTorch/CUDA dependencies wouldn't be available

**Fix Applied** (Line 229):
```bash
# BEFORE: vllm serve "$VLLM_MODEL" ...
# AFTER:
mamba run -n "$MAMBA_ENV_NAME" vllm serve "$VLLM_MODEL" \
    --host "$VLLM_HOST" \
    --port "$VLLM_PORT" \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE" \
    --trust-remote-code \
    > "$VLLM_LOG_FILE" 2>&1 &
```

### 3. ❌ BFCL Commands Running Outside Mamba Environment  
**Problem**: `bfcl generate` and `bfcl evaluate` commands also weren't running in the mamba environment
- BFCL CLI wouldn't be found if installed in isolated environment
- Dependencies wouldn't resolve correctly

**Fix Applied** (Lines 256, 278):
```bash
# Generate phase
mamba run -n "$MAMBA_ENV_NAME" bfcl generate \
    --model "$MODEL_NAME" \
    --test-category "$TEST_CATEGORY" \
    --num-threads "$NUM_THREADS" \
    --openai-api-base "$VLLM_ENDPOINT"

# Evaluate phase  
mamba run -n "$MAMBA_ENV_NAME" bfcl evaluate \
    --model "$MODEL_NAME" \
    --test-category "$TEST_CATEGORY"
```

## Root Cause Analysis

The issue stems from **bash function scope limitations**:

```bash
# This does NOT persist environment activation across function scope:
setup_environment() {
    mamba create -n bfcl_vllm ...
    source activate bfcl_vllm           # ← Only active WITHIN this function
}

setup_environment
python script.py                         # ← python runs in HOST environment, not mamba env!
```

**Solution**: Use `mamba run -n <env_name> COMMAND` pattern instead:

```bash
setup_environment "bfcl_vllm"
MAMBA_ENV_NAME="bfcl_vllm"

# This ALWAYS runs in the specified environment:
mamba run -n "$MAMBA_ENV_NAME" python script.py
```

## Scope of Changes

| Component | Issue | Fix |
|-----------|-------|-----|
| `setup_environment()` | No vllm installed | Added `pip install vllm` |
| `setup_environment()` | PyTorch missing | Added `mamba install pytorch::pytorch pytorch::pytorch-cuda=12.1` |
| vLLM Server Launch | Wrong scope | Wrapped with `mamba run -n "$MAMBA_ENV_NAME"` |
| BFCL Generate | Wrong scope | Wrapped with `mamba run -n "$MAMBA_ENV_NAME"` |
| BFCL Evaluate | Wrong scope | Wrapped with `mamba run -n "$MAMBA_ENV_NAME"` |

## Verification Steps

Before submitting to SOL, verify locally:

```bash
# Check vllm is installed in the created environment:
mamba run -n bfcl_vllm pip show vllm

# Check vllm command is available:
mamba run -n bfcl_vllm vllm --version

# Check BFCL is available:
mamba run -n bfcl_vllm bfcl --version
```

## Next Steps

1. Submit script: `sbatch evaluate_qwen3_32b_vllm.slurm`
2. Monitor vLLM startup: `tail -f outputs/vllm_*.log`
3. Check server health: `curl http://127.0.0.1:8000/v1/models`
4. Verify generation starts: `tail -f outputs/evaluation_*.log`

## Key Variables

- **MAMBA_ENV_NAME**: Global variable set after `setup_environment()` call (Line 207)
- **VLLM_LOG_FILE**: Logs vLLM server startup/shutdown
- **LOG_FILE**: Main evaluation log with generation/evaluation progress

## Critical Success Factors

✅ vllm binary found when running `mamba run -n bfcl_vllm vllm serve`  
✅ BFCL CLI available when running `mamba run -n bfcl_vllm bfcl generate`  
✅ 2-GPU tensor parallel works (specified in evaluate_qwen3_32b_vllm.slurm)  
✅ OpenAI-compatible endpoint responds at http://127.0.0.1:8000/v1

## Files Modified

- `/evaluate_qwen3_32b_vllm.sh` - Main evaluation script (3 critical sections fixed)
