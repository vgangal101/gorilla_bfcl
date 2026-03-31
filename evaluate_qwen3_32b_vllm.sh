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

# Berkeley Function Call Leaderboard Evaluation Script for Qwen3-32B via vLLM
# This script:
#   1. Starts a vLLM server hosting Qwen3-32B (regular model)
#   2. Runs generation phase against the local vLLM endpoint (function calling via prompting)
#   3. Runs evaluation phase
#   4. Cleans up the server
# 
# Can be run directly: ./evaluate_qwen3_32b_vllm.sh
# Or submitted to SLURM: sbatch evaluate_qwen3_32b_vllm.sh

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Project paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BFCL_PROJECT_ROOT="${SCRIPT_DIR}"
export BFCL_PROJECT_ROOT

# Create logs directory if it doesn't exist
mkdir -p "${SCRIPT_DIR}/logs"

# vLLM Server Configuration
VLLM_MODEL="Qwen/Qwen3-32B"  # Regular Qwen3-32B model (function calling enabled via prompting)
VLLM_HOST="127.0.0.1"
VLLM_PORT=8000
VLLM_ENDPOINT="http://${VLLM_HOST}:${VLLM_PORT}/v1"
VLLM_GPU_MEMORY_UTILIZATION=0.9
VLLM_MAX_MODEL_LEN=8192
VLLM_TENSOR_PARALLEL_SIZE=2  # Use 2 GPUs with tensor parallelism

# BFCL Model Configuration
MODEL_NAME="Qwen/Qwen3-32B"  # Regular Qwen3-32B model (prompt-based function calling)
TEST_CATEGORY="multi_turn"  # Evaluate multi-turn function calling
NUM_THREADS=4

# Server management
VLLM_PID_FILE="${SCRIPT_DIR}/.vllm_server.pid"
MAX_RETRIES=30
RETRY_DELAY=5  # seconds

# Timestamp for logging
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="${SCRIPT_DIR}/logs/evaluation_${TIMESTAMP}.log"
VLLM_LOG_FILE="${SCRIPT_DIR}/logs/vllm_server_${TIMESTAMP}.log"

# Use SLURM job ID if available, otherwise use process ID
if [ -z "$SLURM_JOB_ID" ]; then
    JOB_ID="$$"
else
    JOB_ID="$SLURM_JOB_ID"
fi

# ============================================================================
# Helper Functions
# ============================================================================

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

error_exit() {
    log "ERROR: $*"
    cleanup
    exit 1
}

cleanup() {
    log "Cleaning up..."
    
    # Kill vLLM server if it's running
    if [ -f "$VLLM_PID_FILE" ]; then
        VLLM_PID=$(cat "$VLLM_PID_FILE")
        if ps -p "$VLLM_PID" > /dev/null 2>&1; then
            log "Killing vLLM server (PID: $VLLM_PID)..."
            kill -TERM "$VLLM_PID" 2>/dev/null || true
            
            # Wait a bit for graceful shutdown
            sleep 3
            
            # Force kill if still running
            if ps -p "$VLLM_PID" > /dev/null 2>&1; then
                log "Force killing vLLM server..."
                kill -9 "$VLLM_PID" 2>/dev/null || true
            fi
        fi
        rm -f "$VLLM_PID_FILE"
    fi
    
    log "Cleanup completed"
}

wait_for_server() {
    local endpoint=$1
    local max_retries=$2
    local retry_delay=$3
    local retry_count=0
    
    log "Waiting for vLLM server to be ready at ${endpoint}..."
    
    while [ $retry_count -lt $max_retries ]; do
        if curl -s "${endpoint}/models" > /dev/null 2>&1; then
            log "✓ vLLM server is ready!"
            return 0
        fi
        
        retry_count=$((retry_count + 1))
        log "  Attempt $retry_count/$max_retries: Server not ready yet, waiting ${retry_delay}s..."
        sleep "$retry_delay"
    done
    
    error_exit "vLLM server did not become ready after $((max_retries * retry_delay)) seconds"
}

create_mamba_environment() {
    local env_name=$1
    
    log "Creating mamba environment: $env_name"
    mamba create -n "$env_name" python=3.10 -y || error_exit "Failed to create mamba environment"
    log "✓ Mamba environment created"
}

setup_environment() {
    local env_name=$1
    
    log ""
    log "Loading mamba module..."
    module load mamba/latest || error_exit "Failed to load mamba module"
    log "✓ Mamba module loaded"
    
    # Check if environment exists and has vllm + BFCL already installed
    log "Checking if mamba environment exists: $env_name"
    if mamba env list | grep -q "^$env_name[[:space:]]"; then
        log "✓ Mamba environment '$env_name' already exists"
        
        # Check if both vllm and BFCL are installed
        log "Checking if vllm and BFCL are installed..."
        if mamba run -n "$env_name" bash -c "pip show vllm > /dev/null 2>&1 && pip show bfcl-eval > /dev/null 2>&1"; then
            log "✓ vllm and BFCL are already installed, skipping setup"
            return 0
        else
            log "Dependencies not complete, will reinstall"
        fi
    else
        create_mamba_environment "$env_name"
    fi
    
    # Install vllm using mamba (faster than pip)
    log "Installing vllm..."
    mamba install -n "$env_name" -y pytorch::pytorch pytorch::pytorch-cuda=12.1 "pytorch::pytorch-cuda/label/cu12_cudatoolkit_cudnn" > /dev/null 2>&1 || true
    mamba run -n "$env_name" pip install vllm || error_exit "Failed to install vllm"
    log "✓ vllm installed"
    
    # Install BFCL
    log "Installing BFCL..."
    cd "${BFCL_PROJECT_ROOT}/berkeley-function-call-leaderboard" || error_exit "Failed to navigate to BFCL directory"
    mamba run -n "$env_name" pip install -e . || error_exit "Failed to install BFCL"
    log "✓ BFCL installed"
    
    # Verify curl is available for health checks
    if ! command -v curl &> /dev/null; then
        log "WARNING: curl not found, installing..."
        mamba install -y curl -q
    fi
}

# Global variable to store environment name for use in other functions
MAMBA_ENV_NAME=""

# ============================================================================
# Main Script
# ============================================================================

log "========================================================================"
log "Berkeley Function Call Leaderboard - Qwen3-32B (vLLM, Prompt-based FC) Evaluation"
log "========================================================================"
log "Job ID: $JOB_ID"
log "Timestamp: $TIMESTAMP"
log "Model: $MODEL_NAME"
log "Test Category: $TEST_CATEGORY"
log "vLLM Endpoint: $VLLM_ENDPOINT"
log "Number of Threads: $NUM_THREADS"
log "========================================================================"

# Set up trap to cleanup on exit
trap cleanup EXIT INT TERM

# ============================================================================
# Environment Setup
# ============================================================================

ENV_NAME="bfcl_vllm"
setup_environment "$ENV_NAME"
MAMBA_ENV_NAME="$ENV_NAME"

# Change to BFCL directory
cd "${BFCL_PROJECT_ROOT}" || error_exit "Failed to change to BFCL directory"
log "Working directory: $(pwd)"

# ============================================================================
# Phase 0: Start vLLM Server
# ============================================================================

log ""
log "========================================================================"
log "PHASE 0: STARTING VLLM SERVER"
log "========================================================================"

log "Starting vLLM server..."
log "  Model: $VLLM_MODEL"
log "  GPU Memory Utilization: $VLLM_GPU_MEMORY_UTILIZATION"
log "  Tensor Parallel Size: $VLLM_TENSOR_PARALLEL_SIZE"
log "  Max Model Length: $VLLM_MAX_MODEL_LEN"
log "  Endpoint: $VLLM_ENDPOINT"

mamba run -n "$MAMBA_ENV_NAME" vllm serve "$VLLM_MODEL" \
    --host "$VLLM_HOST" \
    --port "$VLLM_PORT" \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE" \
    --trust-remote-code \
    > "$VLLM_LOG_FILE" 2>&1 &

VLLM_PID=$!
echo "$VLLM_PID" > "$VLLM_PID_FILE"
log "vLLM server started with PID: $VLLM_PID"

# Wait for server to be ready
wait_for_server "$VLLM_ENDPOINT" "$MAX_RETRIES" "$RETRY_DELAY"

# ============================================================================
# Phase 1: Generate LLM Responses
# ============================================================================

log ""
log "========================================================================"
log "PHASE 1: GENERATING LLM RESPONSES"
log "========================================================================"

log "Command: bfcl generate --model $MODEL_NAME --test-category $TEST_CATEGORY --num-threads $NUM_THREADS --openai-api-base $VLLM_ENDPOINT"

if mamba run -n "$MAMBA_ENV_NAME" bfcl generate \
    --model "$MODEL_NAME" \
    --test-category "$TEST_CATEGORY" \
    --num-threads "$NUM_THREADS" \
    --openai-api-base "$VLLM_ENDPOINT" \
    2>&1 | tee -a "${LOG_FILE}"; then
    log "✓ Generation phase completed successfully"
else
    error_exit "Generation phase failed"
fi

# ============================================================================
# Phase 2: Evaluate Generated Responses
# ============================================================================

log ""
log "========================================================================"
log "PHASE 2: EVALUATING GENERATED RESPONSES"
log "========================================================================"

log "Command: bfcl evaluate --model $MODEL_NAME --test-category $TEST_CATEGORY"

if mamba run -n "$MAMBA_ENV_NAME" bfcl evaluate \
    --model "$MODEL_NAME" \
    --test-category "$TEST_CATEGORY" \
    2>&1 | tee -a "${LOG_FILE}"; then
    log "✓ Evaluation phase completed successfully"
else
    error_exit "Evaluation phase failed"
fi

# ============================================================================
# Phase 3: Results Summary
# ============================================================================

log ""
log "========================================================================"
log "RESULTS SUMMARY"
log "========================================================================"

# Check if score directory exists
if [ -d "${BFCL_PROJECT_ROOT}/score" ]; then
    log "Score directory contents:"
    ls -lah "${BFCL_PROJECT_ROOT}/score/" | tee -a "${LOG_FILE}"
    
    # Display overall scores
    if [ -f "${BFCL_PROJECT_ROOT}/score/data_overall.csv" ]; then
        log ""
        log "Overall Scores (data_overall.csv):"
        head -20 "${BFCL_PROJECT_ROOT}/score/data_overall.csv" | tee -a "${LOG_FILE}"
    fi
    
    # Display model-specific scores if available
    if [ -d "${BFCL_PROJECT_ROOT}/score/$MODEL_NAME" ]; then
        log ""
        log "Model-specific score files:"
        ls -lah "${BFCL_PROJECT_ROOT}/score/$MODEL_NAME/" | tee -a "${LOG_FILE}"
    fi
else
    log "WARNING: Score directory not found"
fi

# ============================================================================
# Generate Summary Report
# ============================================================================

log ""
log "========================================================================"
log "GENERATING SUMMARY REPORT"
log "========================================================================"

REPORT_FILE="${SCRIPT_DIR}/logs/evaluation_report_${TIMESTAMP}.txt"

cat > "$REPORT_FILE" << EOF
===============================================================================
Berkeley Function Call Leaderboard - Qwen3-32B vLLM (Prompt-based FC) Evaluation Report
===============================================================================

Job ID: $JOB_ID
Timestamp: $TIMESTAMP
Model: $MODEL_NAME
Test Category: $TEST_CATEGORY
Number of Threads: $NUM_THREADS
vLLM Endpoint: $VLLM_ENDPOINT

===============================================================================
vLLM Server Configuration
===============================================================================

Model: $VLLM_MODEL
Host: $VLLM_HOST
Port: $VLLM_PORT
GPU Memory Utilization: $VLLM_GPU_MEMORY_UTILIZATION
Max Model Length: $VLLM_MAX_MODEL_LEN
Tensor Parallel Size: $VLLM_TENSOR_PARALLEL_SIZE
Function Calling: Prompt-based (via chat template)

===============================================================================
Results Location
===============================================================================

Generation Results:
  - Path: ${BFCL_PROJECT_ROOT}/result/${MODEL_NAME}/
  - Files:
    * BFCL_v3_*_result.json (generated responses)
    * inference_logs.json (detailed logs)

Evaluation Scores:
  - Path: ${BFCL_PROJECT_ROOT}/score/
  - Files:
    * data_overall.csv (summary scores)
    * data_non_live.csv (non-live category breakdown)
    * data_live.csv (live category breakdown)
    * data_multi_turn.csv (multi-turn specific scores)

Logs:
  - Main log: ${LOG_FILE}
  - vLLM server log: ${VLLM_LOG_FILE}

===============================================================================
Evaluation Summary
===============================================================================

Start Time: $TIMESTAMP
End Time: $(date +"%Y-%m-%d_%H-%M-%S")

Test Details:
  - Test Category: $TEST_CATEGORY
  - Multi-turn evaluations enabled
  - Context-aware function calling tested
  - Sequential and dependent function calls evaluated
  - Function calling via prompting (chat template)

===============================================================================
Next Steps
===============================================================================

1. Review the evaluation report:
   cat ${LOG_FILE}

2. Check vLLM server logs:
   tail -100 ${VLLM_LOG_FILE}

3. Analyze detailed scores:
   head -20 ${BFCL_PROJECT_ROOT}/score/data_overall.csv

4. For comparative analysis:
   Compare with other model results in ${BFCL_PROJECT_ROOT}/score/

===============================================================================
EOF

log "Summary report saved to: $REPORT_FILE"
log "Detailed logs saved to: $LOG_FILE"
log "Server logs saved to: $VLLM_LOG_FILE"

log ""
log "========================================================================"
log "EVALUATION COMPLETE"
log "========================================================================"
log "Cleaning up vLLM server..."
