#!/bin/bash
#SBATCH --job-name=bfcl_gpt4o_mini
#SBATCH --output=logs/bfcl_gpt4o_mini_%j.out
#SBATCH --error=logs/bfcl_gpt4o_mini_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0

# Berkeley Function Call Leaderboard Evaluation Script for GPT-4o-mini
# This script runs both generation and evaluation phases
# Can be run directly: ./evaluate_gpt4o_mini.sh
# Or submitted to SLURM: sbatch evaluate_gpt4o_mini.sh

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

# API key source file
API_KEY_FILE="${HOME}/openai_key_lab.sh"  # Adjust path as needed

# Model and test category settings
MODEL_NAME="gpt-4o-mini-2024-07-18-FC"
TEST_CATEGORY="all_scoring"  # Can be: simple_python, parallel, multi_turn, all_scoring, etc.
NUM_THREADS=2  # Adjust based on API rate limits

# Timestamp for logging
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="${SCRIPT_DIR}/logs/evaluation_${TIMESTAMP}.log"

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
    exit 1
}

# ============================================================================
# Main Script
# ============================================================================

log "========================================================================"
log "Berkeley Function Call Leaderboard - GPT-4o-mini Evaluation"
log "========================================================================"
log "Job ID: $JOB_ID"
log "Timestamp: $TIMESTAMP"
log "Model: $MODEL_NAME"
log "Test Category: $TEST_CATEGORY"
log "Number of Threads: $NUM_THREADS"
log "========================================================================"

# Source API key
log "Sourcing API key from: $API_KEY_FILE"
if [ ! -f "$API_KEY_FILE" ]; then
    error_exit "API key file not found: $API_KEY_FILE"
fi
source "$API_KEY_FILE" || error_exit "Failed to source API key file"
log "API key loaded successfully"

# Verify OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    error_exit "OPENAI_API_KEY not set after sourcing $API_KEY_FILE"
fi
export OPENAI_API_KEY

# Export BFCL project root
log "BFCL_PROJECT_ROOT: $BFCL_PROJECT_ROOT"

# ============================================================================
# Module and Environment Setup
# ============================================================================

log ""
log "Loading mamba module..."
module load mamba/latest || error_exit "Failed to load mamba module"
log "✓ Mamba module loaded successfully"

# Define environment name
ENV_NAME="bfcl"

# Check if conda environment exists
log "Checking if conda environment exists: $ENV_NAME"
if conda env list | grep -q "^$ENV_NAME[[:space:]]"; then
    log "✓ Conda environment '$ENV_NAME' already exists, skipping construction"
else
    log "Creating conda environment: $ENV_NAME"
    conda create -n "$ENV_NAME" python=3.10 -y || error_exit "Failed to create conda environment"
    log "✓ Conda environment created successfully"
fi

# Install dependencies in the environment (if needed)
log "Installing BFCL dependencies in environment..."
conda run -n "$ENV_NAME" pip install -e . || error_exit "Failed to install BFCL dependencies"
log "✓ Dependencies installed successfully"# Activate environment

log "Activating conda environment: $ENV_NAME"
conda activate "$ENV_NAME" || error_exit "Failed to activate conda environment"
log "✓ Conda environment activated"

# Check if .env file exists
if [ ! -f "${BFCL_PROJECT_ROOT}/.env" ]; then
    log "WARNING: .env file not found at ${BFCL_PROJECT_ROOT}/.env"
    log "Creating .env file..."
    cat > "${BFCL_PROJECT_ROOT}/.env" << EOF
OPENAI_API_KEY=${OPENAI_API_KEY}
EOF
    log ".env file created"
else
    log ".env file found"
fi

# Change to BFCL directory
cd "${BFCL_PROJECT_ROOT}" || error_exit "Failed to change to BFCL directory"
log "Working directory: $(pwd)"

# ============================================================================
# Phase 1: Generate LLM Responses
# ============================================================================

log ""
log "========================================================================"
log "PHASE 1: GENERATING LLM RESPONSES"
log "========================================================================"

log "Command: bfcl generate --model $MODEL_NAME --test-category $TEST_CATEGORY --num-threads $NUM_THREADS"

if bfcl generate \
    --model "$MODEL_NAME" \
    --test-category "$TEST_CATEGORY" \
    --num-threads "$NUM_THREADS" \
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

if bfcl evaluate \
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
Berkeley Function Call Leaderboard - Evaluation Report
===============================================================================

Job ID: $JOB_ID
Timestamp: $TIMESTAMP
Model: $MODEL_NAME
Test Category: $TEST_CATEGORY
Number of Threads: $NUM_THREADS

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
    * data_multi_turn.csv (multi-turn breakdown)
    * ${MODEL_NAME}/BFCL_v3_*_score.json (detailed scores per category)

===============================================================================
Analysis Commands
===============================================================================

View overall scores:
  cat ${BFCL_PROJECT_ROOT}/score/data_overall.csv

View category-specific scores:
  cat ${BFCL_PROJECT_ROOT}/score/data_non_live.csv

View detailed JSON scores:
  cat ${BFCL_PROJECT_ROOT}/score/${MODEL_NAME}/BFCL_v3_simple_python_score.json | python3 -m json.tool

Extract key metrics:
  cut -d',' -f1-5 ${BFCL_PROJECT_ROOT}/score/data_overall.csv

===============================================================================
Log Files
===============================================================================

Main evaluation log: ${LOG_FILE}
This report: ${REPORT_FILE}

===============================================================================
EOF

log "Summary report written to: $REPORT_FILE"
cat "$REPORT_FILE" | tee -a "${LOG_FILE}"

# ============================================================================
# Final Status
# ============================================================================

log ""
log "========================================================================"
log "EVALUATION COMPLETE"
log "========================================================================"
log "All phases completed successfully!"
log "Check the results in: ${BFCL_PROJECT_ROOT}/score/"
log "Detailed logs in: ${LOG_FILE}"
log "========================================================================"
