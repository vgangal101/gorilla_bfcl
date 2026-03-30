#!/bin/bash
# Setup script for Qwen3-32B evaluation on SOL

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BFCL_DIR="${SCRIPT_DIR}/berkeley-function-call-leaderboard"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')] $1${NC}"
}

success() {
    echo -e "${GREEN}✓ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if we're on SOL
if ! hostname | grep -q "sol"; then
    warning "This script is designed for SOL cluster. Some commands may not work elsewhere."
fi

# Create logs directory
log "Creating logs directory..."
mkdir -p "${SCRIPT_DIR}/logs"
success "Logs directory created"

# Check SLURM availability
log "Checking SLURM availability..."
if command -v sbatch &> /dev/null; then
    success "SLURM available"
else
    error "SLURM not found. This script requires SLURM."
    exit 1
fi

# Check GPU availability
log "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    success "Found $GPU_COUNT GPU(s)"
else
    warning "nvidia-smi not found. GPU detection may not work."
fi

# Check mamba/conda
log "Checking conda/mamba..."
if command -v mamba &> /dev/null; then
    success "Mamba available"
elif command -v conda &> /dev/null; then
    success "Conda available (mamba preferred for speed)"
else
    error "Neither mamba nor conda found. Please load mamba module."
    exit 1
fi

# Navigate to BFCL directory
log "Checking BFCL directory..."
if [ ! -d "$BFCL_DIR" ]; then
    error "BFCL directory not found: $BFCL_DIR"
    exit 1
fi
success "BFCL directory found"

# Check if BFCL is installed
log "Checking BFCL installation..."
cd "$BFCL_DIR"
if python -c "import bfcl_eval" &> /dev/null; then
    success "BFCL package already installed"
else
    warning "BFCL package not installed. Will be installed during evaluation."
fi

# Check for existing environment
ENV_NAME="bfcl_vllm"
log "Checking conda environment: $ENV_NAME"
if conda env list | grep -q "^$ENV_NAME[[:space:]]"; then
    success "Conda environment '$ENV_NAME' exists"
else
    warning "Conda environment '$ENV_NAME' not found. Will be created during evaluation."
fi

# Check available disk space
log "Checking disk space..."
DISK_SPACE=$(df -BG "$SCRIPT_DIR" | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$DISK_SPACE" -gt 50 ]; then
    success "Sufficient disk space: ${DISK_SPACE}GB available"
else
    warning "Low disk space: ${DISK_SPACE}GB available. May need cleanup."
fi

# Check evaluation scripts
log "Checking evaluation scripts..."
SCRIPTS=(
    "evaluate_qwen3_32b_vllm.sh"
    "evaluate_qwen3_32b_vllm.slurm"
    "manage_qwen_eval.sh"
    "analyze_qwen_results.py"
)

for script in "${SCRIPTS[@]}"; do
    if [ -x "${SCRIPT_DIR}/${script}" ]; then
        success "$script is executable"
    else
        warning "$script is not executable. Run: chmod +x $script"
    fi
done

# Create .env file if it doesn't exist
if [ ! -f "${SCRIPT_DIR}/.env" ]; then
    log "Creating .env file..."
    cat > "${SCRIPT_DIR}/.env" << EOF
# BFCL Environment Configuration
# Add any API keys here if needed for other models
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
EOF
    success ".env file created"
else
    success ".env file exists"
fi

# Summary
echo ""
echo "========================================================================"
echo "QWEN3-32B EVALUATION SETUP COMPLETE"
echo "========================================================================"
echo ""
echo "Ready to run evaluation:"
echo ""
echo "  1. Submit job:"
echo "     ./manage_qwen_eval.sh submit"
echo ""
echo "  2. Check status:"
echo "     ./manage_qwen_eval.sh status"
echo ""
echo "  3. View logs:"
echo "     ./manage_qwen_eval.sh logs"
echo ""
echo "  4. View results:"
echo "     ./manage_qwen_eval.sh results"
echo ""
echo "  5. Analyze results:"
echo "     python analyze_qwen_results.py --action summary"
echo ""
echo "Documentation:"
echo "  - QWEN3_32B_EVAL_GUIDE.md"
echo ""
echo "Configuration:"
echo "  - Model: Qwen/Qwen3-32B (regular model)"
echo "  - Function Calling: Prompt-based (via Qwen chat template)"
echo "  - Test Category: multi_turn"
echo "  - GPUs: 2 (tensor parallel)"
echo "  - Memory: 64GB"
echo "  - Threads: 4"
echo ""
echo "========================================================================"

# Final checks
echo ""
echo "Final checks:"
echo "  - SLURM: $(sbatch --version 2>/dev/null | head -1 || echo 'Not available')"
echo "  - GPUs: $(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 'Unknown') available"
echo "  - Disk space: $(df -h "$SCRIPT_DIR" | tail -1 | awk '{print $4}') available"
echo "  - Python: $(python --version 2>&1)"
echo ""
