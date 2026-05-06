#!/usr/bin/env bash
# smoke_login.sh — Login-node smoke tests. No GPU required.
# Run this before submitting any SLURM jobs to catch import and data issues early.
#
# Usage (from repo root):
#   bash grpo_pipeline/smoke_tests/smoke_login.sh

set -euo pipefail

module load mamba/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bfcl_rl

PASS=0
FAIL=0

check() {
    local label="$1"
    local cmd="$2"
    printf "  %-50s" "$label"
    if eval "$cmd" > /tmp/smoke_out 2>&1; then
        echo "PASS"
        PASS=$((PASS + 1))
    else
        echo "FAIL"
        cat /tmp/smoke_out
        FAIL=$((FAIL + 1))
    fi
}

echo "========================================"
echo "BFCL RL — Login Node Smoke Tests"
echo "========================================"
echo ""

echo "[ Imports ]"
check "import torch"          "python -c 'import torch; print(torch.__version__)'"
check "import transformers"   "python -c 'import transformers'"
check "import trl"            "python -c 'import trl'"
check "import peft"           "python -c 'import peft'"
check "import accelerate"     "python -c 'import accelerate'"
check "import datasets"       "python -c 'import datasets'"
check "import vllm"           "python -c 'import vllm'"
check "import bfcl_eval"      "python -c 'import sys; sys.path.insert(0, \"berkeley-function-call-leaderboard\"); import bfcl_eval'"
echo ""

echo "[ Data pipeline ]"
check "data_prep.py runs"     "python grpo_pipeline/data_prep.py"
echo ""

echo "[ Reward function ]"
check "reward.py smoke test"  "python grpo_pipeline/reward.py"
echo ""

echo "[ BFCL data files exist ]"
for cat in simple_python simple_java simple_javascript parallel_function multiple_function; do
    check "BFCL_v4_${cat}.json" \
        "test -f berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_${cat}.json"
    check "possible_answer/BFCL_v4_${cat}.json" \
        "test -f berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v4_${cat}.json"
done
echo ""

echo "[ CUDA ]"
check "CUDA available" "python -c 'import torch; assert torch.cuda.is_available(), \"No CUDA\"'"
echo ""

echo "========================================"
echo "Results: $PASS passed, $FAIL failed"
echo "========================================"

[ $FAIL -eq 0 ] && echo "All checks passed. Safe to submit smoke_test.slurm." || echo "Fix failures above before submitting."
exit $FAIL
