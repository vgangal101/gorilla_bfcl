#!/bin/bash
# sol_gaudi/run_java_modulo.sh
#
# One-shot launcher for the LLM-Modulo Java/JS sweep on SOL Gaudi.
# Submits qwen3_8b/14b/32b modulo jobs with configs/modulo_java.yaml
# (simple_python + simple_java + simple_javascript), exercising the
# LanguageASTCritic on multi-language AST categories.
#
# Usage:
#   ./sol_gaudi/run_java_modulo.sh                              # all three models
#   MODELS="qwen3_8b_modulo" ./sol_gaudi/run_java_modulo.sh     # smoke-test 8B only
#
# 4B is intentionally excluded — Qwen3-4B-Instruct-2507 is non-thinking and
# regresses on hybrid-mode prompts.

set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

# 1. First-run config.env handling — copy the example and ask the user to
#    edit only if their account differs from the class defaults.
if [[ ! -f "${SCRIPT_DIR}/config.env" ]]; then
    cp "${SCRIPT_DIR}/config.env.example" "${SCRIPT_DIR}/config.env"
    echo -e "${YELLOW}Created sol_gaudi/config.env from the example.${NC}"
    echo "Edit it only if your SLURM account/QoS/SIF path differs from defaults,"
    echo "then re-run this script."
    exit 0
fi

# 2. Make sure modulo SLURM scripts exist (they're regenerated from the
#    Python template in case the table was edited).
if [[ ! -f "${SCRIPT_DIR}/slurm/bfcl_qwen3_8b_modulo_gaudi.slurm" ]]; then
    echo -e "${BLUE}Regenerating SLURM scripts...${NC}"
    python3 "${SCRIPT_DIR}/generate_bfcl_scripts.py"
fi

# 3. Submit. Defaults to all three modulo models; override with MODELS=...
export MODULO_CONFIG="configs/modulo_java.yaml"
export MODELS="${MODELS:-qwen3_8b_modulo qwen3_14b_modulo qwen3_32b_modulo}"

echo -e "${BLUE}Launching LLM-Modulo Java/JS sweep${NC}"
echo "  config : ${MODULO_CONFIG}"
echo "  models : ${MODELS}"
echo

"${SCRIPT_DIR}/submit_modulo_sweep.sh"

echo
echo -e "${GREEN}Submitted.${NC} Watch / fetch with:"
echo "  ./sol_gaudi/manage_bfcl_gaudi.sh status"
echo "  ./sol_gaudi/manage_bfcl_gaudi.sh logs <JOB_ID>"
echo "  ./sol_gaudi/tar_results.sh           # bundle results once jobs finish"
