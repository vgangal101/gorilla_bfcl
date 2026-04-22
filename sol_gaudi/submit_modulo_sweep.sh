#!/bin/bash
# sol_gaudi/submit_modulo_sweep.sh
#
# Submit the LLM-Modulo Gaudi sweep on the thinking-capable Qwen3 models:
# 8B, 14B, 32B (4B is skipped — the Instruct-2507 variant BFCL registers
# is a non-thinking specialist). Defaults to the full 11-AST-category
# config; override with MODULO_CONFIG for a narrower run.
#
# Usage:
#   ./sol_gaudi/submit_modulo_sweep.sh                                  # full coverage
#   MODULO_CONFIG=configs/modulo_example.yaml ./sol_gaudi/submit_modulo_sweep.sh
#   MODELS="qwen3_14b_modulo qwen3_32b_modulo" ./sol_gaudi/submit_modulo_sweep.sh

set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

MODELS="${MODELS:-qwen3_8b_modulo qwen3_14b_modulo qwen3_32b_modulo}"
export MODULO_CONFIG="${MODULO_CONFIG:-configs/modulo_full.yaml}"

echo "Submitting LLM-Modulo sweep with MODULO_CONFIG=${MODULO_CONFIG}"
echo "Models: ${MODELS}"
echo

for m in ${MODELS}; do
    "${SCRIPT_DIR}/manage_bfcl_gaudi.sh" submit "${m}" || echo "  (skipping ${m} due to submit failure)"
done

echo
echo "Watch:"
echo "  ./sol_gaudi/manage_bfcl_gaudi.sh status"
echo "  ./sol_gaudi/manage_bfcl_gaudi.sh logs <JOB_ID>"
