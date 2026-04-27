#!/bin/bash
# sol_gaudi/quickstart.sh
#
# One-command entry: check + setup + generate + submit smoke test.
# Run this from a SOL login node after cloning the repo and creating config.env.
#
# Usage:
#   ./sol_gaudi/quickstart.sh            # baseline: bfcl generate qwen3_4b simple_python
#   ./sol_gaudi/quickstart.sh modulo     # LLM-Modulo: qwen3_4b_modulo, configs/smoketest.yaml

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

MODE="${1:-baseline}"
case "${MODE}" in
    baseline|modulo) ;;
    *) echo "Unknown mode: ${MODE}. Use 'baseline' or 'modulo'." >&2; exit 1;;
esac

GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
step() { echo -e "\n${BLUE}=== $* ===${NC}"; }
die()  { echo -e "${RED}$*${NC}" >&2; exit 1; }

# --- 0. Ensure config.env exists ---
if [[ ! -f "${SCRIPT_DIR}/config.env" ]]; then
    echo -e "${YELLOW}config.env missing — copying from config.env.example${NC}"
    cp "${SCRIPT_DIR}/config.env.example" "${SCRIPT_DIR}/config.env"
    echo "Edit ${SCRIPT_DIR}/config.env if your account/QoS/SIF path differs from defaults, then re-run."
    echo "If the defaults are fine, just re-run ./sol_gaudi/quickstart.sh now."
    exit 0
fi

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/config.env"

# --- 1. Hardware/software probe ---
step "1/4 Pre-flight check (gaudi_setup_check.sh)"
bash "${SCRIPT_DIR}/gaudi_setup_check.sh" || die "Pre-flight check failed. Fix the [fail] items above, then re-run."

# --- 2. Env + BFCL install ---
step "2/4 Python env + BFCL install (setup_gaudi_env.sh)"
bash "${SCRIPT_DIR}/setup_gaudi_env.sh"

# --- 3. Generate SBATCH scripts ---
step "3/4 Generate SBATCH scripts"
# SOL login default python3 is Python 3.6 (too old for the generator's
# type hints). Activate the env we just built so python3 is 3.11.
if command -v module >/dev/null 2>&1; then
    module load "${MAMBA_MODULE}" 2>/dev/null || true
fi
# shellcheck disable=SC1091
source activate "${BFCL_GAUDI_ENV}"
python3 "${SCRIPT_DIR}/generate_bfcl_scripts.py"

# --- 4. Submit smoke test ---
if [[ "${MODE}" == "modulo" ]]; then
    step "4/4 Submit smoke test: qwen3_4b_modulo on configs/smoketest.yaml"
    MODULO_CONFIG="${MODULO_CONFIG:-configs/smoketest.yaml}" \
        bash "${SCRIPT_DIR}/manage_bfcl_gaudi.sh" submit qwen3_4b_modulo
    SMOKE_TARGET="qwen3_4b_modulo"
else
    step "4/4 Submit smoke test: qwen3_4b on simple_python"
    BFCL_TEST_CATEGORY="${BFCL_TEST_CATEGORY:-simple_python}" \
        bash "${SCRIPT_DIR}/manage_bfcl_gaudi.sh" submit qwen3_4b
    SMOKE_TARGET="qwen3_4b"
fi

echo
echo -e "${GREEN}Quickstart complete.${NC}"
echo
echo "Next steps:"
echo "  ./sol_gaudi/manage_bfcl_gaudi.sh status"
echo "  ./sol_gaudi/manage_bfcl_gaudi.sh logs <JOB_ID>"
echo "  ./sol_gaudi/manage_bfcl_gaudi.sh results ${SMOKE_TARGET}   # once job finishes"
echo
if [[ "${MODE}" == "modulo" ]]; then
    echo "Full modulo sweep (after smoke test passes):"
    echo "  for m in qwen3_4b_modulo qwen3_8b_modulo qwen3_14b_modulo qwen3_32b_modulo; do"
    echo "      ./sol_gaudi/manage_bfcl_gaudi.sh submit \$m; done"
else
    echo "Full sweep (after smoke test passes):"
    echo "  ./sol_gaudi/manage_bfcl_gaudi.sh submit-all"
fi
