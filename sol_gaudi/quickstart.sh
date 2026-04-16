#!/bin/bash
# sol_gaudi/quickstart.sh
#
# One-command entry: check + setup + generate + submit smoke test.
# Run this from a SOL login node after cloning the repo and creating config.env.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

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
python3 "${SCRIPT_DIR}/generate_bfcl_scripts.py"

# --- 4. Submit smoke test ---
step "4/4 Submit smoke test: qwen3_4b on simple_python"
BFCL_TEST_CATEGORY="${BFCL_TEST_CATEGORY:-simple_python}" \
    bash "${SCRIPT_DIR}/manage_bfcl_gaudi.sh" submit qwen3_4b

echo
echo -e "${GREEN}Quickstart complete.${NC}"
echo
echo "Next steps:"
echo "  ./sol_gaudi/manage_bfcl_gaudi.sh status"
echo "  ./sol_gaudi/manage_bfcl_gaudi.sh logs <JOB_ID>"
echo "  ./sol_gaudi/manage_bfcl_gaudi.sh results qwen3_4b   # once job finishes"
echo
echo "Full sweep (after smoke test passes):"
echo "  ./sol_gaudi/manage_bfcl_gaudi.sh submit-all"
