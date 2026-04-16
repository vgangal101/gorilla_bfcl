#!/bin/bash
# sol_gaudi/setup_gaudi_env.sh
#
# Idempotent setup: load mamba, create/update bfcl_gaudi env, install BFCL
# in editable mode, resolve the Apptainer SIF path. Safe to re-run.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/config.env" 2>/dev/null || source "${SCRIPT_DIR}/config.env.example"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BLUE='\033[0;34m'; NC='\033[0m'
say()  { echo -e "${BLUE}[setup]${NC} $*"; }
ok()   { echo -e "${GREEN}[ok]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }
die()  { echo -e "${RED}[fail]${NC} $*" >&2; exit 1; }

say "Loading mamba module"
if command -v module >/dev/null 2>&1; then
    module load "${MAMBA_MODULE}" 2>/dev/null || warn "module load ${MAMBA_MODULE} failed — hoping mamba is already on PATH"
fi
command -v mamba >/dev/null 2>&1 || command -v conda >/dev/null 2>&1 || die "neither mamba nor conda on PATH after module load"
PKG_MGR=$(command -v mamba 2>/dev/null || command -v conda)

say "Ensuring env '${BFCL_GAUDI_ENV}' exists"
if ! "${PKG_MGR}" env list 2>/dev/null | awk '{print $1}' | grep -qx "${BFCL_GAUDI_ENV}"; then
    say "Creating env '${BFCL_GAUDI_ENV}' (python=${PYTHON_VERSION})"
    "${PKG_MGR}" create -n "${BFCL_GAUDI_ENV}" -c conda-forge "python=${PYTHON_VERSION}" -y
else
    ok "env '${BFCL_GAUDI_ENV}' already exists"
fi

say "Installing BFCL into '${BFCL_GAUDI_ENV}'"
"${PKG_MGR}" run -n "${BFCL_GAUDI_ENV}" pip install --upgrade pip
"${PKG_MGR}" run -n "${BFCL_GAUDI_ENV}" pip install -e "${BFCL_ROOT}"
ok "BFCL installed (editable)"

say "Resolving vLLM Gaudi Apptainer SIF"
mkdir -p "${CONTAINERS_DIR}"
if [[ -f "${VLLM_GAUDI_SIF}" ]]; then
    ok "SIF already present at ${VLLM_GAUDI_SIF}"
else
    warn "SIF missing at ${VLLM_GAUDI_SIF}"
    read -r -p "Build it now via build_vllm_gaudi_sif.sh? This pulls ~20 GB. [y/N] " ans
    if [[ "${ans}" =~ ^[Yy]$ ]]; then
        bash "${SCRIPT_DIR}/build_vllm_gaudi_sif.sh"
    else
        warn "Skipped SIF build. Run ./sol_gaudi/build_vllm_gaudi_sif.sh before submitting jobs."
    fi
fi

say "Ensuring HF cache dir exists: ${HF_HOME}"
mkdir -p "${HF_HOME}"
ok "HF_HOME=${HF_HOME}"

ok "Setup complete. Next: python ${SOL_GAUDI_ROOT}/generate_bfcl_scripts.py"
