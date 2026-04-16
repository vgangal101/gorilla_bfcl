#!/bin/bash
# sol_gaudi/gaudi_setup_check.sh
#
# Pre-flight diagnostic: confirm the login/compute node has Gaudi hardware and
# the tooling we need before we try to submit jobs. Exit non-zero if a hard
# requirement is missing; warnings are printed but do not fail.

set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/config.env" 2>/dev/null || source "${SCRIPT_DIR}/config.env.example"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BLUE='\033[0;34m'; NC='\033[0m'

pass=0; warn=0; fail=0

ok()   { echo -e "  ${GREEN}[ok]${NC}   $*"; pass=$((pass+1)); }
miss() { echo -e "  ${YELLOW}[warn]${NC} $*"; warn=$((warn+1)); }
bad()  { echo -e "  ${RED}[fail]${NC} $*"; fail=$((fail+1)); }
hdr()  { echo -e "\n${BLUE}== $* ==${NC}"; }

hdr "1. SLURM availability"
if command -v sbatch >/dev/null 2>&1; then ok "sbatch found ($(command -v sbatch))"; else bad "sbatch not on PATH — are you on a SOL login node?"; fi
if command -v sinfo >/dev/null 2>&1 && sinfo -h -p "${SLURM_PARTITION}" 2>/dev/null | grep -q .; then
    ok "partition '${SLURM_PARTITION}' is visible to sinfo"
else
    miss "partition '${SLURM_PARTITION}' not visible — check SLURM_PARTITION in config.env"
fi

hdr "2. Gaudi hardware"
if command -v hl-smi >/dev/null 2>&1; then
    ok "hl-smi found — querying HPUs"
    hl-smi -L 2>/dev/null | head -5 || miss "hl-smi returned no devices (expected on login nodes; fine if this is a login node)"
else
    miss "hl-smi not on PATH — expected on login node; compute node job will need it"
fi
if ls /dev/accel* >/dev/null 2>&1 || ls /dev/hl* >/dev/null 2>&1; then
    ok "Gaudi device files present under /dev"
else
    miss "no /dev/accel* or /dev/hl* — expected on login node, required on compute node"
fi

hdr "3. Apptainer"
if command -v apptainer >/dev/null 2>&1; then
    ok "apptainer found ($(apptainer --version 2>&1 | head -1))"
elif command -v singularity >/dev/null 2>&1; then
    ok "singularity found (apptainer alias) — $(singularity --version 2>&1 | head -1)"
else
    bad "neither apptainer nor singularity on PATH — required to run vLLM container"
fi

hdr "4. vLLM Gaudi container"
if [[ -f "${VLLM_GAUDI_SIF}" ]]; then
    ok "container present: ${VLLM_GAUDI_SIF} ($(du -h "${VLLM_GAUDI_SIF}" | cut -f1))"
else
    miss "container missing at ${VLLM_GAUDI_SIF} — run ./sol_gaudi/build_vllm_gaudi_sif.sh"
fi

hdr "5. Mamba / conda environment"
if module --version >/dev/null 2>&1 && module avail "${MAMBA_MODULE}" 2>&1 | grep -q "${MAMBA_MODULE}"; then
    ok "module '${MAMBA_MODULE}' is available"
elif command -v mamba >/dev/null 2>&1; then
    ok "mamba already on PATH"
elif command -v conda >/dev/null 2>&1; then
    miss "only conda on PATH (no mamba); setup will still work but slower"
else
    bad "neither mamba nor conda found — cannot build python env"
fi

if command -v mamba >/dev/null 2>&1 || command -v conda >/dev/null 2>&1; then
    mgr=$(command -v mamba 2>/dev/null || command -v conda)
    if "${mgr}" env list 2>/dev/null | awk '{print $1}' | grep -qx "${BFCL_GAUDI_ENV}"; then
        ok "env '${BFCL_GAUDI_ENV}' exists"
    else
        miss "env '${BFCL_GAUDI_ENV}' not found — run ./sol_gaudi/setup_gaudi_env.sh"
    fi
fi

hdr "6. Disk space on /scratch"
if [[ -d "/scratch/${USER}" ]]; then
    avail=$(df -BG "/scratch/${USER}" | awk 'NR==2 {print $4}')
    ok "/scratch/${USER} has ${avail} available"
else
    miss "/scratch/${USER} does not exist yet — will be created by setup"
fi

hdr "7. BFCL package check"
if [[ -f "${BFCL_ROOT}/pyproject.toml" ]]; then
    ok "BFCL package found at ${BFCL_ROOT}"
else
    bad "BFCL package not found at ${BFCL_ROOT} — check REPO_ROOT in config.env"
fi

echo ""
echo -e "${BLUE}== Summary ==${NC}"
echo -e "  ${GREEN}passed:${NC} ${pass}  ${YELLOW}warnings:${NC} ${warn}  ${RED}failures:${NC} ${fail}"
if [[ "${fail}" -gt 0 ]]; then
    echo -e "${RED}One or more hard requirements missing. Fix the [fail] items above before continuing.${NC}"
    exit 1
fi
echo -e "${GREEN}Gaudi setup check passed.${NC} Warnings are OK on a login node; they must be resolved inside the job."
