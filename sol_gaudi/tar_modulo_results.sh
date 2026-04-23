#!/bin/bash
# sol_gaudi/tar_modulo_results.sh
#
# Bundle the artifacts from an LLM-Modulo sweep into a single tarball so it
# can be scp'd off SOL. Filters by date so stale baseline score JSONs in the
# shared score/<Model>/ dirs (branch CLAUDE.md gotcha #4) don't sneak in.
#
# Usage:
#   ./sol_gaudi/tar_modulo_results.sh [OUTPUT_PATH]
#
# Env overrides:
#   SINCE_DATE   ISO date; only files newer than this are included
#                (default: today 00:00 in the shell's local tz)
#   MODELS       Space-separated HF-style model dirs under result_modulo/
#                (default: "Qwen_Qwen3-8B Qwen_Qwen3-14B Qwen_Qwen3-32B")
#   CONFIG_YAML  Path (relative to BFCL_ROOT) to the modulo config to include
#                (default: configs/modulo_full.yaml)

set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/config.env" 2>/dev/null || source "${SCRIPT_DIR}/config.env.example"

GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

SINCE_DATE="${SINCE_DATE:-$(date +%Y-%m-%d)}"
MODELS="${MODELS:-Qwen_Qwen3-8B Qwen_Qwen3-14B Qwen_Qwen3-32B}"
CONFIG_YAML="${CONFIG_YAML:-configs/modulo_full.yaml}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${1:-/scratch/${USER}/modulo_sweep_${STAMP}.tar.gz}"
LIST_FILE="$(mktemp /tmp/modulo_files_XXXXXX.txt)"
trap 'rm -f "${LIST_FILE}"' EXIT

cd "${REPO_ROOT}" || { echo -e "${RED}Cannot cd ${REPO_ROOT}${NC}" >&2; exit 1; }

echo -e "${BLUE}Bundling modulo results${NC}"
echo "  REPO_ROOT    : ${REPO_ROOT}"
echo "  SINCE_DATE   : ${SINCE_DATE}"
echo "  MODELS       : ${MODELS}"
echo "  CONFIG_YAML  : ${CONFIG_YAML}"
echo "  OUTPUT       : ${OUT}"
echo

missing=0

# 1) Modulo generations — full dirs for the listed models
for m in ${MODELS}; do
    dir="berkeley-function-call-leaderboard/result_modulo/${m}"
    if [[ -d "${dir}" ]]; then
        find "${dir}" -type f >> "${LIST_FILE}"
    else
        echo -e "${YELLOW}  skip (no dir): ${dir}${NC}"
        missing=$((missing+1))
    fi
done

# 2) Per-category score JSONs newer than SINCE_DATE — excludes baseline leftovers
for m in ${MODELS}; do
    dir="berkeley-function-call-leaderboard/score/${m}"
    if [[ -d "${dir}" ]]; then
        find "${dir}" -name "BFCL_v4_*_score.json" -newermt "${SINCE_DATE}" >> "${LIST_FILE}"
    fi
done

# 3) Aggregate CSVs (hybrid, but useful for context)
find berkeley-function-call-leaderboard/score -maxdepth 1 -name "data_*.csv" -type f >> "${LIST_FILE}" 2>/dev/null || true

# 4) Modulo job logs newer than SINCE_DATE
find sol_gaudi/logs -maxdepth 1 -type f \
     \( -name "bfcl_qwen3_*_modulo_*.out" -o -name "bfcl_qwen3_*_modulo_*.err" \) \
     -newermt "${SINCE_DATE}" >> "${LIST_FILE}" 2>/dev/null || true

# 5) Run config for reproducibility
cfg="berkeley-function-call-leaderboard/${CONFIG_YAML}"
if [[ -f "${cfg}" ]]; then
    echo "${cfg}" >> "${LIST_FILE}"
else
    echo -e "${YELLOW}  skip (no file): ${cfg}${NC}"
fi

# Dedupe and sanity-check
sort -u -o "${LIST_FILE}" "${LIST_FILE}"
count=$(wc -l < "${LIST_FILE}")
if [[ "${count}" -eq 0 ]]; then
    echo -e "${RED}No files matched — check SINCE_DATE=${SINCE_DATE} and MODELS.${NC}" >&2
    exit 1
fi

echo -e "${BLUE}File list preview (${count} entries):${NC}"
head -20 "${LIST_FILE}"
if [[ "${count}" -gt 20 ]]; then
    echo "  ... ($((count - 20)) more)"
fi
echo

mkdir -p "$(dirname "${OUT}")"
tar czf "${OUT}" -T "${LIST_FILE}"

size=$(du -h "${OUT}" | cut -f1)
echo -e "${GREEN}Wrote ${OUT} (${size}, ${count} files)${NC}"
if [[ "${missing}" -gt 0 ]]; then
    echo -e "${YELLOW}Note: ${missing} model dir(s) missing under result_modulo/${NC}"
fi
echo
echo "Pull to laptop with:"
echo "  scp ${USER}@login.sol.rc.asu.edu:${OUT} ~/Desktop/"
echo
echo "Then extract:"
echo "  mkdir -p ~/Desktop/$(basename "${OUT}" .tar.gz)"
echo "  tar xzf ~/Desktop/$(basename "${OUT}") -C ~/Desktop/$(basename "${OUT}" .tar.gz)"
