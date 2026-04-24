#!/bin/bash
# sol_gaudi/tar_baseline_results.sh
#
# Bundle the artifacts from a baseline BFCL-Gaudi sweep (bfcl generate +
# bfcl evaluate on Qwen3-8B/14B/32B) into a single tarball so it can be
# scp'd off SOL. Filters by date so stale per-category score JSONs in the
# shared score/<Model>/ dirs (branch CLAUDE.md gotcha #5) don't sneak in.
#
# Usage:
#   ./sol_gaudi/tar_baseline_results.sh [OUTPUT_PATH]
#
# Env overrides:
#   SINCE_DATE   ISO date; only files newer than this are included
#                (default: today 00:00 in the shell's local tz)
#   MODELS       Space-separated HF-style model dirs under result/
#                (default: "Qwen_Qwen3-8B Qwen_Qwen3-14B Qwen_Qwen3-32B")

set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/config.env" 2>/dev/null || source "${SCRIPT_DIR}/config.env.example"

GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

SINCE_DATE="${SINCE_DATE:-$(date +%Y-%m-%d)}"
MODELS="${MODELS:-Qwen_Qwen3-8B Qwen_Qwen3-14B Qwen_Qwen3-32B}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${1:-/scratch/${USER}/baseline_sweep_${STAMP}.tar.gz}"
LIST_FILE="$(mktemp /tmp/baseline_files_XXXXXX.txt)"
trap 'rm -f "${LIST_FILE}"' EXIT

cd "${REPO_ROOT}" || { echo -e "${RED}Cannot cd ${REPO_ROOT}${NC}" >&2; exit 1; }

echo -e "${BLUE}Bundling baseline results${NC}"
echo "  REPO_ROOT    : ${REPO_ROOT}"
echo "  SINCE_DATE   : ${SINCE_DATE}"
echo "  MODELS       : ${MODELS}"
echo "  OUTPUT       : ${OUT}"
echo

missing=0

# 1) Baseline generations — full dirs for the listed models
for m in ${MODELS}; do
    dir="berkeley-function-call-leaderboard/result/${m}"
    if [[ -d "${dir}" ]]; then
        find "${dir}" -type f >> "${LIST_FILE}"
    else
        echo -e "${YELLOW}  skip (no dir): ${dir}${NC}"
        missing=$((missing+1))
    fi
done

# 2) Per-category score JSONs newer than SINCE_DATE — excludes stale leftovers
for m in ${MODELS}; do
    dir="berkeley-function-call-leaderboard/score/${m}"
    if [[ -d "${dir}" ]]; then
        find "${dir}" -name "BFCL_v4_*_score.json" -newermt "${SINCE_DATE}" >> "${LIST_FILE}"
    fi
done

# 3) Aggregate CSVs
find berkeley-function-call-leaderboard/score -maxdepth 1 -name "data_*.csv" -type f >> "${LIST_FILE}" 2>/dev/null || true

# 4) Baseline job logs newer than SINCE_DATE (_gaudi jobs, not _modulo)
find sol_gaudi/logs -maxdepth 1 -type f \
     \( -name "bfcl_qwen3_8b_*.out"  -o -name "bfcl_qwen3_8b_*.err" \
     -o -name "bfcl_qwen3_14b_*.out" -o -name "bfcl_qwen3_14b_*.err" \
     -o -name "bfcl_qwen3_32b_*.out" -o -name "bfcl_qwen3_32b_*.err" \) \
     ! -name "*_modulo_*" \
     -newermt "${SINCE_DATE}" >> "${LIST_FILE}" 2>/dev/null || true

# 5) The .slurm files that were submitted, for reproducibility
for slurm in sol_gaudi/slurm/bfcl_qwen3_8b_gaudi.slurm \
             sol_gaudi/slurm/bfcl_qwen3_14b_gaudi.slurm \
             sol_gaudi/slurm/bfcl_qwen3_32b_gaudi.slurm; do
    [[ -f "${slurm}" ]] && echo "${slurm}" >> "${LIST_FILE}"
done

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
    echo -e "${YELLOW}Note: ${missing} model dir(s) missing under result/${NC}"
fi
echo
echo "Pull to laptop with:"
echo "  scp ${USER}@login.sol.rc.asu.edu:${OUT} ~/Desktop/"
echo
echo "Then extract:"
echo "  mkdir -p ~/Desktop/$(basename "${OUT}" .tar.gz)"
echo "  tar xzf ~/Desktop/$(basename "${OUT}") -C ~/Desktop/$(basename "${OUT}" .tar.gz)"
