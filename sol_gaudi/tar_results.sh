#!/bin/bash
# sol_gaudi/tar_results.sh
#
# Bundle BFCL Gaudi run artifacts — both baseline (result/) and modulo
# (result_modulo/) — into a single tarball so it can be scp'd off SOL.
# Works on both `bfcl_gaudi` and `LLM-Modulo_gaudi` branches: whichever of
# result/, result_modulo/ actually exists on disk gets included. Filters
# score JSONs and logs by SINCE_DATE so stale leftovers (branch CLAUDE.md
# gotchas #4 and #5) don't sneak in.
#
# Usage:
#   ./sol_gaudi/tar_results.sh [OUTPUT_PATH]
#
# Env overrides:
#   SINCE_DATE   ISO date; only files newer than this are included
#                (default: today 00:00 in the shell's local tz)
#   MODELS       Space-separated HF-style model dirs
#                (default: "Qwen_Qwen3-8B Qwen_Qwen3-14B Qwen_Qwen3-32B")

set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/config.env" 2>/dev/null || source "${SCRIPT_DIR}/config.env.example"

GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

SINCE_DATE="${SINCE_DATE:-$(date +%Y-%m-%d)}"
MODELS="${MODELS:-Qwen_Qwen3-8B Qwen_Qwen3-14B Qwen_Qwen3-32B}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${1:-/scratch/${USER}/bfcl_sweep_${STAMP}.tar.gz}"
LIST_FILE="$(mktemp /tmp/bfcl_files_XXXXXX.txt)"
trap 'rm -f "${LIST_FILE}"' EXIT

cd "${REPO_ROOT}" || { echo -e "${RED}Cannot cd ${REPO_ROOT}${NC}" >&2; exit 1; }

echo -e "${BLUE}Bundling BFCL Gaudi results${NC}"
echo "  REPO_ROOT  : ${REPO_ROOT}"
echo "  SINCE_DATE : ${SINCE_DATE}"
echo "  MODELS     : ${MODELS}"
echo "  OUTPUT     : ${OUT}"
echo

found_baseline=0
found_modulo=0

# 1) Baseline generations — result/<Model>/
for m in ${MODELS}; do
    dir="berkeley-function-call-leaderboard/result/${m}"
    if [[ -d "${dir}" ]]; then
        find "${dir}" -type f >> "${LIST_FILE}"
        found_baseline=1
    fi
done

# 2) Modulo generations — result_modulo/<Model>/
for m in ${MODELS}; do
    dir="berkeley-function-call-leaderboard/result_modulo/${m}"
    if [[ -d "${dir}" ]]; then
        find "${dir}" -type f >> "${LIST_FILE}"
        found_modulo=1
    fi
done

# 3) Fresh per-category score JSONs (strips baseline-vs-modulo contamination
#    in shared score/<Model>/ dir per gotcha #4)
for m in ${MODELS}; do
    dir="berkeley-function-call-leaderboard/score/${m}"
    if [[ -d "${dir}" ]]; then
        find "${dir}" -name "BFCL_v4_*_score.json" -newermt "${SINCE_DATE}" >> "${LIST_FILE}"
    fi
done

# 4) Aggregate CSVs (hybrid modulo+baseline per gotcha #4, but useful context)
find berkeley-function-call-leaderboard/score -maxdepth 1 -name "data_*.csv" -type f >> "${LIST_FILE}" 2>/dev/null || true

# 5) Job logs newer than SINCE_DATE (baseline + modulo — the * matches both)
find sol_gaudi/logs -maxdepth 1 -type f \
     \( -name "bfcl_qwen3_*.out" -o -name "bfcl_qwen3_*.err" \) \
     -newermt "${SINCE_DATE}" >> "${LIST_FILE}" 2>/dev/null || true

# 6) SLURM scripts (all that exist — both baseline and modulo variants)
find sol_gaudi/slurm -maxdepth 1 -name "bfcl_*_gaudi.slurm" -type f >> "${LIST_FILE}" 2>/dev/null || true

# 7) Modulo config YAMLs (only exist on LLM-Modulo_gaudi, harmless otherwise)
if [[ -d berkeley-function-call-leaderboard/configs ]]; then
    find berkeley-function-call-leaderboard/configs -maxdepth 1 -type f \
         \( -name "modulo_*.yaml" -o -name "smoketest.yaml" \) >> "${LIST_FILE}" 2>/dev/null || true
fi

# Dedupe and sanity-check
sort -u -o "${LIST_FILE}" "${LIST_FILE}"
count=$(wc -l < "${LIST_FILE}")
if [[ "${count}" -eq 0 ]]; then
    echo -e "${RED}No files matched — check SINCE_DATE=${SINCE_DATE} and MODELS.${NC}" >&2
    echo -e "${RED}Did any jobs actually finish and write to result*/?${NC}" >&2
    exit 1
fi

# What kinds of runs are we bundling?
kinds=()
[[ "${found_baseline}" -eq 1 ]] && kinds+=("baseline")
[[ "${found_modulo}"   -eq 1 ]] && kinds+=("modulo")
echo -e "${BLUE}Detected: ${kinds[*]:-(nothing in result*/ — only score/logs)}${NC}"

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
echo
echo "Pull to laptop with:"
echo "  scp ${USER}@login.sol.rc.asu.edu:${OUT} ~/Desktop/"
echo
echo "Then extract:"
echo "  mkdir -p ~/Desktop/$(basename "${OUT}" .tar.gz)"
echo "  tar xzf ~/Desktop/$(basename "${OUT}") -C ~/Desktop/$(basename "${OUT}" .tar.gz)"
