#!/bin/bash
# sol_gaudi/manage_bfcl_gaudi.sh
#
# Submit/manage BFCL Gaudi jobs on SOL. Sources config.env for all paths.

set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/config.env" 2>/dev/null || source "${SCRIPT_DIR}/config.env.example"

GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

MODELS=(qwen3_8b qwen3_14b qwen3_32b)

print_usage() {
    cat <<EOF
Usage: ./sol_gaudi/manage_bfcl_gaudi.sh [COMMAND] [ARGS]

Commands:
    submit <model>        Submit a single model (${MODELS[*]})
    submit-all            Submit all models in MODELS
    status [JOB_ID]       squeue/sacct for your job(s)
    logs <JOB_ID>         Tail the .out log for a job
    results [MODEL]       Show score CSVs (filtered to model if given)
    cancel [JOB_ID]       Cancel one job or all your bfcl_*_gaudi jobs
    clean                 Delete logs older than 30 days
    list                  List generated .slurm files
    help                  This message

Environment overrides:
    BFCL_TEST_CATEGORY    Override test categories per submission
    BFCL_NUM_THREADS      Override num-threads per submission

Examples:
    ./sol_gaudi/manage_bfcl_gaudi.sh submit qwen3_4b
    BFCL_TEST_CATEGORY=all_scoring ./sol_gaudi/manage_bfcl_gaudi.sh submit qwen3_32b
    ./sol_gaudi/manage_bfcl_gaudi.sh submit-all
    ./sol_gaudi/manage_bfcl_gaudi.sh status
    ./sol_gaudi/manage_bfcl_gaudi.sh logs 12345
EOF
}

slurm_file_for() {
    local model=$1
    echo "${SLURM_DIR}/bfcl_${model}_gaudi.slurm"
}

ensure_slurm_files() {
    if ! ls "${SLURM_DIR}"/bfcl_*_gaudi.slurm >/dev/null 2>&1; then
        echo -e "${YELLOW}No .slurm files found — running generator${NC}"
        # Login default python3 may be <3.9; activate env for the generator.
        if command -v module >/dev/null 2>&1; then
            module load "${MAMBA_MODULE}" 2>/dev/null || true
        fi
        # shellcheck disable=SC1091
        source activate "${BFCL_GAUDI_ENV}" 2>/dev/null || true
        python3 "${SCRIPT_DIR}/generate_bfcl_scripts.py"
    fi
}

submit_one() {
    local model=$1
    local slurm
    slurm=$(slurm_file_for "${model}")
    if [[ ! -f "${slurm}" ]]; then
        echo -e "${RED}No SBATCH file for '${model}' at ${slurm}${NC}" >&2
        echo "Known models: ${MODELS[*]}" >&2
        return 1
    fi
    mkdir -p "${LOGS_DIR}"
    echo -e "${BLUE}Submitting ${model}${NC} (test-category=${BFCL_TEST_CATEGORY})"
    local job_id
    job_id=$(BFCL_TEST_CATEGORY="${BFCL_TEST_CATEGORY}" \
             BFCL_NUM_THREADS="${BFCL_NUM_THREADS}" \
             sbatch --parsable \
                    --chdir "${REPO_ROOT}" \
                    --export=ALL,BFCL_TEST_CATEGORY,BFCL_NUM_THREADS \
                    "${slurm}")
    echo -e "${GREEN}Submitted:${NC} ${model} -> job ${job_id}"
    echo "  watch: ./sol_gaudi/manage_bfcl_gaudi.sh status ${job_id}"
    echo "  logs:  ./sol_gaudi/manage_bfcl_gaudi.sh logs ${job_id}"
}

submit_all() {
    ensure_slurm_files
    for m in "${MODELS[@]}"; do
        submit_one "${m}" || true
    done
}

check_status() {
    local job_id=${1:-}
    if [[ -n "${job_id}" ]]; then
        echo -e "${BLUE}squeue for job ${job_id}:${NC}"
        squeue -j "${job_id}" -o "%.12i %.12P %.25j %.8T %.10M %.9l %.6D %R" 2>/dev/null \
            || echo -e "${YELLOW}Not in queue (may be completed)${NC}"
        echo
        echo -e "${BLUE}sacct:${NC}"
        sacct -j "${job_id}" --format="JobID,JobName%25,AllocTRES%40,Elapsed,State" 2>/dev/null || true
    else
        echo -e "${BLUE}Your BFCL Gaudi jobs:${NC}"
        squeue -u "${USER}" --name="$(IFS=,; echo "bfcl_${MODELS[*]}_gaudi" | sed 's/ /,/g')" \
               -o "%.12i %.25j %.8T %.10M %.9l %R" 2>/dev/null | head -40
        squeue -u "${USER}" -o "%.12i %.25j %.8T %.10M %.9l %R" 2>/dev/null \
            | awk 'NR==1 || /bfcl_.*_gaudi/'
    fi
}

show_logs() {
    local job_id=${1:-}
    local log_file
    if [[ -n "${job_id}" ]]; then
        log_file=$(ls -t "${LOGS_DIR}"/bfcl_*_"${job_id}".out 2>/dev/null | head -1)
    else
        log_file=$(ls -t "${LOGS_DIR}"/bfcl_*.out 2>/dev/null | head -1)
    fi
    if [[ -z "${log_file}" || ! -f "${log_file}" ]]; then
        echo -e "${YELLOW}No log file found for${NC} ${job_id:-<latest>}"
        return 1
    fi
    echo -e "${BLUE}=== ${log_file} ===${NC}"
    tail -100 "${log_file}"
}

show_results() {
    local filter=${1:-}
    local score_dir="${BFCL_ROOT}/score"
    if [[ ! -d "${score_dir}" ]]; then
        echo -e "${YELLOW}No score directory at ${score_dir}${NC}"
        return 1
    fi
    for csv in data_overall.csv data_non_live.csv data_live.csv data_multi_turn.csv; do
        local path="${score_dir}/${csv}"
        [[ -f "${path}" ]] || continue
        echo -e "${BLUE}=== ${csv} ===${NC}"
        if [[ -n "${filter}" ]]; then
            head -1 "${path}"
            grep -i "${filter}" "${path}" || echo "  (no match for '${filter}')"
        else
            head -20 "${path}"
        fi
        echo
    done
}

cancel_jobs() {
    local job_id=${1:-}
    if [[ -n "${job_id}" ]]; then
        echo -e "${YELLOW}scancel ${job_id}${NC}"
        scancel "${job_id}"
        return
    fi
    local ids
    ids=$(squeue -u "${USER}" -h -o "%i %j" | awk '/bfcl_.*_gaudi/ {print $1}')
    if [[ -z "${ids}" ]]; then
        echo -e "${YELLOW}No bfcl_*_gaudi jobs to cancel${NC}"
        return
    fi
    echo -e "${YELLOW}Cancelling: ${ids}${NC}"
    echo "${ids}" | xargs -r scancel
}

clean_old() {
    echo -e "${YELLOW}Deleting logs older than 30 days in ${LOGS_DIR}${NC}"
    find "${LOGS_DIR}" -type f \( -name "*.out" -o -name "*.err" \) -mtime +30 -print -delete 2>/dev/null || true
}

list_scripts() {
    ensure_slurm_files
    echo -e "${BLUE}Generated SBATCH scripts:${NC}"
    ls -lh "${SLURM_DIR}"/bfcl_*_gaudi.slurm 2>/dev/null || echo "  (none)"
}

case "${1:-help}" in
    submit)       shift; submit_one "${1:-}";;
    submit-all)   submit_all;;
    status)       check_status "${2:-}";;
    logs)         show_logs "${2:-}";;
    results)      show_results "${2:-}";;
    cancel)       cancel_jobs "${2:-}";;
    clean)        clean_old;;
    list)         list_scripts;;
    help|-h|--help) print_usage;;
    *)            echo -e "${RED}Unknown command: $1${NC}"; echo; print_usage; exit 1;;
esac
