#!/bin/bash
# Helper script to manage Qwen3-32B evaluation on SOL

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EVAL_SCRIPT="${SCRIPT_DIR}/evaluate_qwen3_32b_vllm.slurm"
LOGS_DIR="${SCRIPT_DIR}/logs"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_usage() {
    cat << EOF
Usage: ./manage_qwen_eval.sh [COMMAND] [OPTIONS]

Commands:
    submit                Submit evaluation job to SLURM
    status [JOB_ID]       Check job status (shows latest if no ID given)
    logs [JOB_ID]         Tail latest logs (shows latest job if no ID given)
    cancel [JOB_ID]       Cancel a running job
    results [JOB_ID]      Show evaluation results (latest if no ID given)
    clean                 Clean up old logs and results
    help                  Show this help message

Examples:
    ./manage_qwen_eval.sh submit
    ./manage_qwen_eval.sh status
    ./manage_qwen_eval.sh logs 12345
    ./manage_qwen_eval.sh cancel 12345
    ./manage_qwen_eval.sh results

EOF
}

get_latest_job_id() {
    squeue -u $USER --name=bfcl_qwen3_32b -h -o "%i" | head -1 || echo ""
}

get_latest_log_file() {
    ls -t "${LOGS_DIR}"/evaluation_*.log 2>/dev/null | head -1 || echo ""
}

submit_job() {
    echo -e "${BLUE}Submitting Qwen3-32B evaluation to SLURM...${NC}"
    
    if [ ! -f "$EVAL_SCRIPT" ]; then
        echo -e "${RED}Error: Evaluation script not found: $EVAL_SCRIPT${NC}"
        exit 1
    fi
    
    mkdir -p "$LOGS_DIR"
    
    JOB_ID=$(sbatch "$EVAL_SCRIPT" | awk '{print $NF}')
    
    echo -e "${GREEN}✓ Job submitted successfully${NC}"
    echo -e "Job ID: ${BLUE}${JOB_ID}${NC}"
    echo ""
    echo "Monitor with:"
    echo "  ./manage_qwen_eval.sh status $JOB_ID"
    echo "  ./manage_qwen_eval.sh logs $JOB_ID"
}

check_status() {
    local job_id=$1
    
    if [ -z "$job_id" ]; then
        job_id=$(get_latest_job_id)
        if [ -z "$job_id" ]; then
            echo -e "${YELLOW}No active jobs found${NC}"
            return
        fi
    fi
    
    echo -e "${BLUE}Job Status (ID: $job_id):${NC}"
    squeue -j "$job_id" -o "%.18i %.15P %.10j %.8T %.10M %.9l %.6D %R" || \
        echo -e "${YELLOW}Job not found in queue (may be completed)${NC}"
    
    # Check if job completed and show result
    if ! squeue -j "$job_id" &>/dev/null; then
        echo ""
        echo -e "${BLUE}Job Information:${NC}"
        sacct -j "$job_id" --format="JobID,AllocCPUS,MaxRSS,Elapsed,State" --units=G
    fi
}

show_logs() {
    local job_id=$1
    local log_file=$2
    
    if [ -z "$log_file" ]; then
        if [ -z "$job_id" ]; then
            log_file=$(get_latest_log_file)
        else
            # Try to find log for specific job ID
            log_file=$(ls -t "${LOGS_DIR}"/evaluation_*.log 2>/dev/null | head -1)
        fi
    fi
    
    if [ -z "$log_file" ] || [ ! -f "$log_file" ]; then
        echo -e "${YELLOW}No log files found${NC}"
        echo "waiting for job to generate logs..."
        return
    fi
    
    echo -e "${BLUE}Showing logs from: $log_file${NC}"
    echo ""
    tail -100 "$log_file"
}

cancel_job() {
    local job_id=$1
    
    if [ -z "$job_id" ]; then
        job_id=$(get_latest_job_id)
        if [ -z "$job_id" ]; then
            echo -e "${YELLOW}No active jobs to cancel${NC}"
            return
        fi
    fi
    
    echo -e "${YELLOW}Cancelling job: $job_id${NC}"
    scancel "$job_id"
    echo -e "${GREEN}✓ Job cancelled${NC}"
}

show_results() {
    local job_id=$1
    local score_dir="${SCRIPT_DIR}/score"
    
    if [ ! -d "$score_dir" ]; then
        echo -e "${YELLOW}Score directory not found. Run evaluation first.${NC}"
        return
    fi
    
    echo -e "${BLUE}=== Evaluation Results ===${NC}"
    echo ""
    
    # Show overall scores
    if [ -f "$score_dir/data_overall.csv" ]; then
        echo -e "${YELLOW}Overall Scores:${NC}"
        head -20 "$score_dir/data_overall.csv"
        echo ""
    fi
    
    # Show multi-turn specific scores
    if [ -f "$score_dir/data_multi_turn.csv" ]; then
        echo -e "${YELLOW}Multi-Turn Scores:${NC}"
        head -10 "$score_dir/data_multi_turn.csv"
        echo ""
    fi
    
    # Show all score files
    if ls "$score_dir"/*.csv 1> /dev/null 2>&1; then
        echo -e "${YELLOW}Available Score Files:${NC}"
        ls -lh "$score_dir"/*.csv
    fi
}

clean_old_files() {
    echo -e "${YELLOW}Cleaning old logs and results...${NC}"
    
    # Keep logs from last 30 days, delete older
    find "$LOGS_DIR" -name "*.log" -type f -mtime +30 -delete -print 2>/dev/null | \
        sed 's/^/  Deleted: /' || true
    
    # Show remaining logs
    echo ""
    echo -e "${BLUE}Remaining log files:${NC}"
    ls -lh "$LOGS_DIR"/*.log 2>/dev/null | tail -10 || echo "  (none)"
}

# Main logic
case "${1:-}" in
    submit)
        submit_job
        ;;
    status)
        check_status "$2"
        ;;
    logs)
        show_logs "$2"
        ;;
    cancel)
        cancel_job "$2"
        ;;
    results)
        show_results "$2"
        ;;
    clean)
        clean_old_files
        ;;
    help|--help|-h)
        print_usage
        ;;
    *)
        if [ -n "$1" ]; then
            echo -e "${RED}Unknown command: $1${NC}"
            echo ""
        fi
        print_usage
        ;;
esac
