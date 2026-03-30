# BFCL SLURM Evaluation Guide - GPT-4o-mini

## Overview

This guide explains how to use the SLURM script to evaluate GPT-4o-mini on the Berkeley Function Call Leaderboard in a cluster environment.

---

## Prerequisites

- Access to a SLURM cluster (or local machine with SLURM)
- OpenAI API key with GPT-4o-mini access
- BFCL package installed (see main README)
- Conda/venv environment configured

---

## Step 1: Set Up API Key

### Option A: Interactive Setup (Recommended)

```bash
cd /Users/vgangal/phd_research_workspace/gorilla_bfcl

# Make setup script executable
chmod +x setup_api_key.sh

# Run the setup script
./setup_api_key.sh
```

This will:
1. Prompt for your OpenAI API key
2. Create a secure file at `~/.ssh/gpt4o_api_key.sh`
3. Set permissions to 600 (read/write for owner only)

### Option B: Manual Setup

Create `~/.ssh/gpt4o_api_key.sh`:

```bash
#!/bin/bash
export OPENAI_API_KEY="sk-proj-your-api-key-here"
```

Secure it:
```bash
chmod 600 ~/.ssh/gpt4o_api_key.sh
```

---

## Step 2: Configure SLURM Script (Optional)

Edit `evaluate_gpt4o_mini.slurm` to customize:

### Common Settings to Adjust

```bash
# SLURM configuration
#SBATCH --time=04:00:00              # Increase if evaluating many categories
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4            # Adjust as needed
#SBATCH --mem=16G                    # Memory for job
#SBATCH --partition=gpu              # Change partition as needed
#SBATCH --gres=gpu:0                 # 0 = no GPU needed for API models

# BFCL settings (around line 40)
MODEL_NAME="gpt-4o-mini-2024-07-18-FC"
TEST_CATEGORY="all_scoring"           # Options: simple_python, parallel, multi_turn, all_scoring, etc.
NUM_THREADS=2                         # Threads for parallel API calls
```

### Test Category Options

```bash
# Single category (fast, 1-2 minutes)
TEST_CATEGORY="simple_python"

# Multiple categories (specify comma-separated)
TEST_CATEGORY="simple_python,parallel"

# All scoring categories (comprehensive, 10-15 minutes)
TEST_CATEGORY="all_scoring"

# Specific categories
# - simple_python, simple_java, simple_javascript
# - parallel, multiple, parallel_multiple
# - irrelevance
# - multi_turn_base, multi_turn_miss_func, multi_turn_miss_param, multi_turn_long_context
# - memory_kv, memory_vector, memory_rec_sum
# - web_search_base, web_search_no_snippet
# - live_simple, live_multiple, live_parallel, live_parallel_multiple
# - format_sensitivity
```

### Time Estimation

| Test Category | Estimated Time | Notes |
|---|---|---|
| `simple_python` | 1-2 min | Quick test |
| `simple_python,parallel` | 2-3 min | Basic categories |
| `all_scoring` | 10-15 min | All scoring categories |
| `multi_turn` | 5-10 min | Multi-turn only |

---

## Step 3: Submit SLURM Job

### Submit the Job

```bash
cd /Users/vgangal/phd_research_workspace/gorilla_bfcl

# Make script executable
chmod +x evaluate_gpt4o_mini.slurm

# Submit to SLURM
sbatch evaluate_gpt4o_mini.slurm
```

You'll see output like:
```
Submitted batch job 12345678
```

Keep note of the job ID for monitoring.

### Quick Test First

To test quickly before running full evaluation:

```bash
# Edit the script temporarily
sed -i 's/TEST_CATEGORY="all_scoring"/TEST_CATEGORY="simple_python"/' evaluate_gpt4o_mini.slurm

# Submit
sbatch evaluate_gpt4o_mini.slurm

# Monitor
squeue -u $USER
```

---

## Step 4: Monitor Job Progress

### Check Job Status

```bash
# View your running jobs
squeue -u $USER

# Look for your job (example output):
# JOBID     PARTITION     NAME     USER ST       TIME  NODES CPUS
# 12345678  gpu           bfcl...  vgangal R       2:15      1    4
```

### Watch Job Output

```bash
# View current output (shows live updates)
tail -f logs/bfcl_gpt4o_mini_12345678.out

# View stderr separately
tail -f logs/bfcl_gpt4o_mini_12345678.err
```

### Job Status Legend

| Status | Meaning |
|---|---|
| PD | Pending (waiting for resources) |
| R | Running |
| CA | Cancelled |
| CD | Completed |
| F | Failed |
| TO | Timeout |

---

## Step 5: View Results

### After Job Completes

Check the status:
```bash
# View last few lines of output
tail -20 logs/bfcl_gpt4o_mini_12345678.out

# Check for "EVALUATION COMPLETE" message
grep "EVALUATION COMPLETE" logs/bfcl_gpt4o_mini_*.out
```

### Results Location

```
score/
├── data_overall.csv              ← Main results (all models)
├── data_non_live.csv
├── data_live.csv
├── data_multi_turn.csv
└── gpt-4o-mini-2024-07-18-FC/
    ├── BFCL_v3_simple_python_score.json
    ├── BFCL_v3_parallel_score.json
    └── ... (one per category)

result/
└── gpt-4o-mini-2024-07-18-FC/
    ├── BFCL_v3_simple_python_result.json
    ├── BFCL_v3_parallel_result.json
    ├── inference_logs.json
    └── ... (one per category)
```

### View Score Summary

```bash
# Overall scores (CSV format)
cat score/data_overall.csv

# Pretty print with column formatting
column -t -s',' score/data_overall.csv

# View just model name and overall accuracy (columns 1-2)
cut -d',' -f1-2 score/data_overall.csv

# View complete scores for one model
cat score/gpt-4o-mini-2024-07-18-FC/BFCL_v3_simple_python_score.json | python3 -m json.tool
```

---

## Common Tasks

### Run Multiple Evaluations in Parallel

Create `evaluate_gpt4o_mini_variants.sh`:

```bash
#!/bin/bash

# Evaluate multiple test categories sequentially
sbatch --job-name=bfcl_simple evaluate_gpt4o_mini.slurm
sbatch --job-name=bfcl_parallel evaluate_gpt4o_mini.slurm
sbatch --job-name=bfcl_multiturn evaluate_gpt4o_mini.slurm

echo "Submitted 3 evaluation jobs"
squeue -u $USER
```

Or modify the script to loop:

```bash
# In evaluate_gpt4o_mini.slurm, change TEST_CATEGORY line to:
for CATEGORY in simple_python parallel multi_turn_base; do
    log "Evaluating category: $CATEGORY"
    TEST_CATEGORY="$CATEGORY"
    # ... rest of generation and evaluation ...
done
```

### Compare with Other Models

Create separate SLURM files:
- `evaluate_gpt4o.slurm` - for GPT-4o full version
- `evaluate_claude.slurm` - for Claude models
- etc.

Then submit all:
```bash
sbatch evaluate_gpt4o.slurm
sbatch evaluate_gpt4o_mini.slurm
sbatch evaluate_claude.slurm

# Monitor all
squeue -u $USER
```

### Cancel a Job

```bash
# Cancel specific job
scancel 12345678

# Cancel all your jobs
scancel -u $USER
```

### Adjust Job after Submission

```bash
# Extend time if job is running out of time
scontrol update JobID=12345678 TimeLimit=06:00:00

# Change priority (if supported)
scontrol update JobID=12345678 Priority=1000
```

---

## Troubleshooting

### Issue: "API key not found"

**Problem**: Script runs but fails with API key error

**Solution**:
```bash
# Verify setup script ran correctly
cat ~/.ssh/gpt4o_api_key.sh

# Should show:
# #!/bin/bash
# export OPENAI_API_KEY="sk-proj-..."

# If not, run setup again
./setup_api_key.sh
```

### Issue: "SLURM partition not found"

**Problem**: Error like "Invalid partition specified"

**Solution**:
```bash
# Check available partitions
sinfo

# Output example:
# PARTITION         AVAIL  TIMELIMIT  NODES  STATE NODELIST
# gpu*                 up 7-00:00:00      4  idle node[1-4]
# cpu                  up 3-00:00:00      8  alloc node[5-12]

# Edit evaluate_gpt4o_mini.slurm:
# Change: #SBATCH --partition=gpu
# To:     #SBATCH --partition=cpu
```

### Issue: "QOSMaxCpuPerUserLimit exceeded"

**Problem**: Too many CPU-intensive jobs

**Solution**:
```bash
# Reduce CPU request in SLURM script:
# #SBATCH --cpus-per-task=2  (was 4)

# Or reduce NUM_THREADS:
# NUM_THREADS=1  (was 2)
```

### Issue: "Timeout exceeded"

**Problem**: Job gets killed before finishing

**Solution**:
```bash
# Increase time limit in evaluate_gpt4o_mini.slurm:
# #SBATCH --time=08:00:00  (was 04:00:00)

# Or reduce test scope:
# TEST_CATEGORY="simple_python"  (was all_scoring)
```

### Issue: Results directory permission denied

**Problem**: Can't write to results

**Solution**:
```bash
# Check directory permissions
ls -ld score/ result/

# Fix directory permissions
chmod 755 score result

# Or re-run with correct BFCL_PROJECT_ROOT
export BFCL_PROJECT_ROOT=/Users/vgangal/phd_research_workspace/gorilla_bfcl
```

### Issue: "Module not found" when running bfcl

**Problem**: BFCL package not in Python path

**Solution**:
Add to `evaluate_gpt4o_mini.slurm` after line ~65:

```bash
# Activate conda environment
conda activate bfcl

# Or activate venv
source venv/bin/activate

# Verify BFCL is installed
pip show bfcl-eval || pip install -e .
```

---

## Performance Tuning

### Speed Up Evaluations

```bash
# Increase threads (for API calls - check rate limits first)
NUM_THREADS=4

# Only evaluate small test subset
TEST_CATEGORY="simple_python"

# Use background job
nohup sbatch evaluate_gpt4o_mini.slurm > /dev/null 2>&1 &
```

### Reduce Resource Usage

```bash
# Lower CPU request
#SBATCH --cpus-per-task=2

# Lower memory request
#SBATCH --mem=8G

# Reduce threads to 1
NUM_THREADS=1
```

---

## Advanced Usage

### Submit from Remote Machine

```bash
# SSH to cluster
ssh cluster.example.com

# Navigate to project
cd /Users/vgangal/phd_research_workspace/gorilla_bfcl

# Submit job
sbatch evaluate_gpt4o_mini.slurm

# Logout (job continues running)
exit
```

### Create Job Dependency Chain

Evaluate multiple models in sequence:

```bash
# Submit first job
JOB1=$(sbatch evaluate_gpt4o_mini.slurm | awk '{print $4}')

# Submit second job to run after first completes
JOB2=$(sbatch --dependency=afterok:$JOB1 evaluate_claude.slurm | awk '{print $4}')

# Submit third job to run after second completes
JOB3=$(sbatch --dependency=afterok:$JOB2 evaluate_llama.slurm | awk '{print $4}')

echo "Submitted dependency chain: $JOB1 -> $JOB2 -> $JOB3"
```

### Email Notifications

Add to SLURM header:

```bash
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your-email@example.com
```

Then job completion will email you.

---

## Useful Commands Summary

```bash
# Setup
./setup_api_key.sh                    # Setup API key

# Submit
sbatch evaluate_gpt4o_mini.slurm      # Submit job

# Monitor
squeue -u $USER                       # View jobs
stat -f logs/bfcl_gpt4o_mini_*.out    # Check output
tail -f logs/bfcl_gpt4o_mini_*.out    # Watch output live

# Results
cat score/data_overall.csv             # View summary scores
ls -la score/gpt-4o-mini-*            # List all scores

# Manage
scancel 12345678                       # Cancel specific job
scancel -u $USER                       # Cancel all jobs
scontrol show job JOBID                # View job details
```

---

## Example Workflows

### Workflow 1: Quick Test → Full Evaluation

```bash
# Step 1: Quick test with simple_python (1-2 min)
sed -i 's/TEST_CATEGORY="all_scoring"/TEST_CATEGORY="simple_python"/' evaluate_gpt4o_mini.slurm
sbatch evaluate_gpt4o_mini.slurm
# ... wait for completion ...

# Step 2: Revert and run full evaluation
sed -i 's/TEST_CATEGORY="simple_python"/TEST_CATEGORY="all_scoring"/' evaluate_gpt4o_mini.slurm
sbatch evaluate_gpt4o_mini.slurm
```

### Workflow 2: Compare Multiple Models

```bash
# Setup (once)
./setup_api_key.sh

# Create separate scripts for each model (gpt4o_mini, claude, llama)
# Copy evaluate_gpt4o_mini.slurm -> evaluate_claude.slurm
# Edit MODEL_NAME in each script

# Submit all in parallel
sbatch evaluate_gpt4o_mini.slurm
sbatch evaluate_claude.slurm
sbatch evaluate_llama.slurm

# Monitor
squeue -u $USER

# Compare results when all done
echo "GPT-4o-mini:" && head -2 score/data_overall.csv | tail -1
echo "Claude:" && grep claude score/data_overall.csv
echo "Llama:" && grep llama score/data_overall.csv
```

### Workflow 3: Selective Category Testing

```bash
# Test each category separately to understand failures
for CATEGORY in simple_python parallel multi_turn_base web_search_base; do
    sed -i "s/TEST_CATEGORY=\".*/TEST_CATEGORY=\"$CATEGORY\"/" evaluate_gpt4o_mini.slurm
    echo "Submitting $CATEGORY..."
    sbatch evaluate_gpt4o_mini.slurm
    sleep 2
done

# Check results as they come in
watch -n 10 "squeue -u $USER; echo ''; tail -1 score/data_overall.csv"
```

---

## Support & Resources

- **Check job output**: `tail -20 logs/bfcl_gpt4o_mini_*.out`
- **SLURM documentation**: `man sbatch`, `man sinfo`
- **BFCL documentation**: See `README.md`, `LOG_GUIDE.md`
- **Get help**: Check logs for detailed error messages

---

Last Updated: March 28, 2026
