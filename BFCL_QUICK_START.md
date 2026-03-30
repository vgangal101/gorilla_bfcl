# BFCL Evaluation - Quick Start Setup Guide

## Prerequisites

- Python 3.10+
- Git
- For self-hosted models: CUDA-capable GPU (optional)
- API keys for proprietary models you want to evaluate

---

## Step-by-Step Setup

### 1. Navigate to BFCL Directory
```bash
cd /Users/vgangal/phd_research_workspace/gorilla_bfcl/berkeley-function-call-leaderboard
```

### 2. Create and Activate Python Environment (Recommended)
```bash
# Create conda environment
conda create -n bfcl python=3.10
conda activate bfcl

# Or use venv
python3.10 -m venv bfcl_env
source bfcl_env/bin/activate
```

### 3. Install BFCL Package

#### Option A: Development Installation (Recommended for exploration/modifications)
```bash
pip install -e .
```

#### Option B: PyPI Installation (For just running evaluations)
```bash
pip install bfcl-eval
# Then set environment variable
export BFCL_PROJECT_ROOT=/Users/vgangal/phd_research_workspace/gorilla_bfcl
```

### 4. Setup Configuration Files

#### Copy and Configure `.env` file
```bash
# Copy example
cp bfcl_eval/.env.example .env

# Edit .env and add API keys for models you want to test
nano .env
```

**Minimal `.env` example:**
```bash
# OpenAI
OPENAI_API_KEY=sk-proj-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Google
GOOGLE_API_KEY=...

# Others (as needed)
# COHERE_API_KEY=...
# MISTRAL_API_KEY=...
# GROQ_API_KEY=...
```

### 5. Verify Installation
```bash
# Check CLI is working
bfcl --version

# List available models
bfcl models | head -20

# List test categories
bfcl test-categories
```

---

## Your First Evaluation (5 minutes)

### Quick Test: Evaluate GPT-4o on Simple Python Tests

```bash
# Step 1: Generate responses (takes 1-2 minutes)
bfcl generate \
  --model gpt-4o-2024-11-20-FC \
  --test-category simple_python \
  --num-threads 1

# Step 2: Evaluate (takes 10-30 seconds)
bfcl evaluate \
  --model gpt-4o-2024-11-20-FC \
  --test-category simple_python

# Step 3: Check results
cat score/data_overall.csv
```

**Expected Output:**
- Results in: `result/gpt-4o-2024-11-20-FC/BFCL_v3_simple_python_result.json`
- Scores in: `score/gpt-4o-2024-11-20-FC/BFCL_v3_simple_python_score.json`
- Summary CSV: `score/data_overall.csv`

---

## Practical Workflow Examples

### Example 1: Compare Two Models
```bash
# Generate for both models
bfcl generate \
  --model gpt-4o-2024-11-20-FC,claude-3-5-sonnet-20241022-FC \
  --test-category simple_python,parallel

# Evaluate both
bfcl evaluate \
  --model gpt-4o-2024-11-20-FC,claude-3-5-sonnet-20241022-FC \
  --test-category simple_python,parallel

# View comparison
cat score/data_overall.csv
```

### Example 2: Comprehensive Evaluation
```bash
# Generate on all scoring categories (takes 5-10 minutes per model)
bfcl generate \
  --model gpt-4o-2024-11-20-FC \
  --test-category all_scoring

# Evaluate
bfcl evaluate \
  --model gpt-4o-2024-11-20-FC \
  --test-category all_scoring

# Check detailed breakdown
echo "=== Non-Live Categories ==="
cat score/data_non_live.csv

echo "=== Multi-Turn Categories ==="
cat score/data_multi_turn.csv

echo "=== Live Categories ==="
cat score/data_live.csv
```

### Example 3: Multi-Turn Conversation Testing
```bash
# Generate responses for multi-turn scenarios
bfcl generate \
  --model gpt-4o-2024-11-20-FC \
  --test-category multi_turn

# Evaluate
bfcl evaluate \
  --model gpt-4o-2024-11-20-FC \
  --test-category multi_turn

# View results
cat score/gpt-4o-2024-11-20-FC/BFCL_v3_multi_turn_base_score.json | python3 -m json.tool
```

### Example 4: Test Specific Cases Only
```bash
# Create targeting file
cat > test_case_ids_to_generate.json << 'EOF'
{
  "simple_python": ["simple_python_1", "simple_python_5", "simple_python_10"],
  "parallel": ["parallel_2", "parallel_8"]
}
EOF

# Generate only those
bfcl generate \
  --model gpt-4o-2024-11-20-FC \
  --run-ids

# Evaluate with partial flag
bfcl evaluate \
  --model gpt-4o-2024-11-20-FC \
  --test-category simple_python,parallel \
  --partial-eval
```

---

## Understanding Results

### Files Generated

After running `bfcl generate`:
```
result/
└── MODEL_NAME/
    ├── BFCL_v3_TEST_CATEGORY_result.json     ← Raw responses
    └── inference_logs.json                   ← Detailed logs
```

After running `bfcl evaluate`:
```
score/
├── MODEL_NAME/
│   ├── BFCL_v3_TEST_CATEGORY_score.json     ← Score details
├── data_overall.csv                         ← Summary (all models)
├── data_non_live.csv                        ← Non-live breakdown
├── data_live.csv                            ← Live breakdown
└── data_multi_turn.csv                      ← Multi-turn breakdown
```

### Reading Score Files

**View JSON scores:**
```bash
cat score/gpt-4o-2024-11-20-FC/BFCL_v3_simple_python_score.json | python3 -m json.tool | head -50
```

**Key metrics in JSON:**
- `accuracy`: Overall accuracy for category (0.0-1.0)
- `total_tests`: Number of test cases
- `correct`: Number of correct responses
- `test_case_results`: Array with per-test details

**View CSV summary:**
```bash
# Overall scores
head -5 score/data_overall.csv

# With specific columns (use awk)
cut -d',' -f1-5 score/data_non_live.csv | column -t -s','
```

---

## Debugging & Troubleshooting

### Issue: "Module not found" errors
```bash
# Reinstall in editable mode
pip install -e .
```

### Issue: API key not found
```bash
# Check .env file exists in correct location
ls -la .env

# Verify keys are set
grep -i "openai\|anthropic" .env
```

### Issue: CUDA out of memory (self-hosted models)
```bash
# Reduce memory usage
bfcl generate ... \
  --gpu-memory-utilization 0.6  # Default is 0.9

# Or use fewer GPUs
bfcl generate ... \
  --num-gpus 1
```

### Issue: Rate limit errors on API models
```bash
# Reduce number of threads
bfcl generate ... \
  --num-threads 1  # Default is 1 already

# Add delay between requests (if supported)
# Check LOG_GUIDE.md for more info
```

### Verbose Debugging
```bash
# Generate with detailed logging
bfcl generate \
  --model MODEL_NAME \
  --test-category TEST_CATEGORY \
  --include-input-log

# Check logs
cat result/MODEL_NAME/inference_logs.json | python3 -m json.tool
```

---

## Useful Commands

### View Available Models
```bash
# List all models
bfcl models

# Find specific model
bfcl models | grep -i "claude\|gpt-4o"

# Count models
bfcl models | wc -l
```

### View Test Categories
```bash
# List all categories
bfcl test-categories

# Find specific category
bfcl test-categories | grep -i "multi_turn"
```

### Check Previous Results
```bash
# List generated results
ls result/*/

# Check evaluation scores  
ls score/*/

# View which models have been evaluated
ls result/
```

### Clean Up & Start Fresh
```bash
# Remove all results for a model
rm -rf result/MODEL_NAME/
rm -rf score/MODEL_NAME/

# Or start completely fresh
rm -rf result/ score/
```

---

## Advanced: Custom Model Evaluation

### Using Pre-existing Server
```bash
# Set environment variables for your server
export LOCAL_SERVER_ENDPOINT=your-server.com
export LOCAL_SERVER_PORT=8000

# Generate with skip-server-setup
bfcl generate \
  --model custom-model-name \
  --test-category simple_python \
  --skip-server-setup
```

### Using OpenAI-compatible Endpoint (Remote)
```bash
# Set in .env
export REMOTE_OPENAI_BASE_URL=https://your-server/v1
export REMOTE_OPENAI_API_KEY=your-key
export REMOTE_OPENAI_TOKENIZER_PATH=/path/to/tokenizer

# Generate
bfcl generate \
  --model custom-model \
  --test-category simple_python \
  --skip-server-setup
```

---

## Optimization Tips

### Speed Up Evaluations
1. **Use multiple threads for API calls** (if rate limits allow):
   ```bash
   bfcl generate ... --num-threads 4
   ```

2. **Use smaller test subset first** for testing:
   ```bash
   bfcl generate ... --test-category simple_python
   ```

3. **Use sglang for self-hosted models** (if GPU supports SM 80+):
   ```bash
   pip install -e .[oss_eval_sglang]
   bfcl generate ... --backend sglang
   ```

### Organize Results
```bash
# Create project structure
mkdir evaluations/{baseline,experimental}

# Run with custom directories
bfcl generate ... --result-dir evaluations/baseline
bfcl evaluate ... --score-dir evaluations/baseline
```

---

## Analysis Tips

### Get Quick Summary
```bash
# Column 1: Model name, Columns 2-5: Key metrics
cut -d',' -f1-5 score/data_overall.csv | head -10
```

### Find Best Performing Model
```bash
# Sort by overall accuracy (column 2)
sort -t',' -k2 -nr score/data_overall.csv | head -5
```

### Analyze Failures
```bash
# Check which test IDs failed for debugging
python3 -c "
import json
with open('score/gpt-4o-2024-11-20-FC/BFCL_v3_simple_python_score.json') as f:
    data = json.load(f)
    failed = [r for r in data['test_case_results'] if not r['correct']]
    for r in failed[:5]:
        print(f\"Failed: {r['id']}\")
"
```

---

## Next Steps

1. **Run your first evaluation** - Follow "Your First Evaluation" section
2. **Compare models** - Use Example 1 to compare two models
3. **Explore test categories** - Run different categories to understand what's tested
4. **Analyze results** - Check detailed scores in JSON files
5. **Contribute** - Add new models or improve the leaderboard!

---

## Important Files to Know

| File/Directory | Purpose |
|---|---|
| `.env` | API keys and configuration |
| `result/` | Generated LLM responses |
| `score/` | Evaluation scores |
| `bfcl_eval/model_handler/` | Model integration code |
| `bfcl_eval/eval_checker/` | Evaluation logic |
| `LOG_GUIDE.md` | Understanding inference logs |

---

## Support

- **Documentation**: See main `README.md` in berkeley-function-call-leaderboard/
- **Issues**: Check GitHub issue tracker
- **Discord**: Join community at https://discord.gg/grXXvj9Whz
- **Contact**: Email huanzhimao@berkeley.edu

---

Last Updated: March 2026
