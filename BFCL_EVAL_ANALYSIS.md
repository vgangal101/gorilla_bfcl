# Berkeley Function Call Leaderboard (BFCL) - Evaluation Analysis

## Overview

The Berkeley Function Call Leaderboard (BFCL) is a **comprehensive and executable function call evaluation framework** for assessing Large Language Models' (LLMs) ability to invoke functions correctly. It's the first leaderboard that accounts for various forms of function calls, diverse scenarios, and executability.

### Key Features
- **Executable Function Calls**: Actually runs the generated function calls to verify correctness
- **Multi-format Support**: Handles various function invocation styles (parallel, sequential, multi-turn, multi-step)
- **Live Data**: Includes enterprise and community-contributed test cases
- **Agentic Capabilities**: Tests for web search, memory management, and format sensitivity
- **Comprehensive Models Support**: 100+ models (proprietary APIs, self-hosted OSS, and specialized models)

---

## Core Evaluation Workflow

### 1. **Generation Phase** - Generate LLM Responses
```
Generate LLM responses for test cases using specific models
↓
Stores results in: result/MODEL_NAME/BFCL_v3_TEST_CATEGORY_result.json
```

### 2. **Evaluation Phase** - Score the Responses
```
Evaluate generated responses against ground truth
↓
Stores scores in: score/MODEL_NAME/BFCL_v3_TEST_CATEGORY_score.json
↓
Generates CSV summaries for analysis
```

---

## System Architecture

### Project Structure
```
berkeley-function-call-leaderboard/
├── bfcl_eval/              # Core evaluation module
│   ├── eval_checker/       # Evaluation logic
│   │   ├── ast_eval/       # AST-based evaluation for function calls
│   │   ├── multi_turn_eval/ # Multi-turn conversation evaluation
│   │   ├── agentic_eval/   # Agentic feature evaluation
│   │   └── eval_runner.py  # Main evaluation orchestrator
│   ├── model_handler/      # Model inference support
│   │   ├── api_inference/  # API-based models (OpenAI, Anthropic, etc.)
│   │   ├── local_inference/# Self-hosted models (vLLM, sglang)
│   │   └── parser/         # Response parsing (JSON, XML, etc.)
│   ├── _llm_response_generation.py # Generation orchestrator
│   ├── constants/          # Configuration & metadata
│   ├── data/               # Test case data
│   └── scripts/            # Utility scripts
├── data/                   # Function definitions and test data
│   ├── api/                # API specifications
│   └── apizoo/             # Community-contributed APIs
└── openfunctions_evaluation.py # Legacy entry point
```

---

## Test Categories

BFCL provides comprehensive test coverage across multiple dimensions:

### **Single-Turn Tests** (Basic Function Calling)
- **`simple_python`** - Simple Python function calls
- **`simple_java`** - Simple Java function calls  
- **`simple_javascript`** - Simple JavaScript function calls
- **`parallel`** - Multiple parallel function calls
- **`multiple`** - Sequential function calls
- **`parallel_multiple`** - Mixed parallel & sequential calls
- **`irrelevance`** - Calls with irrelevant documentation

### **Multi-Turn Tests** (Conversation Context)
- **`multi_turn_base`** - Basic multi-turn conversations
- **`multi_turn_miss_func`** - Missing function scenarios
- **`multi_turn_miss_param`** - Missing parameter scenarios
- **`multi_turn_long_context`** - Long conversation contexts

### **Agentic Tests** (Advanced Capabilities)
- **`memory_kv`** - Key-value memory operations
- **`memory_vector`** - Vector database operations
- **`memory_rec_sum`** - Recursive summarization memory
- **`web_search_base`** - Web search function calls
- **`web_search_no_snippet`** - Web search without snippets

### **Live Tests** (Community-Contributed)
- **`live_simple`** - Community simple calls
- **`live_multiple`** - Community sequential calls
- **`live_parallel`** - Community parallel calls
- **`live_parallel_multiple`** - Community mixed calls
- **`live_irrelevance`** - Community irrelevance tests
- **`live_relevance`** - Community relevance tests

### **Special Categories**
- **`format_sensitivity`** - Tests system prompt format variations
- **`all_scoring`** - All scoring categories (for leaderboard)
- **`all`** - All test categories including non-scoring

---

## Supported Models

BFCL supports 100+ models across three provider types:

### **Proprietary API Models**
- **OpenAI**: GPT-4.1, GPT-5 series, GPT-4o family
- **Anthropic**: Claude-3.5, Claude-Opus, Claude-Sonnet
- **Google**: Gemini-2.5, Gemini-3 series
- **Others**: DeepSeek, xAI Grok, Cohere Command, Amazon Nova

### **Self-Hosted Open Source Models** (💻 Local)
- **Meta Llama**: Llama-3.1, Llama-3.2, Llama-3.3, Llama-4
- **Specialized**: Functionary, Granite, Arch-Agent, BitAgent
- **General Purpose**: Falcon, Gemma, GLM-4, etc.

### **Function Calling Modes**
- **Function Calling (FC)** - Native tool/function calling support
- **Prompt Mode** - Text-based function invocation via prompts

---

## Environment Setup

### Installation

**For Development (Editable Install)**
```bash
cd berkeley-function-call-leaderboard
pip install -e .
```

**For Self-Hosted Models**
```bash
# Using vLLM (supports older GPUs like T4, V100)
pip install -e .[oss_eval_vllm]

# Using sglang (faster, requires SM 80+ GPUs like A100)
pip install -e .[oss_eval_sglang]
```

**From PyPI**
```bash
pip install bfcl-eval
```

### Configuration Files

**1. `.env` file** - API keys and credentials
```bash
# Copy example
cp bfcl_eval/.env.example .env

# Fill in API keys for models you want to evaluate
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
SERPAPI_API_KEY=...
```

**2. `test_case_ids_to_generate.json`** - Specific test case targeting
```bash
cp bfcl_eval/test_case_ids_to_generate.json.example ./test_case_ids_to_generate.json
```

### Environment Variables

```bash
# Optional: Set project root (defaults to berkeley-function-call-leaderboard)
export BFCL_PROJECT_ROOT=/path/to/project/root

# Generated results will be stored at:
# $BFCL_PROJECT_ROOT/result/
# $BFCL_PROJECT_ROOT/score/
```

---

## Running Evaluations

### **Step 1: Generate LLM Responses**

**For API-based Models** (OpenAI, Anthropic, etc.)
```bash
bfcl generate \
  --model gpt-4o-2024-11-20-FC,claude-3-5-sonnet-20241022-FC \
  --test-category simple_python,parallel,multi_turn_base \
  --num-threads 1
```

**For Self-Hosted Models** (with vLLM)
```bash
bfcl generate \
  --model meta-llama/Llama-3.1-70B-Instruct-FC \
  --test-category simple_python,parallel \
  --backend vllm \
  --num-gpus 1 \
  --gpu-memory-utilization 0.9 \
  --local-model-path /path/to/model
```

**For Self-Hosted Models** (with sglang)
```bash
bfcl generate \
  --model meta-llama/Llama-3.1-70B-Instruct-FC \
  --test-category simple_python,parallel \
  --backend sglang \
  --num-gpus 1
```

**With LoRA Adapters** (vLLM only)
```bash
bfcl generate \
  --model meta-llama/Llama-3.1-70B-Instruct-FC \
  --test-category simple_python \
  --backend vllm \
  --enable-lora \
  --max-lora-rank 128 \
  --lora-modules adapter1="/path/to/lora/adapter1" adapter2="/path/to/lora/adapter2"
```

**For Pre-existing Servers** (OpenAI-compatible endpoints)
```bash
bfcl generate \
  --model custom-model \
  --test-category simple_python \
  --skip-server-setup
```

### **Step 2: Evaluate Generated Responses**

**Full Evaluation** (all test cases)
```bash
bfcl evaluate \
  --model gpt-4o-2024-11-20-FC \
  --test-category simple_python,parallel,multi_turn_base
```

**Partial Evaluation** (specific subset)
```bash
bfcl evaluate \
  --model gpt-4o-2024-11-20-FC \
  --test-category simple_python \
  --partial-eval
```

---

## Output Structure

### Generation Results
```
result/
├── gpt-4o-2024-11-20-FC/
│   ├── BFCL_v3_simple_python_result.json
│   ├── BFCL_v3_parallel_result.json
│   ├── BFCL_v3_multi_turn_base_result.json
│   └── inference_logs.json
```

**Result File Structure:**
```json
[
  {
    "id": "simple_python_1",
    "question": "Call find_user_by_name with name='John'",
    "test_category": "simple_python",
    "llm_response": "[{\"name\": \"find_user_by_name\", \"arguments\": {\"name\": \"John\"}}]",
    "inference_log": {...},
    "execution_log": {...}
  }
]
```

### Evaluation Scores
```
score/
├── overall_scores.csv          # Master leaderboard scores
├── data_overall.csv            # Overall accuracy by model
├── data_non_live.csv           # Non-live test category breakdown
├── data_live.csv               # Live test category breakdown
├── data_multi_turn.csv         # Multi-turn test breakdown
└── gpt-4o-2024-11-20-FC/
    ├── BFCL_v3_simple_python_score.json
    ├── BFCL_v3_parallel_score.json
    └── BFCL_v3_multi_turn_base_score.json
```

**Score File Structure:**
```json
{
  "model_name": "gpt-4o-2024-11-20-FC",
  "test_category": "simple_python",
  "accuracy": 0.95,
  "test_case_results": [
    {
      "id": "simple_python_1",
      "correct": true,
      "score": 1.0
    }
  ],
  "statistics": {
    "total_tests": 100,
    "correct": 95,
    "accuracy": 0.95
  }
}
```

### CSV Summary Files

**`data_overall.csv`** - Used for official leaderboard updates
```
Model,Overall Acc,Non_Live Overall Acc,Live Overall Acc,Multi_Turn Overall Acc,...
gpt-4o-2024-11-20-FC,0.92,0.94,0.88,0.87,...
claude-3-5-sonnet-20241022-FC,0.91,0.93,0.87,0.85,...
```

---

## Evaluation Metrics

### **Accuracy Scoring**
The primary metric is **accuracy** - the percentage of function calls correctly generated by the model.

### **Score Calculation**
1. **Per-test accuracy**: Binary (1.0 if correct, 0.0 if incorrect)
2. **Category accuracy**: Mean of all test accuracies in category
3. **Overall accuracy**: Mean of all category accuracies

### **What Constitutes "Correct"?**
- ✅ Function name matches exactly
- ✅ All required parameters provided with correct values
- ✅ Parameter types are correct (strings, numbers, arrays, etc.)
- ✅ Call sequencing correct (multi-turn contexts)
- ✅ Parallel vs sequential semantics respected
- ✅ Memory state properly maintained (agentic tests)

---

## Key Evaluation Features

### 1. **AST-Based Evaluation**
- Parses generated function calls into Abstract Syntax Trees
- Validates syntactic correctness independent of formatting
- Handles multiple output formats (JSON, XML, etc.)

### 2. **Multi-Turn Conversation Support**
- Evaluates function calls in context of conversation history
- Tests model's ability to maintain state across turns
- Validates long-context understanding

### 3. **Agentic Capabilities**
- **Memory Backend Tests**: KV stores, vector DBs, summarization
- **Web Search Integration**: Real web search execution
- **Stateful Evaluation**: Tests model consistency over multiple calls

### 4. **Format Sensitivity Analysis**
- Tests how system prompt variations affect function calling
- Identifies models robust to prompt changes
- Non-scoring but insightful metrics

### 5. **Comprehensive Logging**
- Inference logs document model reasoning
- Execution logs show function call results
- Useful for debugging and analysis

---

## Quick Start Examples

### Example 1: Evaluate GPT-4o on Simple Tasks
```bash
# Generate responses
bfcl generate \
  --model gpt-4o-2024-11-20-FC \
  --test-category simple_python,simple_java

# Evaluate
bfcl evaluate \
  --model gpt-4o-2024-11-20-FC \
  --test-category simple_python,simple_java
```

### Example 2: Compare Multiple Models
```bash
# Generate for all models at once
bfcl generate \
  --model gpt-4o-2024-11-20-FC,claude-3-5-sonnet-20241022-FC,gemini-2.5-flash-FC \
  --test-category simple_python,parallel,multi_turn_base

# Evaluate all models
bfcl evaluate \
  --model gpt-4o-2024-11-20-FC,claude-3-5-sonnet-20241022-FC,gemini-2.5-flash-FC \
  --test-category simple_python,parallel,multi_turn_base
```

### Example 3: Test Self-Hosted Model
```bash
# Generate with local Llama
bfcl generate \
  --model meta-llama/Llama-3.1-70B-Instruct-FC \
  --test-category simple_python,parallel \
  --backend vllm \
  --num-gpus 2 \
  --gpu-memory-utilization 0.8

# Evaluate
bfcl evaluate \
  --model meta-llama/Llama-3.1-70B-Instruct-FC \
  --test-category simple_python,parallel
```

### Example 4: Specific Test Cases Only
```bash
# Create test_case_ids_to_generate.json
cat > test_case_ids_to_generate.json << 'EOF'
{
  "simple_python": ["simple_python_1", "simple_python_5", "simple_python_10"],
  "parallel": ["parallel_3", "parallel_7"]
}
EOF

# Generate only specified tests
bfcl generate \
  --model gpt-4o-2024-11-20-FC \
  --run-ids

# Evaluate
bfcl evaluate \
  --model gpt-4o-2024-11-20-FC \
  --test-category simple_python,parallel \
  --partial-eval
```

---

## Advanced Usage

### Parallel Inference for API Models
```bash
# Faster inference with multiple threads
bfcl generate \
  --model gpt-4o-2024-11-20-FC \
  --test-category simple_python,parallel,multi_turn_base \
  --num-threads 4
```
*Note: Adjust thread count based on API rate limits*

### Custom Result/Score Directories
```bash
bfcl generate \
  --model gpt-4o-2024-11-20-FC \
  --test-category simple_python \
  --result-dir /custom/results

bfcl evaluate \
  --model gpt-4o-2024-11-20-FC \
  --test-category simple_python \
  --score-dir /custom/scores
```

### Verbose Logging for Debugging
```bash
bfcl generate \
  --model gpt-4o-2024-11-20-FC \
  --test-category simple_python \
  --include-input-log
```

### Remote Server Setup
```bash
# Set in .env
export REMOTE_OPENAI_BASE_URL=https://your-vllm-server.com/v1
export REMOTE_OPENAI_API_KEY=your-api-key
export REMOTE_OPENAI_TOKENIZER_PATH=/path/to/tokenizer

# Generate against remote server
bfcl generate \
  --model custom-model \
  --test-category simple_python \
  --skip-server-setup
```

---

## Workflow for Research

### Typical Research Workflow

1. **Setup Phase**
   ```bash
   # Install and configure
   pip install -e .
   cp bfcl_eval/.env.example .env
   # Fill in .env with API keys
   ```

2. **Baseline Evaluation**
   ```bash
   # Establish baseline on your model of interest
   bfcl generate --model MODEL_NAME --test-category all_scoring
   bfcl evaluate --model MODEL_NAME --test-category all_scoring
   ```

3. **Analysis Phase**
   ```bash
   # Analyze results
   # Check score/data_overall.csv and category-specific scores
   # Review inference_logs for error patterns
   ```

4. **Targeted Evaluation**
   ```bash
   # Focus on failing categories
   bfcl generate --model MODEL_NAME --test-category failing_category
   bfcl evaluate --model MODEL_NAME --test-category failing_category
   ```

5. **Comparison Phase**
   ```bash
   # Compare against baseline models
   bfcl generate --model baseline_model,your_model --test-category all_scoring
   bfcl evaluate --model baseline_model,your_model --test-category all_scoring
   ```

---

## CLI Commands Reference

```bash
# List all available models
bfcl models

# List all test categories
bfcl test-categories

# Generate responses
bfcl generate --model MODEL_NAME --test-category TEST_CATEGORY [options]

# Evaluate responses
bfcl evaluate --model MODEL_NAME --test-category TEST_CATEGORY [options]

# View results
bfcl results --model MODEL_NAME --test-category TEST_CATEGORY

# View scores
bfcl scores --model MODEL_NAME --test-category TEST_CATEGORY

# Check CLI version
bfcl version
```

---

## Contributing Models

To add a new model to BFCL:

1. Review `bfcl_eval/model_handler/base_handler.py` (API) or 
   `bfcl_eval/model_handler/local_inference/base_oss_handler.py` (local)
2. Implement handler class with your model
3. Update `bfcl_eval/constants/model_config.py`
4. Submit PR to [Gorilla GitHub](https://github.com/ShishirPatil/gorilla)

---

## Resources

- **Live Leaderboard**: https://gorilla.cs.berkeley.edu/leaderboard.html
- **GitHub**: https://github.com/ShishirPatil/gorilla
- **Discord**: https://discord.gg/grXXvj9Whz (use #leaderboard channel)
- **Blog**: https://gorilla.cs.berkeley.edu/blogs/
- **Contact**: huanzhimao@berkeley.edu

---

## Summary

The Berkeley Function Call Leaderboard provides a comprehensive evaluation framework for LLM function calling capabilities. With `100+` models, diverse test categories, and both API and self-hosted support, it enables rigorous comparison of function calling performance across the LLM landscape.

**Key Takeaways:**
- Generate responses with `bfcl generate`
- Evaluate results with `bfcl evaluate`
- Analyze scores in `score/` directory
- Supports API models, self-hosted OSS, and custom endpoints
- Covers single-turn, multi-turn, and agentic scenarios
