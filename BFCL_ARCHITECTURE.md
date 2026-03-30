# BFCL Architecture & Internals

## High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      BFCL Evaluation Pipeline                   │
└─────────────────────────────────────────────────────────────────┘

INPUT PHASE
───────────
  Test Data (.jsonl, .json)
       │
       ├─→ simple_python (Python functions)
       ├─→ parallel (Multiple calls)
       ├─→ multi_turn (Conversations)
       ├─→ web_search (Web search calls)
       └─→ memory (Memory operations)

GENERATION PHASE
────────────────
  ┌────────────────────────────────────────┐
  │  Test Case Preparation                 │
  │  └─ Load test data                    │
  │  └─ Parse function definitions        │
  │  └─ Build prompts/tool definitions    │
  └────────────────────────────────────────┘
              │
              ▼
  ┌────────────────────────────────────────┐
  │  Model Routing                         │
  │  └─ API Models (OpenAI, Anthropic)    │
  │  └─ Local Models (vLLM, sglang)       │
  │  └─ Remote Endpoints                   │
  └────────────────────────────────────────┘
              │
              ▼
  ┌────────────────────────────────────────┐
  │  Model Inference                       │
  │  └─ Function Calling (FC) mode        │
  │  └─ Prompt mode                        │
  │  └─ Multi-turn conversation            │
  │  └─ Context management                 │
  └────────────────────────────────────────┘
              │
              ▼
  ┌────────────────────────────────────────┐
  │  Response Parsing                      │
  │  └─ JSON extraction                    │
  │  └─ XML extraction                     │
  │  └─ Structure validation                │
  │  └─ Normalization                      │
  └────────────────────────────────────────┘
              │
              ▼
  result/ (Generated responses)

EVALUATION PHASE
────────────────
  ┌────────────────────────────────────────┐
  │  Response Validation                   │
  │  └─ Parse generated function calls    │
  │  └─ Extract function names & params   │
  │  └─ Validate structure                 │
  └────────────────────────────────────────┘
              │
              ▼
  ┌────────────────────────────────────────┐
  │  AST-Based Comparison                  │
  │  └─ Build AST for generated calls     │
  │  └─ Build AST for ground truth        │
  │  └─ Compare semantically               │
  │  └─ Ignore formatting differences     │
  └────────────────────────────────────────┘
              │
              ▼
  ┌────────────────────────────────────────┐
  │  Scoring & Aggregation                 │
  │  └─ Per-test: 1.0 (correct) / 0.0     │
  │  └─ Per-category: Mean accuracy        │
  │  └─ Overall: Mean of categories        │
  │  └─ Generate statistics                │
  └────────────────────────────────────────┘
              │
              ▼
  score/ (Evaluation results + CSVs)
```

---

## Core Components

### 1. Model Handler System (`bfcl_eval/model_handler/`)

**Purpose**: Abstracts model inference across different providers

```python
# Base interfaces
bfcl_eval/model_handler/
├── base_handler.py                    # Abstract ModelHandler
├── api_inference/
│   ├── base_api_handler.py            # Abstract API handler
│   ├── openai_handler.py              # OpenAI/Azure
│   ├── anthropic_handler.py           # Anthropic Claude
│   ├── google_handler.py              # Google Gemini
│   ├── cohere_handler.py              # Cohere Command
│   └── ...other API handlers
├── local_inference/
│   ├── base_oss_handler.py            # Abstract OSS handler
│   ├── vllm_handler.py                # vLLM backend
│   ├── sglang_handler.py              # sglang backend
│   └── ...other local handlers
└── parser/
    ├── function_call_parser.py        # Parse function calls
    ├── json_parser.py                 # JSON parsing
    ├── xml_parser.py                  # XML parsing
    └── response_normalizer.py         # Normalize responses
```

**Key Classes:**

```python
# Base ModelHandler (abstract)
class ModelHandler:
    def inference(self, test_case, num_threads=1) -> List[str]:
        """Generate LLM responses"""
        pass
    
    def get_model_config(self) -> Dict:
        """Get model configuration"""
        pass

class ResponseParser:
    def parse(self, response: str) -> ParsedFunctionCall:
        """Parse LLM response into function calls"""
        pass
```

### 2. Evaluation Checker System (`bfcl_eval/eval_checker/`)

**Purpose**: Compare generated vs. ground truth function calls

```python
bfcl_eval/eval_checker/
├── eval_runner.py                    # Main evaluation orchestrator
├── eval_runner_helper.py             # Helper functions
├── ast_eval/                         # AST-based evaluation
│   ├── ast_function_call.py          # Function call AST nodes
│   ├── ast_evaluator.py              # Comparison logic
│   └── ast_generator.py              # Generate ASTs
├── multi_turn_eval/                  # Multi-turn evaluation
│   ├── multi_turn_evaluator.py       # Conversation evaluation
│   ├── func_source_code/             # Function implementations
│   │   ├── java_functions.py
│   │   ├── python_functions.py
│   │   ├── javascript_functions.py
│   │   └── web_search.py
│   └── state_manager.py              # Conversation state
└── agentic_eval/                     # Agentic capabilities
    ├── agentic_evaluator.py          # Agentic evaluation logic
    ├── memory_backend/               # Memory implementations
    │   ├── kv_memory.py
    │   ├── vector_memory.py
    │   └── summarization_memory.py
    └── web_search_eval.py            # Web search evaluation
```

**Key Classes:**

```python
# AST-based evaluation
class ASTEvaluator:
    def evaluate(self, 
                 generated_calls: List[FunctionCall],
                 ground_truth_calls: List[FunctionCall]) -> Score:
        """
        Compare generated vs ground truth using AST.
        Returns accuracy score and detailed comparison.
        """
        pass

# Multi-turn evaluation
class MultiTurnEvaluator:
    def evaluate(self,
                 conversation_history: List[Turn],
                 model_responses: List[str]) -> Score:
        """Evaluate multi-turn conversation function calls"""
        pass

# Agentic evaluation  
class AgenticEvaluator:
    def evaluate(self,
                 test_case: AgenticTestCase,
                 model_responses: List[str]) -> Score:
        """Evaluate agentic capabilities (memory, web search)"""
        pass
```

### 3. LLM Response Generation (`bfcl_eval/_llm_response_generation.py`)

**Purpose**: Orchestrate model inference at scale

**Key Flow:**

```python
def main(args):
    # 1. Initialize
    test_collection = load_test_data(args.test_category)  # Load test cases
    models = initialize_models(args.model)                 # Setup handlers
    
    # 2. Generate
    for model in models:
        for test_case in test_collection:
            # Prepare prompt
            prompt = build_prompt(test_case, model.supports_fc)
            
            # Call model
            response = model.inference(prompt)
            
            # Parse response
            parsed = model.parse_response(response)
            
            # Store result
            store_result(model.name, test_case.id, parsed)
    
    # 3. Output
    write_results_json()
    write_inference_logs()
```

### 4. Test Data System (`bfcl_eval/data/`)

**Purpose**: Store and manage test cases

```python
# Test case structure
class TestCase:
    id: str                          # e.g., "simple_python_1"
    test_category: str               # e.g., "simple_python"
    question: str                    # User query
    functions: List[FunctionDef]     # Available functions
    expected_calls: List[FunctionCall]  # Ground truth
    
    # Optional fields
    context: Optional[str]           # For multi-turn
    memory_state: Optional[Dict]     # For memory tests
    web_search_results: Optional[List]  # For web search
```

**Data Format (JSONL):**

```json
{
  "id": "simple_python_1",
  "test_category": "simple_python", 
  "question": "Get user with ID 123",
  "functions": [
    {
      "name": "get_user", 
      "description": "Get user by ID",
      "parameters": {
        "properties": {"user_id": {"type": "integer"}},
        "required": ["user_id"]
      }
    }
  ],
  "expected_calls": [
    {"name": "get_user", "arguments": {"user_id": 123}}
  ]
}
```

### 5. Constants & Configuration (`bfcl_eval/constants/`)

**Purpose**: Centralize configuration and metadata

```python
bfcl_eval/constants/
├── model_config.py                   # Model definitions
├── category_mapping.py               # Test category groups
├── eval_config.py                    # Output paths, config
└── prompt_templates.py               # System prompts
```

**Model Config Example:**

```python
MODEL_CONFIG_MAPPING = {
    "gpt-4o-2024-11-20-FC": {
        "provider": "openai",
        "model_name": "gpt-4o-2024-11-20",
        "handler": "OpenAIHandler",
        "supports_function_calling": True,
        "api_type": "chat_completion",
        "temperature": 0.0,
    },
    "meta-llama/Llama-3.1-70B-Instruct-FC": {
        "provider": "huggingface",
        "model_name": "meta-llama/Llama-3.1-70B-Instruct",
        "handler": "VLLMHandler",
        "supports_function_calling": True,
        "chat_template": "llama-3.1",
    },
    ...
}
```

---

## Generation Pipeline

### Phase 1: Preparation

```python
# Load test data
test_cases = load_test_collection(test_category)
# Returns: List[TestCase] with function definitions and expected calls

# Initialize models
models = [MODEL_REGISTRY[model_name] for model_name in model_names]
# Returns: List[ModelHandler]

# Setup logging
setup_inference_logging(include_input_log)
```

### Phase 2: Inference with Threading

```python
# For each test case, generate response
for test_case in test_cases:
    # Build prompt based on model capabilities
    if model.supports_function_calling:
        # Use tool/function definitions format
        prompt = build_fc_prompt(test_case, model)
    else:
        # Use text-based prompt format
        prompt = build_prompt_mode(test_case, model)
    
    # Inference (potentially parallelized)
    response = model.inference(prompt)
    
    # Parse response
    function_calls = model.parse_response(response)
    
    # Store
    results.append({
        "id": test_case.id,
        "question": test_case.question,
        "llm_response": response,
        "parsed_calls": function_calls,
        "execution_log": {...},
        "inference_log": {...}
    })
```

### Phase 3: Output Serialization

```python
# Write result JSON
result_file = f"result/{model_name}/BFCL_v3_{test_category}_result.json"
write_json(results, result_file)

# Write inference logs  
logs_file = f"result/{model_name}/inference_logs.json"
write_json(inference_logs, logs_file)
```

---

## Evaluation Pipeline

### Phase 1: Validation

```python
# Load generated results
results = load_json(f"result/{model_name}/BFCL_v3_{test_category}_result.json")

# Load ground truth test cases
test_cases = load_test_collection(test_category)

# Validate structure of parsed results
for result in results:
    validate_function_call_structure(result["parsed_calls"])
```

### Phase 2: AST-Based Comparison

```python
# For each test case:
for i, result in enumerate(results):
    test_case = test_cases[i]
    
    # Build AST for generated calls
    generated_ast = build_ast(result["parsed_calls"])
    
    # Build AST for ground truth
    ground_truth_ast = build_ast(test_case["expected_calls"])
    
    # Compare
    is_correct = compare_asts(generated_ast, ground_truth_ast)
    
    # Store score
    test_case_results.append({
        "id": test_case.id,
        "correct": is_correct,
        "score": 1.0 if is_correct else 0.0,
        "generated": result["parsed_calls"],
        "expected": test_case["expected_calls"]
    })
```

### Phase 3: Scoring Aggregation

```python
# Calculate category accuracy
category_accuracy = sum(r["score"] for r in test_case_results) / len(test_case_results)

# Store score file
score_data = {
    "model_name": model_name,
    "test_category": test_category,
    "accuracy": category_accuracy,
    "test_case_results": test_case_results,
    "statistics": {
        "total_tests": len(test_case_results),
        "correct": sum(1 for r in test_case_results if r["correct"]),
        "accuracy": category_accuracy
    }
}
write_json(score_data, f"score/{model_name}/BFCL_v3_{test_category}_score.json")
```

### Phase 4: CSV Generation

```python
# Aggregate all scores
all_scores = {}
for model_name in models:
    scores = {}
    for category in test_categories:
        score_file = f"score/{model_name}/BFCL_v3_{category}_score.json"
        data = load_json(score_file)
        scores[category] = data["accuracy"]
    all_scores[model_name] = scores

# Generate CSV with models as rows, categories as columns
write_csv_heatmap(all_scores, "score/data_overall.csv")
write_csv_heatmap(filter_live(all_scores), "score/data_live.csv")
write_csv_heatmap(filter_non_live(all_scores), "score/data_non_live.csv")
write_csv_heatmap(filter_multi_turn(all_scores), "score/data_multi_turn.csv")
```

---

## AST Evaluation Details

### Function Call AST Structure

```python
class FunctionCallNode:
    """Represents a function call in AST form"""
    
    function_name: str           # e.g., "get_user"
    arguments: Dict[str, Any]    # {"user_id": 123}
    
    def __eq__(self, other):
        """
        Semantic equality check:
        1. Function names must match exactly
        2. Arguments must match by key and value
        3. Type compatibility checked
        """
        return (self.function_name == other.function_name and 
                self.arguments == other.arguments)

class FunctionCallSequence:
    """Represents a sequence of function calls"""
    
    calls: List[FunctionCallNode]
    call_type: str               # "single", "parallel", "sequential"
    
    def __eq__(self, other):
        """
        Semantic equality for sequences.
        For parallel: order doesn't matter.
        For sequential: order matters.
        """
        pass
```

### Comparison Logic

```python
def compare_function_calls(generated, expected):
    """
    Compare generated vs expected function calls semantically.
    
    Returns: (is_correct: bool, details: Dict)
    """
    
    # Check function names
    if generated.function_name != expected.function_name:
        return False, {"error": "function_name_mismatch"}
    
    # Check parameter count
    if len(generated.arguments) != len(expected.arguments):
        return False, {"error": "parameter_count_mismatch"}
    
    # Check each parameter
    for param_name, expected_value in expected.arguments.items():
        if param_name not in generated.arguments:
            return False, {"error": f"missing_parameter_{param_name}"}
        
        generated_value = generated.arguments[param_name]
        
        # Type-aware comparison
        if not values_match(generated_value, expected_value):
            return False, {
                "error": "value_mismatch",
                "param": param_name,
                "expected": expected_value,
                "got": generated_value
            }
    
    return True, {"success": True}
```

---

## Multi-Turn Evaluation

### Conversation State Management

```python
class ConversationState:
    """Maintains state across multiple turns"""
    
    history: List[Turn]              # Previous turns
    function_execution_results: Dict  # Results from function calls
    memory_state: Optional[Dict]      # For memory tests
    
    def execute_function_call(self, call: FunctionCall) -> Any:
        """Execute function and update state"""
        func = self.get_function(call.name)
        result = func(**call.arguments)
        self.function_execution_results[call.id] = result
        return result
        
    def get_context_for_next_turn(self) -> str:
        """Build prompt context from history"""
        pass
```

### Multi-Turn Evaluation Flow

```python
for turn in test_case.conversation:
    # Build context from previous turns
    context = conversation_state.get_context_for_next_turn()
    
    # Get model response
    response = model.inference(turn.user_input + context)
    
    # Parse function calls
    calls = parser.parse(response)
    
    # Execute functions
    results = []
    for call in calls:
        result = conversation_state.execute_function_call(call)
        results.append(result)
    
    # Update state
    conversation_state.add_turn(Turn(
        user_input=turn.user_input,
        model_response=response,
        function_calls=calls,
        execution_results=results
    ))
    
    # Evaluate this turn
    is_correct = evaluate_turn(calls, turn.expected_calls)
    turn_scores.append(is_correct)

# Overall score: mean of all turns
overall_accuracy = mean(turn_scores)
```

---

## Agentic Evaluation

### Memory Backend Implementation

```python
class MemoryBackend:
    """Abstract memory interface"""
    
    def read(self, key: str) -> Any:
        """Read from memory"""
        pass
    
    def write(self, key: str, value: Any) -> None:
        """Write to memory"""
        pass

class KVMemoryBackend(MemoryBackend):
    """Key-value memory store"""
    store: Dict[str, Any] = {}

class VectorMemoryBackend(MemoryBackend):
    """Vector database (e.g., Faiss)"""
    
    def read(self, query: str, top_k: int = 1) -> List[Any]:
        """Semantic search in vectors"""
        pass

class RecursiveSummarizationMemory(MemoryBackend):
    """Hierarchical summarization memory"""
    
    def write(self, data: str) -> None:
        """Summarize and store hierarchically"""
        pass
```

### Web Search Evaluation

```python
def evaluate_web_search(test_case, model_response):
    """
    Evaluate web search function calls.
    
    1. Extract search query from function call
    2. Execute real web search (SerpAPI)
    3. Verify model correctly processes results
    """
    
    # Extract search query
    calls = parser.parse(model_response)
    search_call = [c for c in calls if c.name == "search"][0]
    query = search_call.arguments["query"]
    
    # Execute search
    search_results = execute_web_search(query)
    
    # Verify model understood results
    # Check if subsequent function calls are relevant
    is_correct = evaluate_subsequent_calls(calls[1:], search_results)
    
    return is_correct
```

---

## Performance Considerations

### Optimization Strategies

1. **Parallel Inference**
   - Multiple threads for API calls (respecting rate limits)
   - Batch inference for local models
   - Async requests for better throughput

2. **Model Routing**
   - Separate handlers for different backends
   - Connection pooling for API models
   - GPU memory management for local models

3. **Caching**
   - Cache model responses to avoid re-inference
   - Cache parsed results
   - Reuse test data across runs

4. **Streaming**
   - Stream responses for large models
   - Incremental result writing
   - Progressive scoring

---

## Extensibility

### Adding a New Model

1. **Create Handler Class**
   ```python
   # custom_handler.py
   from bfcl_eval.model_handler.base_handler import ModelHandler
   
   class CustomModelHandler(ModelHandler):
       def inference(self, test_case, num_threads=1):
           # Implement inference logic
           pass
       
       def parse_response(self, response):
           # Parse model output
           pass
   ```

2. **Register Model**
   ```python
   # In constants/model_config.py
   MODEL_CONFIG_MAPPING["custom-model"] = {
       "handler": "CustomModelHandler",
       "provider": "custom",
       ...
   }
   ```

### Adding a New Evaluation Strategy

1. **Create Evaluator Class**
   ```python
   class CustomEvaluator(BaseEvaluator):
       def evaluate(self, test_case, model_responses):
           # Custom evaluation logic
           pass
   ```

2. **Integrate into Pipeline**
   ```python
   # In eval_runner.py
   if test_case.category == "custom":
       evaluator = CustomEvaluator()
       score = evaluator.evaluate(test_case, responses)
   ```

---

## Monitoring & Debugging

### Inference Logs Structure

```json
{
  "test_id": "simple_python_1",
  "model_name": "gpt-4o-2024-11-20-FC",
  "timestamp": "2024-03-28T10:30:45Z",
  "input": {
    "system_prompt": "...",
    "user_query": "...",
    "functions": [...]
  },
  "output": {
    "raw_response": "...",
    "parsed_calls": [...],
    "parsing_time_ms": 45
  },
  "execution": {
    "function_results": [...],
    "execution_time_ms": 230,
    "errors": []
  },
  "metadata": {
    "tokens_in": 450,
    "tokens_out": 120,
    "model_temperature": 0.0
  }
}
```

### Performance Metrics

```python
class EvaluationMetrics:
    total_tests: int
    inference_time_ms: float      # Total generation time
    evaluation_time_ms: float     # Total scoring time
    accuracy: float               # Overall accuracy
    accuracy_by_category: Dict    # Per-category breakdown
    failed_tests: List[str]       # IDs of failed tests
    errors: Dict                  # Error categorization
```

---

This architecture enables scalable, extensible evaluation of LLM function calling capabilities.
