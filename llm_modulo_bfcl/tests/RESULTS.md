# LLM-Modulo Critic Bank — Test Results

**125/125 pytest tests passed** across two test files — includes adversarial edge cases that found and fixed 2 real bugs.  
**41/41 plot scenarios matched expected output** — every critic fires correctly on both pass and fail inputs.

Run command:
```bash
py -3.11 -m pytest llm_modulo_bfcl/tests/test_critics.py llm_modulo_bfcl/tests/test_adversarial.py -v
py -3.11 llm_modulo_bfcl/tests/generate_plots.py
```

Plots saved to `llm_modulo_bfcl/tests/plots/`:
- `critic_test_outcomes.png` — pass/fail matrix per critic per scenario
- `critic_summary.png` — total pass vs. fail counts per critic
- `bfcl_categories.png` — BFCL category breakdown by group

---

## Bugs Found and Fixed

Adversarial tests (`test_adversarial.py`, 30 cases) probed real boundary conditions and surfaced 2 implementation bugs:

| Bug | File | Fix |
|---|---|---|
| `bool` silently accepted for `number` fields — `isinstance(True, (int, float))` is `True` in Python, so the existing bool guard only on `integer` was insufficient | `argument_schema.py` | Extended the bool check to cover `integer` and `number` types |
| `exclusiveMinimum` / `exclusiveMaximum` JSON Schema keywords silently ignored — a value exactly at the exclusive boundary incorrectly passed | `argument_value.py` | Added enforcement for both `exclusiveMinimum` and `exclusiveMaximum` |

---

## Test Coverage (125 tests across 2 files)

**`test_critics.py` — 95 tests (specification coverage)**

| Class | Tests | What was verified |
|---|---|---|
| `TestConfig` | 4 | Hard/soft critic classification, ParserCritic as hard |
| `TestFunctionValidityCritic` | 7 | Known fns pass; unknown names, empty schema set fail; suggestion includes valid names |
| `TestArgumentSchemaCritic` | 11 | Required args, optional args, `$var` type bypass; missing required, unknown arg, str-for-int, bool-for-int all fail |
| `TestArgumentValueCritic` | 12 | Enum, email format, ISO date, min/max bounds, regex pattern — both valid and invalid |
| `TestDependencyCritic` | 9 | Valid chains, inline interpolation, forward references, duplicate `output_var`, nested list refs |
| `TestExecutionCritic` | 7 | Handler success/fail, chaining with `$var`, exception capture, step index on second-step failure |
| `TestRedundancyCritic` | 7 | Adjacent and non-adjacent duplicates; same fn different args is OK |
| `TestProposalParser` | 12 | Valid JSON, markdown fences, `output_var`; invalid JSON, missing/wrong-type `steps`, missing `function` |
| `TestToolExecutor` | 8 | Execution, full `$var` resolution, inline interpolation, unresolved vars, list args, missing handlers |
| `TestMetaController` | 8 | Initial prompt content, aggregation sorting (hard before soft), counts, backprompt content |
| `TestBFCLAdapter` | 9 | All three query key formats, all three schema key formats, OpenAI tool wrapper normalization |

**`test_adversarial.py` — 30 tests (boundary/edge-case coverage)**

| Class | Tests | What was probed |
|---|---|---|
| `TestArgumentSchemaEdgeCases` | 6 | `float` for integer, `bool` for number, `None` for string, missing `properties`, null parameters |
| `TestArgumentValueEdgeCases` | 5 | `exclusiveMinimum`, integer enums, subdomain email, float min boundary |
| `TestDependencyEdgeCases` | 5 | `$var` in nested dicts, bare `$`, space after `$`, duplicate output_var with different names |
| `TestParserEdgeCases` | 6 | Empty/whitespace input, trailing garbage, empty `output_var`, `null` output_var, prose + fence |
| `TestExecutorEdgeCases` | 5 | Handler returns `None`, `$var` bound to dict, multiple inline vars, execution continues after error |
| `TestFunctionValidityEdgeCases` | 3 | Schema entry missing `name`, case sensitivity, empty string function name |

---

## BFCL Full Category Listing (23 total, 22 scoring)

| Group | Count | Categories | Scoring |
|---|---|---|---|
| Non-Live (single-turn) | 7 | `simple_python`, `simple_java`, `simple_javascript`, `multiple`, `parallel`, `parallel_multiple`, `irrelevance` | Yes |
| Live (single-turn) | 6 | `live_simple`, `live_multiple`, `live_parallel`, `live_parallel_multiple`, `live_irrelevance`, `live_relevance` | Yes |
| Multi-Turn | 4 | `multi_turn_base`, `multi_turn_miss_func`, `multi_turn_miss_param`, `multi_turn_long_context` | Yes |
| Memory (Agentic) | 3 | `memory_kv`, `memory_vector`, `memory_rec_sum` | Yes |
| Web Search (Agentic) | 2 | `web_search_base`, `web_search_no_snippet` | Yes (skipped — paid SerpAPI) |
| Non-Scoring | 1 | `format_sensitivity` | No |

**Default run coverage:** `single_turn,multi_turn,memory` = 20 of 22 scoring categories (web_search excluded).

---

## Critic Design Gaps by Category Group

The current 6 critics are generic and cover **single-turn categories** well. Domain-specific critics are needed for the remaining groups.

### Non-Live / Live (single-turn) — current critics sufficient
All 6 critics apply directly:
- `FunctionValidityCritic` — called function exists in schema
- `ArgumentSchemaCritic` — required args present, no unknown args, correct types
- `ArgumentValueCritic` — enum values, email/date formats, numeric bounds, regex patterns
- `DependencyCritic` — `$var` references resolve to prior `output_var` values
- `ExecutionCritic` — plan executes without errors end-to-end
- `RedundancyCritic` *(soft)* — no duplicate function calls

Note: Live categories hit real API endpoints. The `ExecutionCritic` needs real handlers or sandboxed mocks mirroring those APIs.

### Multi-Turn (4 categories) — new critics needed
Stateful conversation; plan validity spans multiple turns.

| Proposed Critic | Type | Role |
|---|---|---|
| `TurnOrderCritic` | Hard | Each turn's calls address the correct conversational context |
| `StateConsistencyCritic` | Hard | Function outputs are correctly threaded between turns |
| `MissFuncCritic` | Hard | Detects when a required function call is omitted (targets `multi_turn_miss_func`) |
| `MissParamCritic` | Hard | Detects when a required argument is left out (targets `multi_turn_miss_param`) |
| `LongContextCritic` | Hard | Plan remains coherent over extended conversation history (targets `multi_turn_long_context`) |

### Memory / Agentic (3 categories) — new critics needed

| Category | Proposed Critic | Role |
|---|---|---|
| `memory_kv` | `MemoryConsistencyCritic` | KV store correctness — set/get/overwrite semantics |
| `memory_vector` | `RecallRelevanceCritic` | Embedding recall precision — right document retrieved |
| `memory_rec_sum` | `SummarizationCoherenceCritic` | Summarization coherence across recursive turns |

### Web Search / Agentic (2 categories) — skipped for now
Requires paid SerpAPI key ($75+/mo; free tier of 100 searches is well below one full run).

| Proposed Critic | Role |
|---|---|
| `GroundingCritic` | Response is grounded in actual search results |
| `SnippetUseCritic` | Correct behavior with snippet present vs. absent (`web_search_no_snippet`) |

### Format Sensitivity (non-scoring) — optional
Tests model fragility to prompt phrasing variations. A `RobustnessCritic` could be added if format sensitivity becomes a scoring concern.

---

## Current Critic Bank Summary

| Critic | Type | Gate |
|---|---|---|
| `ParserCritic` | Hard | JSON parses and has valid `steps` structure |
| `FunctionValidityCritic` | Hard | All called functions exist in the schema |
| `ArgumentSchemaCritic` | Hard | Required args present, no unknowns, correct JSON-Schema types |
| `ArgumentValueCritic` | Hard | Enum values, format hints (email/date), numeric bounds, regex patterns |
| `DependencyCritic` | Hard | `$var` references resolve to prior `output_var`; no forward refs, no redefinitions |
| `ExecutionCritic` | Hard | Plan executes end-to-end without errors |
| `RedundancyCritic` | **Soft** | No duplicate (function, arguments) steps |
