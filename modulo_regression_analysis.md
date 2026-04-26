# LLM-Modulo Regression Analysis: simple_java

Diagnostic write-up of why the LLM-Modulo run regresses against the plain-agent
baseline on `simple_java` (and to a lesser extent `simple_javascript`), with a
proposed `LanguageASTCritic` to close the gap.

**Repos**

- Analysis workspace (this file, result data, plots, scoring CSVs):
  `/Users/vgangal/phd_research_workspace/llm_modulo_bfcl_results`
- BFCL / LLM-Modulo runner repo (where the critic should be implemented):
  `/Users/vgangal/phd_research_workspace/gorilla_bfcl`
  (specifically `berkeley-function-call-leaderboard/run_bfcl_llm_modulo.py`
  and the critic bank referenced from `configs/modulo_full.yaml`)

## 1. Headline

The LLM-Modulo loop, as currently configured, **regresses 20pp on Java**
relative to a plain-agent baseline that uses no critic loop at all. The
regression is driven by the modulo agent prompt forcing JSON output that gets
serialized back into a Python-style call string — which BFCL's Java AST
decoder cannot parse. The critic bank never sees this string and so never
catches it.

| Model | Baseline failures (n=100) | Modulo failures (n=100) | Regressions* | Recovered† |
|---|---|---|---|---|
| Qwen3-14B | 36 | 58 | **23** | 1 |
| Qwen3-32B | 33 | 53 | **20** | 0 |
| Total     | 69 | 111 | **43** | **1** |

*regression = baseline correct, modulo wrong  †recovered = baseline wrong, modulo correct

The loop loses 43 tests it didn't have to lose and only fixes 1.

## 2. Per-category accuracy delta vs baseline (modulo − baseline)

| Model | py | **java** | **js** | mult | par | par_m | l_simp | l_mult | l_par† | l_par_m† |
|---|---|---|---|---|---|---|---|---|---|---|
| 14B | -0.8 | **-22.0** | -8.0 | +0.5 | -1.5 | -3.0 | +1.9 | +0.8 | +0.0 | +0.0 |
| 32B | -2.2 | **-20.0** | -6.0 | -2.0 | -1.0 | -2.5 | +0.0 | -5.7 | +18.8 | +12.5 |

†tiny N (16 and 24 respectively — single test = 6.25/4.17pp, so the
positives on l_par/l_par_m are noise-floor.)

Plain-agent baseline numbers from
`bfcl_sweep_20260425_164820/berkeley-function-call-leaderboard/score/data_non_live.csv`.
Modulo numbers from `score_v2/`.

## 3. Regression mechanism

### 3.1 The modulo agent prompt

Logged at line 10840 of
`sol_gaudi/logs/bfcl_qwen3_14b_modulo_51776938.out`:

```
You are a function-calling planner. Produce a JSON plan that solves the query.

Output format (strict JSON, no prose, no markdown fences):
{
  "steps": [
    {"function": "<name>", "arguments": {...}, "output_var": "<optional>"}
  ]
}

Rules:
- Use ONLY functions defined in the schema below.
- Provide ALL required arguments with correct JSON types.
- To reuse a previous step's result, set its `output_var` and reference it
  as `$<output_var>` in a later argument.
- Return valid JSON only.
```

This same prompt fires for **every** test category — including Java and
JavaScript, where the resulting JSON gets stitched into a call-string that
doesn't match the target language.

### 3.2 Concrete output diffs (same test IDs)

```
simple_java_6:
  baseline: [SpreadsheetPresentation.refreshData(refreshMetadata="true",
                                                 append="true", keepState="true")]
  modulo:   [SpreadsheetPresentation.refreshData(refreshMetadata=True,
                                                 append=True,   keepState=True)]
                                                            ^ ^^^^ Python bool

simple_java_14:
  baseline: [FunGameBase.onFinish(layout=gameLayout, success="true")]
  modulo:   [FunGameBase.onFinish(layout='gameLayout', success=True)]
                                                              ^ ^^^^ Python bool

simple_java_69:
  baseline: [DurationImpl.alignSigns(buf=textBuffer, ...)]   # bare identifier
  modulo:   [DurationImpl.alignSigns(buf='$durations', start=2, end=5)]
                                          ^^^^^^^^^^^ "$" placeholder leaked
                                                       from prompt template

simple_java_1:
  baseline: [SQLCompletionAnalyzer.makeProposalsFromObject(
                object="Customers", useShortName="true",
                params="{'limit': '50', 'schemaFilter': 'public'}")]
  modulo:   [SQLCompletionAnalyzer.makeProposalsFromObject(
                object='Customers', useShortName=True,
                params={'limit': '50', 'schema_filter': 'public'})]
                                       ^^^^^^^^^^^^^^^   snake/camel slip
                                ^^^^   Python dict literal
                       ^^^^   Python bool
```

### 3.3 Triggers, ranked

Counted across the 43 baseline-correct → modulo-wrong cases on simple_java:

| Trigger | Cases | What happened |
|---|---|---|
| 1. Python bool literals | **22 / 43 (51%)** | `True` / `False` in the modulo output, absent in baseline. Java AST rejects (`type_error:simple` or `ast_decoder:decoder_failed`). |
| 2. Python dict / list literals | **10 / 43 (23%)** | `params={'k': 'v'}` instead of either a Java map literal or the gold's stringified form. Mostly `ast_decoder:decoder_failed`. |
| 3. `$variable` placeholder leak | several (visible in iter=5 failures: simple_java_28, 43, 51, 69) | Modulo agent uses the prompt's `$varname` reference syntax even when the test expects a bare identifier. |
| 4. Misc value-string mismatches | ~5 | snake_case vs camelCase (`schema_filter` vs `schemaFilter`), wrong literal form (`Element` vs `Element.class`). |

**31 of 43 (72%)** of the regressions are Python-syntax artifacts (1+2)
that the baseline never produced — these are entirely caused by the modulo
pipeline, not the agent's intrinsic Java ability.

### 3.4 Why JS regresses less than Java (−6 to −8pp)

JS's BFCL AST decoder is more permissive about Python-flavored syntax
(JS's `true`/`false` are close enough that a case-insensitive or lenient
decoder can limp through, and JS object literals look much more like
Python dict literals than Java map syntax does), so the same JSON-roundtrip
damages JS less than Java. Python tests barely regress (1–2pp) because the
Python AST decoder accepts what the modulo serializer emits.

## 4. Why the existing critic bank doesn't catch this

The current modulo critic bank is:
```
critics:
  - FunctionValidityCritic
  - ArgumentSchemaCritic
  - ArgumentValueCritic
  - DependencyCritic
  - RedundancyCritic
```
(from `berkeley-function-call-leaderboard/configs/modulo_full.yaml`)

These all operate on the **JSON plan**, not on the language-specific
call-string that BFCL eventually decodes. From their perspective,
`{"success": true}` in the JSON is perfectly valid — schema-checked,
value-checked, all green. The downstream serializer then writes
`success=True` into the call string, and Java parsing dies at evaluation
time. **The critic literally never sees the string that BFCL will fail to
parse.**

This also explains the high iter=1 acceptance rate on Java (~85% in the v1
run): the critic checks JSON, the JSON is fine, ship it. Iter≥2 cases
(simple_java_28, _43, _51, _69 ran iter=5 and were eventually rejected)
tend to be cases where the critic *did* object — but to schema-level
issues like missing required params or wrong types in the JSON, not to the
Python-vs-Java syntax problem.

## 5. Proposed fix: `LanguageASTCritic`

Add a new critic that runs **BFCL's own AST decoder** on the serialized
call string. If the decoder rejects the string, the critic rejects the
candidate and re-prompts the agent. Using BFCL's own decoder guarantees
that a critic-accepted call is also eval-decodable.

### 5.1 Critic skeleton

```python
# In the modulo critic bank, alongside FunctionValidityCritic etc.

class LanguageASTCritic:
    """Reject candidate calls that BFCL's own AST decoder can't parse
    in the test category's target language."""

    LANG_BY_CATEGORY = {
        "simple_python": "python",  "multiple": "python",
        "parallel": "python",       "parallel_multiple": "python",
        "live_simple": "python",    "live_multiple": "python",
        "live_parallel": "python",  "live_parallel_multiple": "python",
        "simple_java": "java",
        "simple_javascript": "javascript",
    }

    def critique(self, candidate_call_string, test_category, function_spec):
        lang = self.LANG_BY_CATEGORY.get(test_category)
        if lang is None or lang == "python":
            return None  # accept; Python literal_eval already works

        try:
            from bfcl_eval.model_handler.utils import ast_parse
            ast_parse(candidate_call_string, language=lang)
            return None  # accept
        except Exception as e:
            return self._format_rejection(lang, candidate_call_string, str(e))
```

### 5.2 Glue point — IMPORTANT

The critic must run on the **serialized call string** (what BFCL eventually
decodes), not the JSON plan the agent emits. So insert `LanguageASTCritic`
*after* the JSON→call-string serializer in
`/Users/vgangal/phd_research_workspace/gorilla_bfcl/berkeley-function-call-leaderboard/run_bfcl_llm_modulo.py`,
not in the JSON-validation phase where the other critics live. If you keep
it in the JSON phase, it has nothing to parse.

### 5.3 Re-prompt sent back to the agent on rejection

```
Your previous output failed to parse as valid {LANG} syntax.

Decoder error: {parser error message}

Your output (post-serialization):
{candidate_call_string}

Common pitfalls when the target is {LANG}:
- Booleans: write `true` / `false`, not `True` / `False`.
- Strings: use double quotes (Java) / either quote style (JavaScript).
- Map / object arguments: pass them in the form the function spec shows
  (e.g. a quoted JSON string, NOT a Python dict literal like {'k': 'v'}).
- Identifier references: write the bare identifier (e.g. `myVar`), not the
  template's `$varname` placeholder syntax.

Rewrite the JSON plan so that the resulting {LANG} call string parses.
The function name and arguments must remain the same; only adjust syntax.
```

### 5.4 Dry-run before wiring

Before integrating, feed all 43 simple_java regression IDs from
`result_modulo_apr242026_v2/Qwen_Qwen3-{14B,32B}/non_live/BFCL_v4_simple_java_result.json`
through `ast_parse(..., language="java")`. If ≥31 of them raise
(the Python-bool + Python-dict cases), the critic would have rejected exactly
those regressions, and the loop has a clean path to recover the
−20pp delta.

A reasonable success criterion for the integration:
- Critic rejects ≥31/43 of the current regression IDs (close to the syntax-only
  ceiling).
- Modulo-on-Java accuracy moves from 42–47% to ≥ baseline (64–67%).
- Aspirational: with the critic plus argument-value enforcement,
  modulo-on-Java exceeds baseline by 15–20pp.

## 6. Tradeoffs and methodological notes

- **Parser, not runtime.** No JVM/V8 needed. The failures here are syntactic, not behavioral.
- **Use BFCL's own decoder.** Reusing `bfcl_eval.model_handler.utils.ast_parse` means a critic-accepted call is guaranteed eval-decodable. Methodologically, state this explicitly in the writeup: "the critic runs BFCL's own Java AST decoder, so a critic-accepted call is guaranteed to parse at eval time." That's a feature (no decoder mismatch), not a leakage concern, as long as it's flagged.
- **Alternative: external parser.** `tree-sitter-java` + `tree-sitter-javascript` (both pip-installable) decouple critic from BFCL but introduce a small parser-vs-eval gap.
- **What the critic still won't catch.** A parser-only critic catches form (`True` vs `true`, `{...}` vs Java map syntax) but not value (`'$durations'` vs the bare identifier `durations`, or wrong enum strings). The residual ~12/43 Java regression cases need a follow-up `ArgumentValueCritic` that's language-aware (e.g., reject when an argument value matches the literal `"$<word>"` pattern but no prior step defines `output_var: <word>`).

## 7. Implication for a Java-focused writeup

Once `LanguageASTCritic` is in:
- modulo-on-Java should at minimum match baseline (~64–67%);
- with the residual `$varname` filter and a value-aware critic, you should
  exceed baseline.

The narrative becomes:
- baseline: 64–67%
- modulo with naive prompt: 42–47%   (current — regression)
- modulo with `LanguageASTCritic`: ≥ baseline    (target)
- modulo with full language-aware critic stack: ~85%+    (aspirational)

That's a credible LLM-Modulo result because the critic is *adding* a
verifier (parser-in-the-loop) that the baseline doesn't have, rather than
showcasing self-inflicted regressions.

## 8. Files referenced

Paths under the analysis workspace
`/Users/vgangal/phd_research_workspace/llm_modulo_bfcl_results`:

- Result data:
  - Modulo (LLM-Modulo runs): `result_modulo_apr242026_v2/`
  - Baseline (plain agent): `bfcl_sweep_20260425_164820/berkeley-function-call-leaderboard/result/`
- Score data:
  - Modulo: `score_v2/`
  - Baseline: `score_baseline/` (re-evaluated locally to get per-test failure JSONs)
- Modulo run config: `berkeley-function-call-leaderboard/configs/modulo_full.yaml`
- Modulo agent prompt (verbatim, line 10840):
  `sol_gaudi/logs/bfcl_qwen3_14b_modulo_51776938.out`
- Plots produced during this analysis:
  - `analysis_output/baseline_vs_modulo/accuracy_baseline_vs_modulo.png`
  - `analysis_output/baseline_vs_modulo/delta_modulo_minus_baseline.png`
  - `analysis_output/baseline_vs_modulo/critic_recovery.png`
  - `analysis_output/baseline_vs_modulo/baseline_full_picture.png`
  - `analysis_output/baseline_vs_modulo/summary.csv`

Paths under the BFCL / LLM-Modulo runner repo
`/Users/vgangal/phd_research_workspace/gorilla_bfcl`:

- Modulo runner entry point:
  `berkeley-function-call-leaderboard/run_bfcl_llm_modulo.py`
- BFCL AST decoder source (target for `LanguageASTCritic` to call):
  `berkeley-function-call-leaderboard/bfcl_eval/model_handler/utils.py:248`
  (function `ast_parse(input_str, language)`; routes to
  `parse_java_function_call` / `parse_javascript_function_call`)
- Plots produced during this analysis:
  - `analysis_output/baseline_vs_modulo/accuracy_baseline_vs_modulo.png`
  - `analysis_output/baseline_vs_modulo/delta_modulo_minus_baseline.png`
  - `analysis_output/baseline_vs_modulo/critic_recovery.png`
  - `analysis_output/baseline_vs_modulo/baseline_full_picture.png`
  - `analysis_output/baseline_vs_modulo/summary.csv`
