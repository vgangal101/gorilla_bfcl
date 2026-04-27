"""Run LLM-Modulo evaluation on single-turn BFCL categories.

Writes BFCL-compatible result files so the standard `bfcl evaluate` can
score them. Use a separate `--result-dir` to keep modulo runs isolated
from baseline `run_bfcl_generate.py` runs.

Two ways to drive it:

  # (a) Config file (YAML or JSON) — preferred for repeatable experiments.
  python run_bfcl_llm_modulo.py --config configs/modulo_example.yaml

  # (b) Pure CLI — handy for one-offs.
  python run_bfcl_llm_modulo.py \
      --model Qwen/Qwen3-14B \
      --test-category simple_python,multiple,parallel \
      --endpoint http://localhost:1053/v1 \
      --result-dir result_modulo

CLI flags always override config; config overrides built-in defaults.

Then evaluate as usual:
    bfcl evaluate --model <model> --test-category <cats> --result-dir <dir>

Scope (v1): single-turn AST categories only. Multi-turn, agentic, memory,
irrelevance/relevance, and format_sensitivity are deliberately rejected —
they need turn-state/execution/classification logic that the current
critic bank does not implement.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# Make the BFCL package importable when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Add the sibling llm_modulo_bfcl package to the path.
_MODULO_PKG = Path(__file__).resolve().parent.parent / "llm_modulo_bfcl"
if str(_MODULO_PKG) not in sys.path:
    sys.path.insert(0, str(_MODULO_PKG))

from dotenv import load_dotenv

# BFCL helpers
from bfcl_eval.constants.category_mapping import VERSION_PREFIX
from bfcl_eval.constants.eval_config import DOTENV_PATH, PROMPT_PATH

# llm_modulo_bfcl framework
from controller.meta_controller import MetaController  # noqa: E402
from critics.argument_schema import ArgumentSchemaCritic  # noqa: E402
from critics.argument_value import ArgumentValueCritic  # noqa: E402
from critics.dependency import DependencyCritic  # noqa: E402
from critics.function_validity import FunctionValidityCritic  # noqa: E402
from critics.language_ast import LanguageASTCritic  # noqa: E402
from critics.redundancy import RedundancyCritic  # noqa: E402
from llm_interface import LLMInterface  # noqa: E402
from loop import run_llm_modulo  # noqa: E402
from parser import ProposalParser  # noqa: E402
from render import plan_to_bfcl_ast  # noqa: E402
from schemas import Plan  # noqa: E402


# ----------------------------------------------------------------------
# Critic registry. ExecutionCritic is intentionally NOT registered here:
# BFCL tools aren't executable without a sandbox, and we don't want a
# config file to silently request something we can't honor.
# ----------------------------------------------------------------------

_CRITIC_REGISTRY: dict[str, type] = {
    "FunctionValidityCritic": FunctionValidityCritic,
    "ArgumentSchemaCritic": ArgumentSchemaCritic,
    "ArgumentValueCritic": ArgumentValueCritic,
    "DependencyCritic": DependencyCritic,
    "LanguageASTCritic": LanguageASTCritic,
    "RedundancyCritic": RedundancyCritic,  # soft
}

_DEFAULT_CRITICS: list[str] = [
    "FunctionValidityCritic",
    "ArgumentSchemaCritic",
    "ArgumentValueCritic",
    "DependencyCritic",
    "LanguageASTCritic",
    "RedundancyCritic",
]


# ----------------------------------------------------------------------
# Run-config defaults. Everything the script needs to execute is keyed
# here; config files and CLI flags both feed into this same shape.
# ----------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {
    "model": None,
    "categories": None,
    "endpoint": None,
    "api_key": None,
    "temperature": 0.0,
    "max_iters": 5,
    "critics": _DEFAULT_CRITICS,
    # When None, `result_dir` is auto-derived from the experiment config
    # (see _derive_result_dir). Set explicitly in the config or via
    # --result-dir to pin it.
    "result_dir": None,
    "experiment_tag": None,
    "num_samples": None,
    "allow_overwrite": False,
    # Server lifecycle. When `endpoint` isn't set, the runner spins up a
    # local vLLM/sglang server via BFCL's OSSHandler.spin_up_local_server
    # and tears it down at the end. Mirrors the `bfcl generate` flags.
    "backend": "vllm",
    "num_gpus": 1,
    "gpu_memory_utilization": 0.9,
    "skip_server_setup": False,
    "local_model_path": None,
}


def _slug(s: str) -> str:
    """Sanitize an arbitrary string for use as a filesystem path segment."""
    import re
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


def _derive_result_dir(cfg: dict) -> str:
    """Build a descriptive default directory name that encodes the run config.

    Shape: results_llm_modulo_<model>_iters<N>_temp<T>_critics<K>[_<tag>]

    Two runs that differ in any of model / max_iters / temperature /
    critic-bank size land in different directories, so a sweep doesn't
    silently overwrite itself. Categories aren't in the name because
    different categories always write different filenames within the
    same result dir.
    """
    parts = ["results_llm_modulo"]
    parts.append(_slug((cfg.get("model") or "unknown").replace("/", "_")))
    parts.append(f"iters{cfg['max_iters']}")
    # `:g` strips trailing zeros, keeps 0.7 as "0.7" and 0.0 as "0".
    parts.append(f"temp{cfg['temperature']:g}")
    parts.append(f"critics{len(cfg['critics'])}")
    tag = cfg.get("experiment_tag")
    if tag:
        parts.append(_slug(str(tag)))
    return "_".join(parts)


# ----------------------------------------------------------------------
# Category gating. Keep this conservative — add categories only once the
# critic bank actually covers their failure modes.
# ----------------------------------------------------------------------

ALLOWED_CATEGORIES: set[str] = {
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "parallel",
    "parallel_multiple",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
}


# ----------------------------------------------------------------------
# BFCL -> JSON-Schema-ish normalization (same shape as the validator uses)
# ----------------------------------------------------------------------

_BFCL_TYPE_REWRITE = {"dict": "object", "float": "number", "tuple": "array", "any": None}


def _normalize_type(t: Any) -> Any:
    if not isinstance(t, str):
        return t
    return _BFCL_TYPE_REWRITE.get(t, t)


def _normalize_bfcl_schema(schema: dict) -> dict:
    s = copy.deepcopy(schema)
    params = s.get("parameters") or {}
    params["type"] = _normalize_type(params.get("type")) or "object"
    props = params.get("properties", {}) or {}
    for name, spec in list(props.items()):
        if not isinstance(spec, dict):
            continue
        nt = _normalize_type(spec.get("type"))
        if nt is None:
            spec.pop("type", None)
        else:
            spec["type"] = nt
        items = spec.get("items")
        if isinstance(items, dict) and "type" in items:
            nt = _normalize_type(items.get("type"))
            if nt is None:
                items.pop("type", None)
            else:
                items["type"] = nt
    s["parameters"] = params
    return s


# ----------------------------------------------------------------------
# Plan -> BFCL AST string. Rendering is delegated to llm_modulo_bfcl/render.py
# so the safety-net LanguageASTCritic and the runner emit byte-identical
# strings. Language is derived from the test category — Python categories
# get Python `repr()` (existing behavior); Java/JavaScript get language-
# specific bool/string/dict rendering so their AST decoders accept the
# output instead of choking on Python literals like `True` and `{'k':'v'}`.
# ----------------------------------------------------------------------

_CATEGORY_LANGUAGE: dict[str, str] = {
    "simple": "python",
    "simple_python": "python",
    "simple_java": "java",
    "simple_javascript": "javascript",
    "multiple": "python",
    "parallel": "python",
    "parallel_multiple": "python",
    "live_simple": "python",
    "live_multiple": "python",
    "live_parallel": "python",
    "live_parallel_multiple": "python",
}


def _category_language(category: str) -> str:
    return _CATEGORY_LANGUAGE.get(category, "python")


# ----------------------------------------------------------------------
# OpenAI-compatible client. Works against vLLM / sglang / OpenAI proper.
# ----------------------------------------------------------------------

class OpenAICompatLLM(LLMInterface):
    def __init__(self, model: str, base_url: str, api_key: str | None, temperature: float = 0.0):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "run_bfcl_llm_modulo.py requires the `openai` package. "
                "Install it with `pip install openai`."
            ) from e

        self.client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY", timeout=1800.0)
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------

def _load_test_entries(category: str) -> list[dict]:
    path = None
    for candidate in PROMPT_PATH.rglob(f"{VERSION_PREFIX}_{category}.json"):
        path = candidate
        break
    if path is None:
        raise FileNotFoundError(
            f"No BFCL test file found for category '{category}' under {PROMPT_PATH}."
        )
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _query_from_entry(entry: dict) -> str:
    q = entry.get("question")
    if isinstance(q, list) and q and isinstance(q[0], list):
        turns = q[0]
        return "\n".join(str(t.get("content", "")) for t in turns if isinstance(t, dict))
    if isinstance(q, list) and q and isinstance(q[0], dict):
        return str(q[0].get("content", ""))
    if isinstance(q, str):
        return q
    raise ValueError(f"Entry {entry.get('id')!r} has no parseable 'question' field.")


def _category_to_group(category: str) -> str:
    # Matches BFCL's directory grouping for single-turn results.
    return "live" if category.startswith("live_") else "non_live"


# ----------------------------------------------------------------------
# Per-category runner
# ----------------------------------------------------------------------

def _build_critics(names: list[str] | None = None) -> list:
    """Instantiate critics by registry name. Unknown names raise cleanly."""
    names = names if names is not None else _DEFAULT_CRITICS
    out = []
    for n in names:
        cls = _CRITIC_REGISTRY.get(n)
        if cls is None:
            raise ValueError(
                f"Unknown critic '{n}'. Available: {sorted(_CRITIC_REGISTRY)}"
            )
        out.append(cls())
    return out


# ----------------------------------------------------------------------
# Config file loading + merge
# ----------------------------------------------------------------------

def _load_config_file(path: Path) -> dict:
    """Load YAML or JSON config and flatten nested sections to a flat dict."""
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "YAML configs require PyYAML. Install it or use a .json config."
            ) from e
        with path.open() as f:
            data = yaml.safe_load(f) or {}
    elif suffix == ".json":
        with path.open() as f:
            data = json.load(f)
    else:
        raise ValueError(
            f"Unsupported config extension: {suffix!r}. Use .yaml, .yml, or .json."
        )

    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}.")

    flat: dict[str, Any] = {}
    # Top-level flat keys win.
    for k in _DEFAULTS:
        if k in data:
            flat[k] = data[k]
    # Also accept the nested layout used in the example config.
    modulo = data.get("modulo") or {}
    if isinstance(modulo, dict):
        for k in ("max_iters", "critics", "temperature"):
            if k in modulo:
                flat[k] = modulo[k]
    output = data.get("output") or {}
    if isinstance(output, dict):
        for k in ("result_dir", "num_samples", "allow_overwrite"):
            if k in output:
                flat[k] = output[k]
    server = data.get("server") or {}
    if isinstance(server, dict):
        for k in ("backend", "num_gpus", "gpu_memory_utilization",
                 "skip_server_setup", "local_model_path", "endpoint", "api_key"):
            if k in server:
                flat[k] = server[k]
    # Accept the singular "test_category" alias for "categories".
    if "test_category" in data and "categories" not in flat:
        flat["categories"] = data["test_category"]

    unknown = set(flat) - set(_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return flat


def _resolve_run_config(config: dict, cli_overrides: dict) -> dict:
    """Defaults <- config <- CLI. CLI keys set to None are treated as unset."""
    merged = dict(_DEFAULTS)
    merged.update(config)
    for k, v in cli_overrides.items():
        if v is not None:
            merged[k] = v
    return merged


# ----------------------------------------------------------------------
# Server lifecycle
# ----------------------------------------------------------------------

def _spin_up_server_if_needed(cfg: dict):
    """Return (handler, endpoint).

    If `cfg['endpoint']` is explicitly set, no handler is built and the
    endpoint is used as-is. Otherwise we build BFCL's handler for `cfg['model']`
    and call `spin_up_local_server`, which either spawns a vLLM/sglang process
    (when `skip_server_setup` is False) or just waits for a server that's
    already running. Either way, `handler.base_url` is the endpoint the
    LLM-Modulo loop should talk to.

    Caller is responsible for `handler.shutdown_local_server()` if handler
    is not None.
    """
    if cfg.get("endpoint"):
        print(f"[modulo] using explicit endpoint: {cfg['endpoint']}")
        return None, cfg["endpoint"]

    # Lazy imports so --help and pure-config invocations don't pay the
    # torch/transformers cost.
    from bfcl_eval._llm_response_generation import build_handler
    from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler

    handler = build_handler(cfg["model"], cfg["temperature"])
    if not isinstance(handler, OSSHandler):
        raise RuntimeError(
            f"Model '{cfg['model']}' is not an OSS/local model — the built-in "
            f"server spin-up only works for models whose BFCL handler "
            f"subclasses OSSHandler. Pass `--endpoint <url>` to point at a "
            f"remote OpenAI-compatible server instead."
        )

    mode = "connecting to existing server" if cfg["skip_server_setup"] else \
           f"spinning up {cfg['backend']} server"
    print(f"[modulo] {mode} for {cfg['model']} "
          f"(gpus={cfg['num_gpus']}, util={cfg['gpu_memory_utilization']})")

    handler.spin_up_local_server(
        num_gpus=cfg["num_gpus"],
        gpu_memory_utilization=cfg["gpu_memory_utilization"],
        backend=cfg["backend"],
        skip_server_setup=cfg["skip_server_setup"],
        local_model_path=cfg["local_model_path"],
    )
    print(f"[modulo] server ready at {handler.base_url}")
    return handler, handler.base_url


def _tear_down_server(handler) -> None:
    if handler is None:
        return
    try:
        handler.shutdown_local_server()
    except Exception as e:  # noqa: BLE001 — always continue, we're exiting
        print(f"[modulo] WARNING: server shutdown raised: {e}")


def run_category(
    *,
    model: str,
    category: str,
    endpoint: str,
    api_key: str | None,
    max_iters: int,
    num_samples: int | None,
    result_dir: Path,
    temperature: float,
    allow_overwrite: bool,
    critic_names: list[str] | None = None,
) -> dict:
    if category not in ALLOWED_CATEGORIES:
        raise ValueError(
            f"Category '{category}' is not supported by run_bfcl_llm_modulo v1. "
            f"Allowed: {sorted(ALLOWED_CATEGORIES)}"
        )

    entries = _load_test_entries(category)
    if num_samples is not None:
        entries = entries[:num_samples]

    language = _category_language(category)

    model_dir = result_dir / model.replace("/", "_")
    group_dir = model_dir / _category_to_group(category)
    group_dir.mkdir(parents=True, exist_ok=True)
    out_path = group_dir / f"{VERSION_PREFIX}_{category}_result.json"

    if out_path.exists() and not allow_overwrite:
        raise FileExistsError(
            f"{out_path} already exists. Re-run with --allow-overwrite (-o) to replace."
        )

    llm = OpenAICompatLLM(
        model=model, base_url=endpoint, api_key=api_key, temperature=temperature
    )

    meta = MetaController()
    parser_ = ProposalParser()

    stats = {"total": 0, "success": 0, "failure": 0, "iters_sum": 0, "errors": 0}
    t0 = time.time()

    with out_path.open("w") as f:
        for entry in entries:
            entry_id = entry.get("id", "?")
            try:
                query = _query_from_entry(entry)
                schemas = [
                    _normalize_bfcl_schema(s)
                    for s in entry.get("function", []) or []
                ]
                result = run_llm_modulo(
                    query=query,
                    function_schemas=schemas,
                    llm=llm,
                    critics=_build_critics(critic_names),
                    meta_controller=meta,
                    parser=parser_,
                    max_iters=max_iters,
                    extra_context={"language": language, "category": category},
                )
            except Exception as e:
                stats["total"] += 1
                stats["errors"] += 1
                f.write(json.dumps({
                    "id": entry_id,
                    "result": "[]",
                    "llm_modulo": {
                        "status": "error",
                        "iterations": 0,
                        "error": f"{type(e).__name__}: {e}",
                    },
                }) + "\n")
                traceback.print_exc()
                continue

            stats["total"] += 1
            stats["iters_sum"] += result["iterations"]

            if result["status"] == "success" and result["plan"] is not None:
                stats["success"] += 1
                ast = plan_to_bfcl_ast(result["plan"], language)
            else:
                stats["failure"] += 1
                # Write the last proposal's plan (if parseable) so the
                # evaluator still counts the entry; otherwise an empty list.
                ast = (
                    plan_to_bfcl_ast(result["plan"], language)
                    if result["plan"] is not None
                    else "[]"
                )

            f.write(json.dumps({
                "id": entry_id,
                "result": ast,
                "llm_modulo": {
                    "status": result["status"],
                    "iterations": result["iterations"],
                },
            }) + "\n")

    elapsed = time.time() - t0
    avg_iters = stats["iters_sum"] / max(stats["total"], 1)
    print(
        f"[modulo] {model}/{category}: accepted={stats['success']}/{stats['total']} "
        f"rejected={stats['failure']} errors={stats['errors']} "
        f"avg_iters={avg_iters:.2f} time={elapsed:.1f}s -> {out_path}"
    )
    return {"category": category, "out_path": str(out_path), "elapsed_sec": elapsed, **stats}


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _parse_csv(value: str) -> list[str]:
    return [c.strip() for c in value.split(",") if c.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--config", default=None,
        help="Path to a YAML or JSON config file. CLI flags override config values.",
    )
    # All overrides use `None` sentinels so we can tell whether the user
    # explicitly set them vs. accepted the config/default.
    ap.add_argument("--model", default=None, help="Model id passed to the server.")
    ap.add_argument(
        "--test-category", default=None, type=_parse_csv,
        help="Comma-separated single-turn categories (e.g. simple,multiple,live_simple).",
    )
    ap.add_argument(
        "--endpoint", default=None,
        help="OpenAI-compatible base URL. If set, the runner skips server "
             "spin-up and talks directly to this endpoint. If unset, the "
             "runner spins up a local vLLM/sglang server for --model.",
    )
    ap.add_argument("--api-key", default=None, help="API key (or $OPENAI_API_KEY).")
    ap.add_argument(
        "--backend", default=None, choices=["vllm", "sglang"],
        help="Local server backend when spinning up (default: vllm).",
    )
    ap.add_argument("--num-gpus", type=int, default=None,
                    help="Tensor-parallel size for the local server (default: 1).")
    ap.add_argument("--gpu-memory-utilization", type=float, default=None,
                    help="vLLM/sglang memory fraction (default: 0.9).")
    ap.add_argument(
        "--skip-server-setup", action="store_true", default=None,
        help="Don't spawn a server; expect one already running at "
             "$LOCAL_SERVER_ENDPOINT:$LOCAL_SERVER_PORT.",
    )
    ap.add_argument("--local-model-path", default=None,
                    help="Optional path to a local model snapshot to serve.")
    ap.add_argument("--max-iters", type=int, default=None, help="Critic-loop iteration budget.")
    ap.add_argument(
        "--num-samples", type=int, default=None,
        help="Cap samples per category (handy for smoke-testing).",
    )
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument(
        "--result-dir", default=None,
        help="Where to write result files. If unset, auto-derives a name "
             "like `results_llm_modulo_<model>_iters<N>_temp<T>_critics<K>` "
             "from the resolved run config so different sweeps don't collide.",
    )
    ap.add_argument(
        "--experiment-tag", default=None,
        help="Short label (e.g. 'baseline', 'diversified') appended to the "
             "auto-derived result_dir. Ignored if --result-dir is set.",
    )
    ap.add_argument(
        "--allow-overwrite", "-o", action="store_true", default=None,
    )
    args = ap.parse_args()

    load_dotenv(dotenv_path=DOTENV_PATH, verbose=False, override=False)

    # Load config file (if any).
    file_config: dict = {}
    if args.config:
        file_config = _load_config_file(Path(args.config))

    # Build CLI overrides (only keys the user actually passed).
    cli_overrides = {
        "model": args.model,
        "categories": args.test_category,
        "endpoint": args.endpoint,
        "api_key": args.api_key,
        "temperature": args.temperature,
        "max_iters": args.max_iters,
        "result_dir": args.result_dir,
        "experiment_tag": args.experiment_tag,
        "num_samples": args.num_samples,
        "allow_overwrite": args.allow_overwrite,
        "backend": args.backend,
        "num_gpus": args.num_gpus,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "skip_server_setup": args.skip_server_setup,
        "local_model_path": args.local_model_path,
    }

    cfg = _resolve_run_config(file_config, cli_overrides)

    # Auto-derive result_dir if the user didn't pin one.
    if not cfg["result_dir"]:
        cfg["result_dir"] = _derive_result_dir(cfg)

    # Required keys.
    missing = [k for k in ("model", "categories") if not cfg.get(k)]
    if missing:
        print(
            f"ERROR: required setting(s) {missing} not provided. "
            f"Set them via --{missing[0].replace('_','-')} or in --config.",
            file=sys.stderr,
        )
        return 2

    api_key = cfg["api_key"] or os.environ.get("OPENAI_API_KEY")

    # Validate critic names up front.
    try:
        _build_critics(cfg["critics"])
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    # Category gating.
    unknown = [c for c in cfg["categories"] if c not in ALLOWED_CATEGORIES]
    if unknown:
        print(
            f"ERROR: categories {unknown} are not supported by LLM-Modulo v1.\n"
            f"Allowed: {sorted(ALLOWED_CATEGORIES)}",
            file=sys.stderr,
        )
        return 2

    # Echo the resolved config so experiments are reproducible from logs.
    print("[modulo] resolved run config:")
    for k in ("model", "categories", "endpoint", "temperature", "max_iters",
              "critics", "experiment_tag", "result_dir",
              "num_samples", "allow_overwrite",
              "backend", "num_gpus", "gpu_memory_utilization",
              "skip_server_setup", "local_model_path"):
        print(f"  {k}: {cfg[k]}")
    print()

    handler = None
    try:
        handler, endpoint = _spin_up_server_if_needed(cfg)

        per_cat: list[dict] = []
        for cat in cfg["categories"]:
            per_cat.append(run_category(
                model=cfg["model"],
                category=cat,
                endpoint=endpoint,
                api_key=api_key,
                max_iters=cfg["max_iters"],
                num_samples=cfg["num_samples"],
                result_dir=Path(cfg["result_dir"]),
                temperature=cfg["temperature"],
                allow_overwrite=bool(cfg["allow_overwrite"]),
                critic_names=cfg["critics"],
            ))
    finally:
        _tear_down_server(handler)

    total = sum(c["total"] for c in per_cat)
    success = sum(c["success"] for c in per_cat)
    errors = sum(c["errors"] for c in per_cat)
    print(
        f"\n[modulo] overall: accepted={success}/{total} "
        f"errors={errors} across {len(per_cat)} categories. "
        f"Next step: `bfcl evaluate --model {cfg['model']} "
        f"--test-category {','.join(cfg['categories'])} "
        f"--result-dir {cfg['result_dir']}`"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
