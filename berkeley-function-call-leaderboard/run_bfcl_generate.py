"""
Standalone script to run BFCL generation on a cluster.

Usage:
  python run_bfcl_generate.py --model gorilla-openfunctions-v2 --test-category all
  python run_bfcl_generate.py --model model1,model2 --test-category simple,live --num-gpus 4
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
from bfcl_eval._llm_response_generation import main as generation_main
from bfcl_eval.constants.eval_config import DOTENV_PATH, RESULT_PATH


def parse_comma_separated(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run BFCL generation")
    parser.add_argument("--model", type=parse_comma_separated, default=["gorilla-openfunctions-v2"],
                        help="Comma-separated list of model names")
    parser.add_argument("--test-category", type=parse_comma_separated, default=["all"],
                        help="Comma-separated list of test categories")
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--include-input-log", action="store_true")
    parser.add_argument("--exclude-state-log", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--backend", type=str, default="sglang", choices=["sglang", "vllm"])
    parser.add_argument("--skip-server-setup", action="store_true")
    parser.add_argument("--local-model-path", type=str, default=None)
    parser.add_argument("--result-dir", type=str, default=str(RESULT_PATH))
    parser.add_argument("--allow-overwrite", "-o", action="store_true")
    parser.add_argument("--run-ids", action="store_true")
    parser.add_argument("--enable-lora", action="store_true")
    parser.add_argument("--max-lora-rank", type=int, default=None)
    parser.add_argument("--lora-modules", type=str, nargs="*", default=None)

    args = parser.parse_args()

    # Remap hyphenated names to underscored names expected by generation_main
    args.test_category = args.__dict__.pop("test_category")
    args.include_input_log = args.__dict__.pop("include_input_log")
    args.exclude_state_log = args.__dict__.pop("exclude_state_log")
    args.num_gpus = args.__dict__.pop("num_gpus")
    args.num_threads = args.__dict__.pop("num_threads")
    args.gpu_memory_utilization = args.__dict__.pop("gpu_memory_utilization")
    args.skip_server_setup = args.__dict__.pop("skip_server_setup")
    args.local_model_path = args.__dict__.pop("local_model_path")
    args.result_dir = args.__dict__.pop("result_dir")
    args.allow_overwrite = args.__dict__.pop("allow_overwrite")
    args.run_ids = args.__dict__.pop("run_ids")
    args.enable_lora = args.__dict__.pop("enable_lora")
    args.max_lora_rank = args.__dict__.pop("max_lora_rank")
    args.lora_modules = args.__dict__.pop("lora_modules")

    load_dotenv(dotenv_path=DOTENV_PATH, verbose=True, override=True)
    generation_main(args)


if __name__ == "__main__":
    main()
