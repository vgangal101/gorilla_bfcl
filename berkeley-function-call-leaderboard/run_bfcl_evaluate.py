"""
Standalone script to run BFCL evaluation on a cluster.

Usage:
  python run_bfcl_evaluate.py --model Qwen/Qwen3-14B --test-category all
  python run_bfcl_evaluate.py --model model1,model2 --test-category simple,live
  python run_bfcl_evaluate.py --model Qwen/Qwen3-14B --partial-eval
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
from bfcl_eval.eval_checker.eval_runner import main as evaluation_main
from bfcl_eval.constants.eval_config import DOTENV_PATH


def parse_comma_separated(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run BFCL evaluation")
    parser.add_argument("--model", type=parse_comma_separated, default=None,
                        help="Comma-separated list of model names to evaluate")
    parser.add_argument("--test-category", type=parse_comma_separated, default=["all"],
                        help="Comma-separated list of test categories")
    parser.add_argument("--result-dir", type=str, default=None,
                        help="Path to the model response folder (relative to project root)")
    parser.add_argument("--score-dir", type=str, default=None,
                        help="Path to the evaluation score folder (relative to project root)")
    parser.add_argument("--partial-eval", action="store_true",
                        help="Run evaluation on partial set of benchmark entries")

    args = parser.parse_args()

    load_dotenv(dotenv_path=DOTENV_PATH, verbose=True, override=True)
    evaluation_main(
        args.model,
        args.test_category,
        args.result_dir,
        args.score_dir,
        args.partial_eval,
    )


if __name__ == "__main__":
    main()
