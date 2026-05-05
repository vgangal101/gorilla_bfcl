"""
hf_job.py — Submit SFT + GRPO as a HuggingFace Job.

Runs the full pipeline (data prep → SFT → GRPO → push checkpoint to Hub)
on a cloud A100 without needing local GPU or Gaudi setup.

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login   (or set HF_TOKEN env var)

Usage:
    python grpo_pipeline/hf_job.py

The job script below is self-contained (PEP 723 inline deps) so HF Jobs
can run it with `uv run` without a separate requirements file.
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "transformers>=4.45",
#   "trl>=0.12",
#   "datasets>=2.20",
#   "peft>=0.12",
#   "accelerate>=0.34",
#   "huggingface_hub>=0.24",
# ]
# ///

import os
import subprocess
import sys
import textwrap
from pathlib import Path

HF_REPO_ID = "SaiGanesh314/bfcl-grpo-qwen3-8b"   # change to your HF username
HF_TOKEN = os.environ.get("HF_TOKEN", "")


INLINE_JOB_SCRIPT = textwrap.dedent("""\
    # /// script
    # requires-python = ">=3.10"
    # dependencies = [
    #   "transformers>=4.45",
    #   "trl>=0.12",
    #   "datasets>=2.20",
    #   "peft>=0.12",
    #   "accelerate>=0.34",
    #   "huggingface_hub>=0.24",
    # ]
    # ///

    import subprocess, sys, os
    from pathlib import Path
    from huggingface_hub import snapshot_download, HfApi

    # ------------------------------------------------------------------
    # 1. Pull repo code onto the job container
    # ------------------------------------------------------------------
    subprocess.run([
        "git", "clone", "--branch", "bfcl_rl", "--depth", "1",
        "https://huggingface.co/datasets/SaiGanesh314/gorilla-bfcl-code",  # mirror your repo here
        "/workspace/gorilla_bfcl"
    ], check=True)

    sys.path.insert(0, "/workspace/gorilla_bfcl/berkeley-function-call-leaderboard")
    sys.path.insert(0, "/workspace/gorilla_bfcl/grpo_pipeline")
    os.chdir("/workspace/gorilla_bfcl")

    # ------------------------------------------------------------------
    # 2. SFT
    # ------------------------------------------------------------------
    from sft_train import main as run_sft
    run_sft()

    # ------------------------------------------------------------------
    # 3. GRPO
    # ------------------------------------------------------------------
    from grpo_train import main as run_grpo
    run_grpo()

    # ------------------------------------------------------------------
    # 4. Push final checkpoint to Hub
    # ------------------------------------------------------------------
    api = HfApi()
    api.upload_folder(
        folder_path="checkpoints/grpo_final",
        repo_id="{repo_id}",
        repo_type="model",
    )
    print("Done — checkpoint pushed to Hub.")
""").format(repo_id=HF_REPO_ID)


def submit():
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub")
        sys.exit(1)

    job_script_path = Path("/tmp/bfcl_grpo_job.py")
    job_script_path.write_text(INLINE_JOB_SCRIPT)

    api = HfApi(token=HF_TOKEN)
    print("Submitting HuggingFace Job...")

    # huggingface_hub >= 0.24 has jobs API
    # Hardware options: "nvidia-a100-x1" | "nvidia-a100-x2" | "nvidia-a100-x4"
    result = api.create_job(
        script=str(job_script_path),
        hardware="nvidia-a100-x1",          # single A100-80G; enough for Qwen3-8B LoRA
        timeout=86400,                       # 24h max
    )
    print(f"Job submitted: {result.job_id}")
    print(f"Status URL: https://huggingface.co/jobs/{result.job_id}")
    return result.job_id


if __name__ == "__main__":
    submit()
