"""
smoke_grpo.py — Minimal GRPO smoke test.

Mirrors grpo_train.py exactly but uses 4 examples, K=2 rollouts,
and 2 training steps. Confirms the GRPO code path works end-to-end.

Requires smoke_sft.py to have run first (needs the SFT checkpoint).

Run:
    python grpo_pipeline/smoke_tests/smoke_grpo.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "berkeley-function-call-leaderboard"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

import json as _json

from data_prep import build_dataset
from reward import bfcl_reward_fn

MODEL_KEY      = os.environ.get("BFCL_MODEL_KEY",      "qwen3_8b")
HF_MODEL       = os.environ.get("BFCL_HF_MODEL",       "Qwen/Qwen3-8B")
SFT_CHECKPOINT = f"checkpoints/{MODEL_KEY}/smoke_sft_final"
OUTPUT_DIR     = f"checkpoints/{MODEL_KEY}/smoke_grpo"


def main():
    if not Path(SFT_CHECKPOINT).exists():
        print(f"[SMOKE] ERROR: SFT checkpoint not found at {SFT_CHECKPOINT}")
        print("  Run smoke_sft.py first.")
        sys.exit(1)

    print("[SMOKE] Loading 4 GRPO examples...")
    _, grpo_data = build_dataset(split=0.9)
    grpo_data = grpo_data[:4]
    print(f"  Using {len(grpo_data)} examples")

    for d in grpo_data:
        d["function"]     = _json.dumps(d["function"])
        d["ground_truth"] = _json.dumps(d["ground_truth"])

    dataset = Dataset.from_list(grpo_data)

    print(f"[SMOKE] Loading base model: {HF_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL,
        torch_dtype=torch.bfloat16,
    )
    print(f"[SMOKE] Applying SFT adapter: {SFT_CHECKPOINT}")
    model = PeftModel.from_pretrained(base_model, SFT_CHECKPOINT, is_trainable=True)

    tokenizer = AutoTokenizer.from_pretrained(SFT_CHECKPOINT)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        max_steps=2,                # 2 steps only
        per_device_train_batch_size=2,   # must be divisible by num_generations
        gradient_accumulation_steps=1,
        learning_rate=5e-6,
        num_generations=2,          # K=2 rollouts (minimum for group-relative)
        max_completion_length=64,
        temperature=0.8,
        beta=0.01,
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[bfcl_reward_fn],
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("[SMOKE] Running 2 GRPO steps...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR + "_final")
    tokenizer.save_pretrained(OUTPUT_DIR + "_final")
    print(f"[SMOKE] GRPO smoke test PASSED. Checkpoint: {OUTPUT_DIR}_final")


if __name__ == "__main__":
    main()
