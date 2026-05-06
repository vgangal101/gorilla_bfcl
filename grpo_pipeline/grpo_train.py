"""
grpo_train.py — Phase 2: GRPO fine-tuning using BFCL AST checker as reward.

Starts from the SFT checkpoint. Generates K=4 rollouts per prompt,
scores each with ast_checker, and updates via group-relative policy gradient.

Run:
    python grpo_pipeline/grpo_train.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "berkeley-function-call-leaderboard"))

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

import json as _json

from data_prep import build_dataset
from reward import bfcl_reward_fn

# Configurable via environment variables — set by run_pipeline.slurm
MODEL_KEY      = os.environ.get("BFCL_MODEL_KEY",      "qwen3_8b")
HF_MODEL       = os.environ.get("BFCL_HF_MODEL",       "Qwen/Qwen3-8B")
SFT_CHECKPOINT = os.environ.get("BFCL_SFT_CHECKPOINT", f"checkpoints/{MODEL_KEY}/sft_final")
OUTPUT_DIR     = f"checkpoints/{MODEL_KEY}/grpo"


def main():
    print("Loading BFCL data...")
    _, grpo_data = build_dataset(split=0.9)
    print(f"  {len(grpo_data)} GRPO examples")

    # Serialize nested dicts to JSON strings so PyArrow can build a consistent
    # Arrow schema. reward.py deserializes them back before calling ast_checker.
    for d in grpo_data:
        d["function"]     = _json.dumps(d["function"])
        d["ground_truth"] = _json.dumps(d["ground_truth"])

    dataset = Dataset.from_list(grpo_data)

    # Load base model then apply the SFT LoRA adapter.
    # GRPOTrainer cannot load a PEFT adapter directory directly via model=path
    # because the directory has no full model weights — only adapter weights.
    print(f"Loading base model: {HF_MODEL}")
    # No device_map — accelerate/DDP assigns each process its own GPU.
    # device_map="auto" would spread the model across ALL visible GPUs inside
    # each DDP worker (model parallelism), conflicting with DDP's expectation
    # that each process holds the full model replica.
    base_model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL,
        torch_dtype=torch.bfloat16,
    )
    print(f"Applying SFT adapter: {SFT_CHECKPOINT}")
    # is_trainable=True keeps LoRA params unfrozen for continued GRPO training
    model = PeftModel.from_pretrained(base_model, SFT_CHECKPOINT, is_trainable=True)

    tokenizer = AutoTokenizer.from_pretrained(SFT_CHECKPOINT)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,    # effective batch = 16
        learning_rate=5e-6,               # lower than SFT
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        # GRPO-specific
        num_generations=4,                # K rollouts per prompt
        max_new_tokens=256,
        temperature=0.8,                  # needs variance for group rewards
        beta=0.01,                        # KL penalty vs reference (SFT) policy
        # Logging / saving
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to="none",                 # swap to "wandb" if you want tracking
    )

    # No peft_config — model already has the SFT LoRA loaded with is_trainable=True
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[bfcl_reward_fn],
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting GRPO...")
    print("  Watch 'rewards/mean' and 'rewards/std' in logs.")
    print("  If std stays near 0 for >50 steps, increase temperature or check reward fn.")
    trainer.train()
    trainer.save_model(OUTPUT_DIR + "_final")
    tokenizer.save_pretrained(OUTPUT_DIR + "_final")
    print(f"GRPO complete. Checkpoint saved to {OUTPUT_DIR}_final")


if __name__ == "__main__":
    main()
