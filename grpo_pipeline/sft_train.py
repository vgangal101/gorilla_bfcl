"""
sft_train.py — Phase 1: Supervised fine-tuning on BFCL ground-truth examples.

Warm-starts the model so that GRPO phase begins with parseable outputs.
Uses LoRA so 8B fits on a single A100-80G.

Run:
    python grpo_pipeline/sft_train.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "berkeley-function-call-leaderboard"))

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig

from data_prep import build_dataset

# Configurable via environment variables — set by run_pipeline.slurm
MODEL_KEY = os.environ.get("BFCL_MODEL_KEY", "qwen3_8b")
MODEL     = os.environ.get("BFCL_HF_MODEL",  "Qwen/Qwen3-8B")
OUTPUT_DIR = f"checkpoints/{MODEL_KEY}/sft"


def main():
    print("Loading BFCL data...")
    sft_data, _ = build_dataset(split=0.9)
    print(f"  {len(sft_data)} SFT examples across {len(set(d['category'] for d in sft_data))} categories")

    # SFTTrainer expects a single "text" field: prompt + completion concatenated.
    # Only pass prompt+completion to Dataset — the function/ground_truth fields
    # contain nested dicts with mixed types that PyArrow cannot schema-infer.
    def format_example(ex):
        return {"text": ex["prompt"] + "\n" + ex["completion"]}

    sft_simple = [{"prompt": d["prompt"], "completion": d["completion"]} for d in sft_data]
    dataset = Dataset.from_list(sft_simple).map(format_example, remove_columns=["prompt", "completion"])

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,    # effective batch = 16
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_seq_length=1024,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to="none",                 # swap to "wandb" if you want tracking
    )

    trainer = SFTTrainer(
        model=MODEL,
        args=config,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print("Starting SFT...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR + "_final")
    tokenizer.save_pretrained(OUTPUT_DIR + "_final")
    print(f"SFT complete. Checkpoint saved to {OUTPUT_DIR}_final")


if __name__ == "__main__":
    main()
