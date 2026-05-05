"""
grpo_train.py — Phase 2: GRPO fine-tuning using BFCL AST checker as reward.

Starts from the SFT checkpoint. Generates K=4 rollouts per prompt,
scores each with ast_checker, and updates via group-relative policy gradient.

Run:
    python grpo_pipeline/grpo_train.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "berkeley-function-call-leaderboard"))

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

from data_prep import build_dataset
from reward import bfcl_reward_fn

SFT_CHECKPOINT = "checkpoints/sft_final"
OUTPUT_DIR = "checkpoints/grpo"


def main():
    print("Loading BFCL data...")
    _, grpo_data = build_dataset(split=0.9)
    print(f"  {len(grpo_data)} GRPO examples")

    # GRPOTrainer reads extra dataset columns and passes them as kwargs to reward_fn.
    # We keep category, function, ground_truth so bfcl_reward_fn can use them.
    dataset = Dataset.from_list(grpo_data)

    tokenizer = AutoTokenizer.from_pretrained(SFT_CHECKPOINT)
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

    trainer = GRPOTrainer(
        model=SFT_CHECKPOINT,
        reward_funcs=[bfcl_reward_fn],
        args=config,
        train_dataset=dataset,
        peft_config=lora_config,
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
