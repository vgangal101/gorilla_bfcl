"""
smoke_sft.py — Minimal SFT smoke test.

Mirrors sft_train.py exactly but uses 4 examples and 3 training steps.
Confirms the SFT code path works end-to-end without running for hours.

Run:
    python grpo_pipeline/smoke_tests/smoke_sft.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "berkeley-function-call-leaderboard"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig

from data_prep import build_dataset

MODEL_KEY  = os.environ.get("BFCL_MODEL_KEY", "qwen3_8b")
MODEL      = os.environ.get("BFCL_HF_MODEL",  "Qwen/Qwen3-8B")
OUTPUT_DIR = f"checkpoints/{MODEL_KEY}/smoke_sft"


def main():
    print("[SMOKE] Loading 4 SFT examples...")
    sft_data, _ = build_dataset(split=0.9)
    sft_data = sft_data[:4]
    print(f"  Using {len(sft_data)} examples")

    def format_example(ex):
        return {"text": ex["prompt"] + "\n" + ex["completion"]}

    dataset = Dataset.from_list(sft_data).map(format_example, remove_columns=list(sft_data[0].keys()))

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=4,                        # small rank — smoke test only
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    config = SFTConfig(
        output_dir=OUTPUT_DIR,
        max_steps=3,                # 3 steps only
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        max_seq_length=256,
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=MODEL,
        args=config,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print("[SMOKE] Running 3 SFT steps...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR + "_final")
    tokenizer.save_pretrained(OUTPUT_DIR + "_final")
    print(f"[SMOKE] SFT smoke test PASSED. Checkpoint: {OUTPUT_DIR}_final")


if __name__ == "__main__":
    main()
