"""
merge_lora.py — Merge a PEFT LoRA adapter into the base model weights.

Called by run_pipeline.slurm before vLLM serving so the evaluation phase
can serve a plain HuggingFace model instead of juggling vLLM LoRA aliases.

Usage:
    python grpo_pipeline/merge_lora.py <hf_model_id> <adapter_path> <output_path>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "berkeley-function-call-leaderboard"))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge(hf_model: str, adapter_path: str, output_path: str) -> None:
    print(f"Loading base model: {hf_model}")
    base = AutoModelForCausalLM.from_pretrained(
        hf_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",   # spread across GPUs — merge only, not training
    )

    print(f"Applying adapter: {adapter_path}")
    peft_model = PeftModel.from_pretrained(base, adapter_path)

    print("Merging and unloading adapter...")
    merged = peft_model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    merged.save_pretrained(output_path)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(output_path)

    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python merge_lora.py <hf_model> <adapter_path> <output_path>")
        sys.exit(1)
    merge(sys.argv[1], sys.argv[2], sys.argv[3])
