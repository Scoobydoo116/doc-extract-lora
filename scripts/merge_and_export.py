"""Merge a trained LoRA adapter into its base model and save the result
as a standalone model directory - what the eval harness's --backend local
option and the FastAPI demo (app/webdemo/main.py) both expect to load.

Run this after training/train_lora.py (or after downloading the adapter
produced by notebooks/finetune_doc_extract.ipynb) and before either
evaluating the fine-tuned model or quantizing it for edge deployment
(GGUF/ONNX conversion tools generally expect a merged model, not a
base model + separate adapter).

Usage:
    python scripts/merge_and_export.py \
        --base-model Qwen/Qwen2.5-1.5B-Instruct \
        --adapter checkpoints/doc-extract-lora/final_adapter \
        --out checkpoints/doc-extract-lora/merged
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-model", required=True, help="HF model id the adapter was trained on top of")
    ap.add_argument("--adapter", required=True, help="path to the trained LoRA adapter")
    ap.add_argument("--out", required=True, help="path to write the merged, standalone model")
    args = ap.parse_args()

    print(f"loading base model {args.base_model}...")
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"loading adapter from {args.adapter}...")
    model = PeftModel.from_pretrained(base, args.adapter)

    print("merging adapter into base weights...")
    merged = model.merge_and_unload()

    print(f"saving merged model to {args.out}...")
    merged.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print("done")


if __name__ == "__main__":
    main()
