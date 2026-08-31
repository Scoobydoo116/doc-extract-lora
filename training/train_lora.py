"""LoRA/QLoRA fine-tuning entry point.

Usage (once data/processed/{train,val}.jsonl exist - see data/prepare.py):

    python training/train_lora.py \
        --base-model Qwen/Qwen2.5-3B-Instruct \
        --train-file data/processed/train.jsonl \
        --val-file data/processed/val.jsonl \
        --output-dir checkpoints/qwen2.5-3b-doc-extract \
        --epochs 3 --lr 2e-4 --lora-r 16

Requires a CUDA GPU with enough VRAM for the chosen base model (QLoRA
4-bit quantization keeps a 3B model trainable on ~12GB). Not runnable
on a CPU-only machine in reasonable time - use Colab/RunPod/Lambda if
you don't have a local GPU.
"""

import argparse
import gc
import os

# reduces allocator fragmentation on memory-constrained GPUs - must be set
# before torch initializes a CUDA context
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-model", required=True, help="HF model id, e.g. Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--val-file", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument(
        "--per-device-batch-size",
        type=int,
        default=2,
        help="without gradient checkpointing, a T4's ~14.5GB gets tight around batch_size=4 at max_length=1280",
    )
    ap.add_argument("--grad-accum", type=int, default=8, help="paired with the lower default batch size to keep the same effective batch size (16)")
    ap.add_argument(
        "--max-seq-len",
        type=int,
        default=1280,
        help="measured on the actual training data: median 721 tokens, p90 1039, max 1233 - "
        "1024 truncates the completion (which comes after the prompt) on the longer ~10%% of examples",
    )
    ap.add_argument("--no-4bit", action="store_true", help="disable QLoRA 4-bit quantization")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # bf16 needs Ampere+ (compute capability 8.0+, e.g. A100) - older cards
    # like a T4 (Turing, 7.5) only support fp16. Detect instead of assuming.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    quant_config = None
    if not args.no_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        device_map="auto",
    )
    if quant_config is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            # the reentrant (default) checkpoint implementation is a known
            # source of "different number of tensors saved" errors with
            # 4-bit quantized layers - non-reentrant avoids it
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = load_dataset("json", data_files=args.train_file, split="train")
    val_ds = load_dataset("json", data_files=args.val_file, split="train")
    # each example is already {"prompt": ..., "completion": ...} - trl
    # recognizes this shape as its "prompt-completion" dataset format and
    # automatically masks the loss to the completion tokens only

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_batch_size,
        # trl defaults this to 8 independently of the train batch size, which
        # can push a tight-fitting training config over the edge during the
        # end-of-epoch eval pass
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_seq_len,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=use_bf16,
        fp16=not use_bf16,
        report_to=["wandb"],
        # must match the use_reentrant=False passed to prepare_model_for_kbit_training
        # above - trl re-applies gradient checkpointing via this config, so both
        # need to agree or one silently overrides the other
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    gc.collect()
    torch.cuda.empty_cache()

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
