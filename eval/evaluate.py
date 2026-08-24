"""Run the eval harness over a JSONL eval set for any of:
  - a local HF/PEFT model (base or fine-tuned)
  - a hosted API model (used as the "large model" ceiling comparison)

Records per-example latency alongside accuracy so the results table can
show the quality/latency/cost tradeoff, not just accuracy in isolation.

Usage:
    python eval/evaluate.py --backend local --model checkpoints/qwen2.5-3b-doc-extract \
        --eval-file data/processed/val.jsonl --out results/finetuned.json

    python eval/evaluate.py --backend local --model Qwen/Qwen2.5-3B-Instruct \
        --eval-file data/processed/val.jsonl --out results/base.json

    python eval/evaluate.py --backend anthropic --model claude-sonnet-5 \
        --eval-file data/processed/val.jsonl --out results/claude.json
"""

import argparse
import json
import time
from pathlib import Path
from typing import Protocol

from metrics import aggregate, score_example


class Backend(Protocol):
    def generate(self, prompt: str) -> str: ...


class LocalHFBackend:
    def __init__(self, model_path: str, max_new_tokens: int = 512):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", torch_dtype=torch.bfloat16
        )
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
        )
        text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text


class AnthropicBackend:
    """Ceiling-comparison backend. Requires ANTHROPIC_API_KEY in the environment."""

    def __init__(self, model: str):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model

    def generate(self, prompt: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text


def build_backend(name: str, model: str) -> Backend:
    if name == "local":
        return LocalHFBackend(model)
    if name == "anthropic":
        return AnthropicBackend(model)
    raise ValueError(f"unknown backend: {name}")


def run(backend: Backend, eval_file: Path) -> dict:
    results = []
    latencies = []
    with eval_file.open(encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]

    for ex in examples:
        gold = json.loads(ex["completion"])
        t0 = time.perf_counter()
        raw_output = backend.generate(ex["prompt"])
        latencies.append(time.perf_counter() - t0)
        results.append(score_example(gold, raw_output))

    summary = aggregate(results)
    summary["mean_latency_sec"] = sum(latencies) / len(latencies)
    summary["p95_latency_sec"] = sorted(latencies)[int(0.95 * len(latencies))]
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["local", "anthropic"], required=True)
    ap.add_argument("--model", required=True, help="HF model path/id, or API model name")
    ap.add_argument("--eval-file", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    backend = build_backend(args.backend, args.model)
    summary = run(backend, args.eval_file)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
