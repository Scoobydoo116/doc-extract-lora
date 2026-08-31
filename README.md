# doc-extract-lora

A small, fine-tuned language model that turns messy documents (receipts/invoices) into clean structured JSON, benchmarked against its own base model and a large hosted LLM, then quantized and deployed as a live demo.

## Why this project

Most "LLM projects" are thin wrappers around a hosted API. This one instead asks: **can a small open model, fine-tuned on a narrow task, match a much larger model's accuracy at a fraction of the cost and latency?** That requires actually training something (LoRA fine-tuning), measuring it rigorously (field-level accuracy, not just vibes), and making the tradeoffs explicit (quality vs. size vs. speed vs. cost).

Task: **structured extraction**. Given raw OCR/text of a receipt or invoice, output a JSON object conforming to a fixed schema (vendor, date, line items, subtotal, tax, total).

## Why structured extraction specifically

- Objective evaluation: compare extracted fields against ground truth with **field-level precision/recall/F1**, plus a **JSON-validity rate** (did it even produce parseable, schema-conformant JSON?)
- Real commercial relevance: this exact task sits underneath expense automation, invoice processing, and every "upload a doc, get structured data" product
- Immediately demoable: paste a messy receipt, watch clean JSON come out, side-by-side with the base model producing malformed or incomplete output
- Small, well-scoped output schema makes it tractable to fine-tune well in a few weeks

## Target schema (v1)

```json
{
  "vendor": "string",
  "date": "YYYY-MM-DD",
  "line_items": [
    {"description": "string", "quantity": "number", "unit_price": "number", "amount": "number"}
  ],
  "subtotal": "number",
  "tax": "number",
  "total": "number"
}
```

## Architecture

```
Raw document text (OCR output or plain text)
            |
            v
   Fine-tuned small LLM (LoRA)
            |
            v
   Generated JSON (schema-constrained)
            |
            v
  Validated against JSON schema
            |
            v
   Shown in UI next to source doc
```

## Repo layout

```
data/           dataset sourcing + preprocessing (receipts/invoices -> instruction format)
training/       LoRA/QLoRA fine-tuning scripts
eval/           field-level accuracy + JSON-validity evaluation harness
scripts/        one-off utility scripts (data fetch/convert, adapter merge/export)
app/webdemo/    FastAPI backend + minimal frontend for the live demo
notebooks/      self-contained Colab notebook for baseline eval + fine-tuning
tests/          pytest suite for schema validation, eval metrics, and data conversion
```

## Data

**Source:** [`mychen76/invoices-and-receipts_ocr_v1`](https://huggingface.co/datasets/mychen76/invoices-and-receipts_ocr_v1) (2,043 synthetic invoices with OCR word lists + structured header/items/summary labels).

CORD and SROIE (the two standard receipt-parsing benchmarks) were the original plan, but CORD-v2's rows bundle full images and exceed the HF datasets-server scan limit even for a single row, and full parquet streaming was too slow for this pipeline's needs. Since the task only needs OCR text (not pixels), a text-only dataset was a better fit anyway - this one is synthetic rather than scanned, which is a real limitation (no genuine OCR noise like misreads or skew), but it unblocks getting a real pipeline and real numbers now. Swapping in SROIE/CORD (or a licensed real-world set) later to check whether results hold up on real OCR is on the roadmap.

**Pipeline:** `scripts/fetch_dataset.py` (pull all rows via the HF datasets-server REST API, text fields only) -> `scripts/convert_invoices.py` (map this dataset's header/items/summary shape onto our schema, handling European-formatted numbers and a labeling gap where ~80% of rows leave the structured `invoice_date` field blank even though the date is present in the OCR text - recovered via regex fallback) -> `data/prepare.py` (build final instruction-tuning JSONL).

**Yield:** 403 of 2,043 rows survive as clean, schema-valid examples (363 train / 40 val). The other ~80% are dropped because they have *no* items and *no* date in the structured labels (confirmed correlated - not a bug in the converter, just incomplete upstream labeling for those rows). 403 examples is workable for LoRA fine-tuning on a narrow task, but is a real ceiling on this dataset; a larger or cleaner source would be the first thing to revisit if fine-tuning results are noisy.

**Known data-quality issue to note in the eventual write-up:** in some rows the `summary.subtotal` doesn't sum from the line items shown - the source dataset has some internal inconsistency. Left as-is rather than "fixed," since forcing consistency would mean overwriting the ground truth with a guess. Worth calling out explicitly in the final write-up as a limitation, since it puts a ceiling on how high field accuracy can meaningfully go on this data.

## Plan (roughly 10-12 weeks)

**Phase 1 - Data & baseline (weeks 1-2)**
- [x] Source a receipt/invoice dataset (see Data section above)
- [x] Define the JSON schema precisely (above is a v1 draft) and write a validator
- [x] Convert to instruction-tuning format: `(document text, schema) -> JSON`
- [ ] Get a baseline number: run the *base* model (zero-shot / few-shot prompted) through the eval harness - needs GPU access (Colab/RunPod/Lambda) or an API key, not available in this environment

**Phase 2 - Fine-tuning (weeks 3-6)**
- [x] Pick base model (Qwen2.5-1.5B-Instruct by default in the notebook - fits a free-tier T4 with QLoRA; swap to 3B on a bigger GPU)
- [x] Build a ready-to-run notebook for baseline eval + LoRA/QLoRA fine-tuning (`notebooks/finetune_doc_extract.ipynb`) - **run this next**, it's the blocking step for everything below
- [ ] Track experiments (learning rate, rank, epochs) with W&B
- [ ] Re-run eval harness on fine-tuned checkpoint, compare to baseline
- [x] Merge/export script for turning a trained adapter into a standalone deployable model (`scripts/merge_and_export.py`)

**Phase 3 - Benchmarking (weeks 6-7)**
- [ ] Add a large hosted model (e.g. GPT-4o-class or Claude) as the "ceiling" comparison point
- [ ] Build results table: field-level F1, JSON-validity rate, latency, cost per 1k documents, model size
- [ ] Error analysis: what kinds of documents/fields does the small fine-tuned model still get wrong (multi-item tables? handwritten totals? currency formatting?)

**Phase 4 - Efficiency & deployment (weeks 8-10)**
- [ ] Quantize fine-tuned model (GGUF via llama.cpp, or ONNX)
- [ ] Get it running in-browser (transformers.js / WebGPU) or as a lightweight hosted API
- [ ] Build the demo UI: paste/upload a receipt, see extracted JSON rendered as a clean summary

**Phase 5 - Write-up & polish (weeks 11-12)**
- [ ] README results table + charts (accuracy vs. model size, accuracy vs. latency)
- [ ] Architecture diagram
- [ ] Short writeup: what worked, what didn't, what you'd try next

## Status

Data pipeline is real and verified (403 examples, fetched/converted/tested against the live source). No successful training run yet - that happens in `notebooks/finetune_doc_extract.ipynb`, which needs a GPU this environment doesn't have. Once that produces an adapter, `scripts/merge_and_export.py` turns it into a model the eval harness and demo can load directly.

The deterministic logic (schema validation, eval metrics, data conversion) has a pytest suite (`tests/`, 37 tests) that runs in CI on every push (`.github/workflows/tests.yml`) - this is the code that's easy to get subtly wrong (number parsing, JSON extraction, line-item matching) and easy to test without a GPU, so it's tested now rather than left until something looks wrong during training.

### Resume point (paused mid-debugging session, day 2)

Working through the Colab notebook has hit a chain of errors, each fixed as it surfaced (the notebook's `pip install -U` always grabs latest `trl`/`transformers`/`peft`, and their APIs have moved since this was written). In order:

1. `line_items_prf` crashed on a currency-formatted `"amount"` like `"$2,620.00"` from the base model - fixed with lenient coercion (`eval/metrics.py`)
2. `SFTConfig.__init__()` no longer accepts `max_seq_length` - renamed to `max_length`
3. `bf16=True` was hardcoded, but free-tier Colab's T4 (Turing, 7.5) doesn't support it - only Ampere+ (8.0+) does. Fixed to auto-detect via `torch.cuda.is_bf16_supported()`
4. `SFTTrainer` rejected `formatting_func` once trl auto-detected our `{"prompt","completion"}` dataset shape as its native prompt-completion format - fixed by dropping `formatting_func` entirely
5. Measured actual token lengths in the data (median 721, p90 1039, max 1233) and found `max_length=1024` was silently truncating completions on the longest ~10% of examples - bumped to 1280. Also reduced to 1 epoch for a faster first validation pass.
6. `CheckpointError: different number of tensors saved during forward and recomputation` - a known gradient-checkpointing + bitsandbytes-4-bit interaction. **First attempt (wrong call): disabled gradient checkpointing entirely** to dodge it.
7. That caused a genuine `OutOfMemoryError` (T4's ~14.5GB, short by only 112MB). Tried lowering batch sizes (train 4→2, and setting `per_device_eval_batch_size` which had defaulted to 8, independent of train batch size) - **same OOM recurred with byte-for-byte identical numbers**, which strongly suggested the fix was never actually picked up by the live Colab session (stale notebook state), not that the fix was wrong.
8. Rather than keep guessing one variable at a time - the user was reasonably frustrated after 3 days on this - fix #6 was corrected properly instead of avoided: `gradient_checkpointing_kwargs={"use_reentrant": False}` (the actual documented fix), **and** the notebook's default model was switched from Qwen2.5-1.5B-Instruct to **Qwen2.5-0.5B-Instruct** for this validation pass (roughly a third the memory, makes OOM unlikely regardless of the checkpointing question). Also added `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and an explicit memory clear before training.

**Commit `5f5a15c` has all of this. It has not been confirmed working yet** - the session ended before the user re-ran it. Verification limit to be upfront about: 4-bit quantization needs real CUDA/bitsandbytes kernels not available in this dev environment, so none of the gradient-checkpointing or OOM fixes could be verified beyond "the code runs and uses the right API" - only a real T4 run confirms the actual memory behavor.

**Next step:** have the user do a **full restart** (`Runtime → Restart session`, then `File → Revert notebook` or reopen the notebook link fresh - a prior round was wasted because the live session didn't actually have the latest code) and run it again. Get the actual result (success, or a new traceback) before changing anything else. If it OOMs again even at 0.5B, something structural needs rethinking (e.g. batch_size=1, or moving off T4 to Colab Pro) rather than another small parameter tweak - three OOM/checkpoint round trips on the same cell is a signal to step back, not keep iterating narrowly.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements-dev.txt   # includes pytest; use requirements.txt for runtime-only
pytest tests/ -v
```

GPU note: LoRA fine-tuning of a 1.5-3B model is feasible on a single consumer GPU (12GB+ VRAM) with QLoRA, or on a cloud GPU (Colab Pro, RunPod, Lambda) if you don't have one locally - or just use the Colab notebook, which needs nothing installed locally.
