"""Demo API: paste receipt text, get back extracted JSON.

    uvicorn app.webdemo.main:app --reload --port 8000

Then open app/webdemo/index.html (served at / below) in a browser.

Expects a fine-tuned model at MODEL_PATH (default: checkpoints/latest).
Falls back to raising a clear error at request time (not import time)
so the server can still start before training is done.
"""

import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from metrics import try_parse_json  # noqa: E402
from prepare import build_prompt  # noqa: E402

MODEL_PATH = os.environ.get("DOC_EXTRACT_MODEL_PATH", "checkpoints/latest")

app = FastAPI(title="doc-extract-lora demo")

_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return
    if not Path(MODEL_PATH).exists():
        raise HTTPException(
            status_code=503,
            detail=f"no model found at {MODEL_PATH} - train one first (training/train_lora.py) "
            f"or set DOC_EXTRACT_MODEL_PATH",
        )
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # bf16 needs Ampere+ (compute capability 8.0+) - older cards like a T4
    # only support fp16, so detect instead of assuming
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    _model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=dtype)


class ExtractRequest(BaseModel):
    text: str


class ExtractResponse(BaseModel):
    raw_output: str
    parsed: dict | None


@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest) -> ExtractResponse:
    _load_model()
    prompt = build_prompt(req.text)
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)
    out = _model.generate(**inputs, max_new_tokens=512, do_sample=False)
    raw = _tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return ExtractResponse(raw_output=raw, parsed=try_parse_json(raw))


app.mount("/", StaticFiles(directory=Path(__file__).parent, html=True), name="static")
