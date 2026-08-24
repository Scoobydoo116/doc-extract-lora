"""Convert raw receipt/invoice annotations into instruction-tuning examples.

Expected raw input: a JSONL file where each line has
    {"text": "<ocr text of the document>", "label": {<fields matching schema.RECEIPT_SCHEMA>}}

Output: a JSONL file where each line has
    {"prompt": "<instruction + schema + document text>", "completion": "<label as compact JSON>"}

This script is dataset-agnostic on purpose - point --raw at whatever
source you've converted into the intermediate {"text", "label"} format
(e.g. SROIE, CORD, or a synthetic set). Add a small per-dataset converter
in this file (or a sibling script) when you pick the source.
"""

import argparse
import json
from pathlib import Path

from schema import validate

INSTRUCTION = (
    "Extract the following fields from the receipt text as a single JSON object: "
    "vendor, date (YYYY-MM-DD), line_items (list of {description, quantity, unit_price, amount}), "
    "subtotal, tax, total. Output only the JSON object, no other text."
)


def build_prompt(document_text: str) -> str:
    return f"{INSTRUCTION}\n\nReceipt text:\n{document_text.strip()}\n\nJSON:"


def convert(raw_path: Path, out_path: Path) -> None:
    n_in, n_valid, n_skipped = 0, 0, 0
    with raw_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            row = json.loads(line)
            label = row["label"]
            ok, err = validate(label)
            if not ok:
                n_skipped += 1
                continue
            example = {
                "prompt": build_prompt(row["text"]),
                "completion": json.dumps(label, ensure_ascii=False, separators=(",", ":")),
            }
            fout.write(json.dumps(example, ensure_ascii=False) + "\n")
            n_valid += 1

    print(f"read {n_in} rows -> wrote {n_valid} examples ({n_skipped} skipped, failed schema validation)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, required=True, help="path to raw {text,label} JSONL")
    ap.add_argument("--out", type=Path, required=True, help="path to write instruction-format JSONL")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    convert(args.raw, args.out)
