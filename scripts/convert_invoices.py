"""Convert the raw mychen76/invoices-and-receipts_ocr_v1 rows (fetched by
fetch_dataset.py) into the intermediate {"text", "label"} format that
data/prepare.py expects, mapping this dataset's header/items/summary
shape onto our fixed schema (data/schema.py).

Handles two data-quality quirks observed in the source data:
  - numbers use European formatting (comma decimal, optional '.' thousands
    separator, occasional leading '$')
  - a handful of item rows use "total_net_worth" instead of
    "item_net_worth" for the line amount (inconsistent key naming
    upstream) - fall back to computing qty * unit_price when neither key
    is present

Usage:
    python scripts/convert_invoices.py \
        --raw data/raw/invoices_raw.jsonl \
        --out data/interim/invoices_labeled.jsonl
"""

import argparse
import ast
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from schema import validate  # noqa: E402


def parse_number(raw: str) -> float | None:
    if raw is None:
        return None
    s = re.sub(r"[^\d,.\-]", "", str(raw)).strip()
    if not s:
        return None
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def parse_date(raw: str) -> str | None:
    for fmt in ("%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw.strip(), fmt).strftime("%Y-%m-%d")
        except (ValueError, AttributeError):
            continue
    return None


_DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b")


def _date_from_ocr(ocr_words: list[str]) -> str | None:
    for w in ocr_words:
        m = _DATE_RE.search(w)
        if m:
            d = parse_date(m.group(0))
            if d is not None:
                return d
    return None


def convert_row(row: dict) -> dict | None:
    try:
        parsed = json.loads(row["parsed_data"])
        j = ast.literal_eval(parsed["json"])
        raw = json.loads(row["raw_data"])
        ocr_words = ast.literal_eval(raw["ocr_words"])
    except (KeyError, SyntaxError, ValueError, json.JSONDecodeError):
        return None

    header = j.get("header", {})
    items = j.get("items", [])
    summary = j.get("summary", {})

    date = parse_date(header.get("invoice_date", ""))
    if date is None:
        # the source dataset leaves invoice_date blank in ~80% of rows even
        # though the date is present in the OCR text right after "Date of
        # issue:" - recover it from there instead of discarding the example
        date = _date_from_ocr(ocr_words)
    if date is None:
        return None

    line_items = []
    for it in items:
        desc = it.get("item_desc")
        qty = parse_number(it.get("item_qty"))
        unit_price = parse_number(it.get("item_net_price"))
        amount = parse_number(it.get("item_net_worth") or it.get("total_net_worth"))
        if amount is None and qty is not None and unit_price is not None:
            amount = round(qty * unit_price, 2)
        if desc is None or qty is None or unit_price is None or amount is None:
            continue
        line_items.append(
            {"description": desc, "quantity": qty, "unit_price": unit_price, "amount": amount}
        )
    if not line_items:
        return None

    subtotal = parse_number(summary.get("total_net_worth"))
    tax = parse_number(summary.get("total_vat"))
    total = parse_number(summary.get("total_gross_worth"))
    if subtotal is None or tax is None or total is None:
        return None

    label = {
        "vendor": header.get("seller", "").strip(),
        "date": date,
        "line_items": line_items,
        "subtotal": subtotal,
        "tax": tax,
        "total": total,
    }
    if not label["vendor"]:
        return None

    ok, _ = validate(label)
    if not ok:
        return None

    text = "\n".join(ocr_words)
    return {"text": text, "label": label}


def main(raw_path: Path, out_path: Path) -> None:
    n_in, n_ok = 0, 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            row = json.loads(line)
            converted = convert_row(row)
            if converted is None:
                continue
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            n_ok += 1
    print(f"read {n_in} raw rows -> wrote {n_ok} labeled examples ({n_in - n_ok} skipped)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=Path("data/raw/invoices_raw.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/interim/invoices_labeled.jsonl"))
    args = ap.parse_args()
    main(args.raw, args.out)
