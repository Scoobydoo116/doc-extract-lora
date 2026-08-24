"""Pull the mychen76/invoices-and-receipts_ocr_v1 dataset via the HF
datasets-server REST API (rows endpoint), text fields only - deliberately
avoids the `datasets` library's parquet loader, whose row groups bundle
full images and are too large to scan efficiently for a text-only task.

Source: https://huggingface.co/datasets/mychen76/invoices-and-receipts_ocr_v1
2043 synthetic invoices with OCR word lists + structured header/items/summary
labels. Synthetic, so not a substitute for real-world OCR noise, but a
reasonable dataset to get the pipeline working end-to-end; swap in
SROIE/CORD (or a licensed real dataset) later if the eval numbers need to
reflect real-world OCR error patterns.

Usage:
    python scripts/fetch_dataset.py --out data/raw/invoices_raw.jsonl
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

DATASET = "mychen76/invoices-and-receipts_ocr_v1"
PAGE_SIZE = 100
API = "https://datasets-server.huggingface.co/rows"


def fetch_page(offset: int, length: int) -> dict:
    url = (
        f"{API}?dataset={DATASET.replace('/', '%2F')}"
        f"&config=default&split=train&offset={offset}&length={length}"
    )
    max_attempts = 8
    for attempt in range(max_attempts):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if attempt == max_attempts - 1:
                raise
            # 429 needs a longer backoff than transient 5xx errors - the API
            # rate-limits bursts of requests (e.g. re-running this script
            # right after a previous run), and a short backoff just gets
            # rate-limited again
            wait = 30 if e.code == 429 else 2**attempt
            print(f"  retry {attempt + 1} after HTTP {e.code} (waiting {wait}s)")
            time.sleep(wait)
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            wait = 2**attempt
            print(f"  retry {attempt + 1} after error: {e} (waiting {wait}s)")
            time.sleep(wait)


def main(out_path: Path) -> None:
    first = fetch_page(0, 1)
    total = first["num_rows_total"]
    print(f"dataset has {total} rows")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        offset = 0
        while offset < total:
            length = min(PAGE_SIZE, total - offset)
            page = fetch_page(offset, length)
            for row in page["rows"]:
                r = row["row"]
                # drop the image field entirely - we only need OCR text + labels
                record = {
                    "id": r["id"],
                    "parsed_data": r["parsed_data"],
                    "raw_data": r["raw_data"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1
            offset += length
            print(f"  fetched {offset}/{total}")

    print(f"wrote {n_written} rows -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/raw/invoices_raw.jsonl"))
    args = ap.parse_args()
    main(args.out)
