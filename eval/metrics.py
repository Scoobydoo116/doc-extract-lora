"""Scoring functions for the extraction task.

Two things get measured, deliberately kept separate:
  1. JSON validity rate - did the model even produce parseable output
     that conforms to the schema? (a model that's fluent prose but never
     valid JSON is useless here, regardless of "accuracy")
  2. Field-level precision/recall/F1 - among valid outputs, how correct
     is the extracted content?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from schema import FIELDS, validate  # noqa: E402


def try_parse_json(text: str) -> dict[str, Any] | None:
    """Best-effort JSON extraction: models sometimes wrap output in
    markdown fences or trailing commentary despite instructions."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def scalar_field_match(gold: dict, pred: dict) -> dict[str, bool]:
    return {f: gold.get(f) == pred.get(f) for f in FIELDS}


def line_items_prf(gold_items: list[dict], pred_items: list[dict]) -> dict[str, float]:
    def key(item: dict) -> tuple:
        return (
            str(item.get("description", "")).strip().lower(),
            round(float(item.get("amount", 0) or 0), 2),
        )

    gold_keys = [key(i) for i in gold_items]
    pred_keys = [key(i) for i in pred_items]

    gold_remaining = list(gold_keys)
    tp = 0
    for k in pred_keys:
        if k in gold_remaining:
            tp += 1
            gold_remaining.remove(k)

    precision = tp / len(pred_keys) if pred_keys else (1.0 if not gold_keys else 0.0)
    recall = tp / len(gold_keys) if gold_keys else (1.0 if not pred_keys else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def score_example(gold: dict, raw_prediction: str) -> dict[str, Any]:
    pred = try_parse_json(raw_prediction)
    if pred is None:
        return {"valid_json": False, "schema_valid": False}

    schema_ok, _ = validate(pred)

    result: dict[str, Any] = {"valid_json": True, "schema_valid": schema_ok}
    result["scalar_matches"] = scalar_field_match(gold, pred)
    result["line_items"] = line_items_prf(
        gold.get("line_items", []), pred.get("line_items", []) if isinstance(pred.get("line_items"), list) else []
    )
    return result


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    valid_json_rate = sum(r["valid_json"] for r in results) / n
    schema_valid_rate = sum(r["schema_valid"] for r in results) / n

    scored = [r for r in results if r.get("scalar_matches")]
    field_accuracy = {}
    if scored:
        for f in FIELDS:
            field_accuracy[f] = sum(r["scalar_matches"][f] for r in scored) / len(scored)

    line_item_f1 = sum(r["line_items"]["f1"] for r in scored) / len(scored) if scored else 0.0

    return {
        "n_examples": n,
        "json_validity_rate": valid_json_rate,
        "schema_validity_rate": schema_valid_rate,
        "field_accuracy": field_accuracy,
        "mean_field_accuracy": sum(field_accuracy.values()) / len(field_accuracy) if field_accuracy else 0.0,
        "line_items_f1": line_item_f1,
    }
