"""JSON schema for the extraction target and a validator used by both
the data-prep pipeline (to sanity-check labels) and the eval harness
(to score model output)."""

from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator, ValidationError

RECEIPT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["vendor", "date", "line_items", "subtotal", "tax", "total"],
    "additionalProperties": False,
    "properties": {
        "vendor": {"type": "string"},
        "date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"},
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["description", "quantity", "unit_price", "amount"],
                "additionalProperties": False,
                "properties": {
                    "description": {"type": "string"},
                    "quantity": {"type": "number"},
                    "unit_price": {"type": "number"},
                    "amount": {"type": "number"},
                },
            },
        },
        "subtotal": {"type": "number"},
        "tax": {"type": "number"},
        "total": {"type": "number"},
    },
}

_validator = Draft202012Validator(RECEIPT_SCHEMA)

FIELDS = ("vendor", "date", "subtotal", "tax", "total")


def validate(record: dict[str, Any]) -> tuple[bool, str | None]:
    """Return (is_valid, error_message)."""
    try:
        _validator.validate(record)
        return True, None
    except ValidationError as e:
        return False, e.message


def is_valid_json_schema(record: Any) -> bool:
    ok, _ = validate(record) if isinstance(record, dict) else (False, None)
    return ok
