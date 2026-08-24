from schema import validate

VALID = {
    "vendor": "Coffee Shop Inc.",
    "date": "2026-03-14",
    "line_items": [
        {"description": "Latte", "quantity": 1, "unit_price": 4.5, "amount": 4.5},
    ],
    "subtotal": 4.5,
    "tax": 0.4,
    "total": 4.9,
}


def test_valid_record_passes():
    ok, err = validate(VALID)
    assert ok
    assert err is None


def test_missing_required_field_fails():
    record = {k: v for k, v in VALID.items() if k != "total"}
    ok, _ = validate(record)
    assert not ok


def test_wrong_type_fails():
    record = {**VALID, "subtotal": "4.50"}
    ok, _ = validate(record)
    assert not ok


def test_additional_property_rejected():
    record = {**VALID, "notes": "extra field not in schema"}
    ok, _ = validate(record)
    assert not ok


def test_bad_date_format_rejected():
    record = {**VALID, "date": "03/14/2026"}
    ok, _ = validate(record)
    assert not ok


def test_line_item_missing_field_rejected():
    record = {**VALID, "line_items": [{"description": "Latte", "quantity": 1, "unit_price": 4.5}]}
    ok, _ = validate(record)
    assert not ok


def test_empty_line_items_is_valid():
    record = {**VALID, "line_items": []}
    ok, _ = validate(record)
    assert ok
