import json

from metrics import aggregate, line_items_prf, score_example, scalar_field_match, try_parse_json

GOLD = {
    "vendor": "Coffee Shop Inc.",
    "date": "2026-03-14",
    "line_items": [
        {"description": "Latte", "quantity": 1, "unit_price": 4.5, "amount": 4.5},
        {"description": "Croissant", "quantity": 2, "unit_price": 3.0, "amount": 6.0},
    ],
    "subtotal": 10.5,
    "tax": 0.9,
    "total": 11.4,
}


def test_try_parse_json_plain():
    assert try_parse_json(json.dumps(GOLD)) == GOLD


def test_try_parse_json_markdown_fenced():
    text = "```json\n" + json.dumps(GOLD) + "\n```"
    assert try_parse_json(text) == GOLD


def test_try_parse_json_with_surrounding_commentary():
    text = "Sure, here's the JSON:\n" + json.dumps(GOLD) + "\nLet me know if you need anything else!"
    assert try_parse_json(text) == GOLD


def test_try_parse_json_garbage_returns_none():
    assert try_parse_json("not json at all") is None


def test_try_parse_json_malformed_returns_none():
    assert try_parse_json('{"vendor": "X", "total":}') is None


def test_scalar_field_match_all_correct():
    matches = scalar_field_match(GOLD, GOLD)
    assert all(matches.values())


def test_scalar_field_match_detects_wrong_value():
    pred = {**GOLD, "total": 999.0}
    matches = scalar_field_match(GOLD, pred)
    assert matches["total"] is False
    assert matches["vendor"] is True


def test_line_items_prf_perfect_match():
    result = line_items_prf(GOLD["line_items"], GOLD["line_items"])
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0


def test_line_items_prf_partial_match():
    pred = [GOLD["line_items"][0]]  # missing the croissant
    result = line_items_prf(GOLD["line_items"], pred)
    assert result["precision"] == 1.0
    assert result["recall"] == 0.5


def test_line_items_prf_extra_hallucinated_item():
    pred = GOLD["line_items"] + [{"description": "Bagel", "quantity": 1, "unit_price": 2.0, "amount": 2.0}]
    result = line_items_prf(GOLD["line_items"], pred)
    assert result["recall"] == 1.0
    assert result["precision"] < 1.0


def test_line_items_prf_handles_currency_formatted_amount():
    # a base model with no reason to follow our schema may emit "amount" as
    # a currency-formatted string instead of a plain number - this must not
    # crash the eval run, just fail to match
    pred = [{"description": "Latte", "quantity": 1, "unit_price": 4.5, "amount": "$4.50"}]
    result = line_items_prf(GOLD["line_items"], pred)
    assert result["precision"] == 1.0  # "$4.50" coerces to 4.5, matches gold
    assert result["recall"] == 0.5


def test_line_items_prf_handles_unparseable_amount():
    pred = [{"description": "Latte", "quantity": 1, "unit_price": 4.5, "amount": "N/A"}]
    result = line_items_prf(GOLD["line_items"], pred)  # must not raise
    assert result["precision"] == 0.0


def test_line_items_prf_both_empty_is_perfect():
    result = line_items_prf([], [])
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0


def test_score_example_invalid_json():
    result = score_example(GOLD, "not valid json")
    assert result["valid_json"] is False
    assert result["schema_valid"] is False


def test_score_example_valid_and_correct():
    result = score_example(GOLD, json.dumps(GOLD))
    assert result["valid_json"] is True
    assert result["schema_valid"] is True
    assert all(result["scalar_matches"].values())
    assert result["line_items"]["f1"] == 1.0


def test_aggregate_mixed_results():
    good = score_example(GOLD, json.dumps(GOLD))
    bad = score_example(GOLD, "garbage")
    summary = aggregate([good, bad])
    assert summary["n_examples"] == 2
    assert summary["json_validity_rate"] == 0.5
    assert summary["schema_validity_rate"] == 0.5
    # field accuracy is only computed over examples that produced valid JSON
    assert summary["mean_field_accuracy"] == 1.0
