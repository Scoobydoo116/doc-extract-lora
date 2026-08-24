import json

from convert_invoices import _date_from_ocr as date_from_ocr
from convert_invoices import convert_row, parse_date, parse_number


def test_parse_number_european_comma_decimal():
    assert parse_number("7,50") == 7.5


def test_parse_number_with_dollar_sign():
    assert parse_number("$7,50") == 7.5


def test_parse_number_thousands_and_decimal():
    assert parse_number("1.234,56") == 1234.56


def test_parse_number_none_input():
    assert parse_number(None) is None


def test_parse_number_empty_string():
    assert parse_number("") is None


def test_parse_date_mm_dd_yyyy():
    assert parse_date("10/15/2012") == "2012-10-15"


def test_parse_date_invalid_returns_none():
    assert parse_date("not a date") is None


def test_date_from_ocr_finds_date():
    words = ["Invoice no: 123", "Date of issue:", "07/06/2019", "Seller:"]
    assert date_from_ocr(words) == "2019-07-06"


def test_date_from_ocr_no_date_returns_none():
    words = ["Invoice no: 123", "Seller: Acme Corp"]
    assert date_from_ocr(words) is None


def _make_row(header, items, summary, ocr_words):
    py_json_str = repr({"header": header, "items": items, "summary": summary})
    parsed_data = json.dumps({"xml": "", "json": py_json_str})
    raw_data = json.dumps({"ocr_words": repr(ocr_words), "ocr_boxes": "[]", "ocr_labels": "[]"})
    return {"id": 0, "parsed_data": parsed_data, "raw_data": raw_data}


def test_convert_row_happy_path():
    row = _make_row(
        header={"invoice_date": "10/15/2012", "seller": "Acme Corp"},
        items=[{"item_desc": "Widget", "item_qty": "2,00", "item_net_price": "5,00", "item_net_worth": "10,00"}],
        summary={"total_net_worth": "$10,00", "total_vat": "$1,00", "total_gross_worth": "$11,00"},
        ocr_words=["Date of issue:", "10/15/2012"],
    )
    result = convert_row(row)
    assert result is not None
    label = result["label"]
    assert label["vendor"] == "Acme Corp"
    assert label["date"] == "2012-10-15"
    assert label["line_items"] == [{"description": "Widget", "quantity": 2.0, "unit_price": 5.0, "amount": 10.0}]
    assert label["subtotal"] == 10.0
    assert label["total"] == 11.0


def test_convert_row_recovers_date_from_ocr_when_header_blank():
    row = _make_row(
        header={"invoice_date": "", "seller": "Acme Corp"},
        items=[{"item_desc": "Widget", "item_qty": "1,00", "item_net_price": "5,00", "item_net_worth": "5,00"}],
        summary={"total_net_worth": "$5,00", "total_vat": "$0,50", "total_gross_worth": "$5,50"},
        ocr_words=["Date of issue:", "01/02/2020", "rest of receipt"],
    )
    result = convert_row(row)
    assert result is not None
    assert result["label"]["date"] == "2020-01-02"


def test_convert_row_no_items_and_no_recoverable_date_returns_none():
    row = _make_row(
        header={"invoice_date": "", "seller": "Acme Corp"},
        items=[],
        summary={"total_net_worth": "$0", "total_vat": "$0", "total_gross_worth": "$0"},
        ocr_words=["no date anywhere in here"],
    )
    assert convert_row(row) is None


def test_convert_row_missing_amount_falls_back_to_qty_times_price():
    row = _make_row(
        header={"invoice_date": "10/15/2012", "seller": "Acme Corp"},
        # simulates the upstream inconsistency where some items use
        # "total_net_worth" instead of "item_net_worth"
        items=[{"item_desc": "Widget", "item_qty": "3,00", "item_net_price": "2,00", "total_net_worth": "6,00"}],
        summary={"total_net_worth": "$6,00", "total_vat": "$0,60", "total_gross_worth": "$6,60"},
        ocr_words=["10/15/2012"],
    )
    result = convert_row(row)
    assert result is not None
    assert result["label"]["line_items"][0]["amount"] == 6.0


def test_convert_row_blank_vendor_returns_none():
    row = _make_row(
        header={"invoice_date": "10/15/2012", "seller": ""},
        items=[{"item_desc": "Widget", "item_qty": "1,00", "item_net_price": "5,00", "item_net_worth": "5,00"}],
        summary={"total_net_worth": "$5,00", "total_vat": "$0,50", "total_gross_worth": "$5,50"},
        ocr_words=["10/15/2012"],
    )
    assert convert_row(row) is None
