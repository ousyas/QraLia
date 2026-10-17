"""Contrats OCR testés sans dépendre du binaire Tesseract."""

from qralia.domain.extracted_invoice import FieldStatus
from qralia.domain.invoice_evidence import build_invoice_evidence
from qralia.ocr.extractor import OcrToken, _parse_money, parse_ocr_tokens
from qralia.validation.validator import validate_invoice


def _tokens(*texts: str) -> list[OcrToken]:
    return [OcrToken(text, 0.95, 1, (10 + index * 10, 20, 18 + index * 10, 32)) for index, text in enumerate(texts)]


def _realistic_tokens() -> list[OcrToken]:
    return _tokens(
        "tarif", "0,72", "DH/HT", "par", "minute",
        "N°", "Contrat", ":", "1234567", "N°", "Facture", ":", "F00000001",
        "Date", ":", "07.02.2024", "Type", ":", "Facture", "Tarif", "MT", "Général",
        "Total", "HT", ":", "20.537,49", "Taxes", ":", "2.881,71",
        "Total", "à", "régler", "(DH", "TTC)", ":", "23.419,20",
        "Date", "d'échéance", "du", "règlement", ":", "01.03.2024",
        "droit", "de", "timbre", "est", "58,55", "Dh",
    )


def test_moroccan_money_formats():
    assert _parse_money("23.419,20") == 23419.20
    assert _parse_money("23 419,20") == 23419.20
    assert _parse_money("930.50") == 930.50
    assert _parse_money("01.03.2017") is None


def test_parse_validate_and_build_evidence():
    invoice = parse_ocr_tokens(_realistic_tokens(), "synthetic.pdf")
    assert invoice.invoice_id == "F00000001"
    assert invoice.fields.total_check.normalized_value == 23419.20
    assert invoice.fields.total_cash.normalized_value == 23477.75
    assert invoice.fields.due_date.normalized_value == "2024-03-01"
    assert invoice.fields.tariff.normalized_value == "MT_GENERAL"
    assert invoice.fields.previous_unpaid.status == FieldStatus.NOT_PRESENT
    validation = validate_invoice(invoice)
    assert validation["valid"] is True
    assert validation["scope_supported"] is False
    assert validation["critical_fields_complete"] is True
    evidence = build_invoice_evidence(invoice)
    assert {"total_check", "due_date", "invoice_number"} <= {item.field for item in evidence}


def test_missing_stamp_never_creates_a_default_amount():
    tokens = _realistic_tokens()
    tokens = tokens[: next(index for index, token in enumerate(tokens) if token.text == "droit")]
    invoice = parse_ocr_tokens(tokens, "synthetic.pdf")
    assert invoice.fields.total_cash.status == FieldStatus.UNREADABLE
    assert invoice.fields.total_cash.normalized_value is None


def test_date_between_total_label_and_amount_is_not_parsed_as_money():
    tokens = _tokens("Total", "à", "régler", "Date", "01.03.2017", "23.419,20")
    invoice = parse_ocr_tokens(tokens, "synthetic.pdf")
    assert invoice.fields.total_check.normalized_value == 23419.20


def test_split_thousands_prefix_is_reassembled():
    tokens = _tokens("Total", "à", "régler", "23.", "419,20")
    invoice = parse_ocr_tokens(tokens, "synthetic.pdf")
    assert invoice.fields.total_check.normalized_value == 23419.20


def test_split_decimal_suffix_is_reassembled():
    tokens = _tokens("Total", "HT", "20537,", "49")
    invoice = parse_ocr_tokens(tokens, "synthetic.pdf")
    assert invoice.fields.amount_before_tax.normalized_value == 20537.49


def test_short_residential_tariff_does_not_absorb_next_label():
    invoice = parse_ocr_tokens(
        _tokens("Tarif", "BT", "Periode", "Septembre", "2026"),
        "synthetic.pdf",
    )
    assert invoice.fields.tariff.normalized_value == "BT"

