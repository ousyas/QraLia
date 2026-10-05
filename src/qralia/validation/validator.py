"""Deterministic validation of extracted invoice fields."""

STAMP_DUTY = 0.33
VAT_RATE = 0.14
TOLERANCE = 0.01


def _get(extracted: dict, field: str):
    """Get a field's normalized_value from the extraction dict."""
    return extracted["fields"][field]["normalized_value"]


def validate_invoice(extracted: dict) -> dict:
    """Check arithmetic and date coherence of extracted fields.

    Returns {"valid": bool, "errors": [...], "warnings": [...]}.
    """
    errors = []
    warnings = []

    total_check = _get(extracted, "total_check")
    total_cash = _get(extracted, "total_cash")
    amount_ht = _get(extracted, "amount_before_tax")
    taxes = _get(extracted, "taxes")
    previous = _get(extracted, "previous_unpaid")
    due_date = _get(extracted, "due_date")
    inv_date = _get(extracted, "invoice_date")

    # Rule 1 — total_check == amount_before_tax + taxes + previous_unpaid
    expected_check = amount_ht + taxes + previous
    if abs(total_check - expected_check) > TOLERANCE:
        errors.append(
            f"total_check ({total_check}) != amount_before_tax + taxes + previous_unpaid ({expected_check:.2f})"
        )

    # Rule 2 — total_cash == total_check + stamp_duty
    expected_cash = total_check + STAMP_DUTY
    if abs(total_cash - expected_cash) > TOLERANCE:
        errors.append(
            f"total_cash ({total_cash}) != total_check + timbre ({expected_cash:.2f})"
        )

    # Rule 3 — taxes ≈ amount_before_tax × VAT_RATE (warning, not error — rate may vary)
    expected_taxes = round(amount_ht * VAT_RATE, 2)
    if abs(taxes - expected_taxes) > 1.0:
        warnings.append(
            f"taxes ({taxes}) differ from expected {VAT_RATE*100:.0f}% of HT ({expected_taxes:.2f})"
        )

    # Rule 4 — due_date > invoice_date
    if due_date and inv_date and due_date <= inv_date:
        errors.append(
            f"due_date ({due_date}) <= invoice_date ({inv_date})"
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    from pathlib import Path
    from qralia.ocr.extractor import extract_fields

    pdf = Path("data/generated/F08653065.pdf")    
    extracted = extract_fields(pdf)
    result = validate_invoice(extracted)
    print(json.dumps(result, indent=2, ensure_ascii=False))