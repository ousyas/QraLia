"""Extract invoice fields from a PDF using EasyOCR."""

import re
from pathlib import Path

import easyocr
import numpy as np
from pdf2image import convert_from_path


# Initialize reader once (model loading is slow)
_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["ar", "en"], gpu=False)
    return _reader


def _ocr_pdf(pdf_path: Path) -> list[tuple[str, float]]:
    """Convert PDF to image, run OCR, return [(text, confidence), ...]."""
    images = convert_from_path(pdf_path)
    reader = _get_reader()
    results = []
    for image in images:
        detections = reader.readtext(np.array(image))
        for _bbox, text, conf in detections:
            results.append((text.strip(), conf))
    return results


def _find_value_after(results, keyword, start=0):
    """Find keyword in results, return the next item's text and confidence."""
    for i in range(start, len(results)):
        text, _conf = results[i]
        if keyword.lower() in text.lower():
            if i + 1 < len(results):
                return i + 1, results[i + 1]  # index, (text, conf)
    return None, None


def _collect_money_after(results, keyword):
    """Find a money value after a keyword, merging split thousands digits.

    OCR splits '1 034,48' into ['1', '034,48'] because of the space.
    This function detects that pattern and merges them.
    """
    idx, val = _find_value_after(results, keyword)
    if val is None:
        return 0.0, 0.0

    val_text, val_conf = val

    # If the value looks like a partial number (starts with 0 or is 2-3 digits + comma)
    # check the PREVIOUS item — it might be the leading digit(s)
    if idx >= 2:
        prev_text, prev_conf = results[idx - 1]
        # Previous item is isolated digits AND not the keyword itself
        if (re.fullmatch(r"\d{1,2}", prev_text)
                and keyword.lower() not in prev_text.lower()):
            # Check if merging makes a valid number
            merged = prev_text + val_text
            if re.search(r"\d+[.,]\d{2}", merged):
                return _parse_money(merged), min(val_conf, prev_conf)

    # Also check if the value at idx is just isolated digits and the NEXT item
    # has the decimal part
    if re.fullmatch(r"\d{1,3}", val_text) and idx + 1 < len(results):
        next_text, next_conf = results[idx + 1]
        if re.search(r"\d+[.,]\d{2}", next_text):
            merged = val_text + next_text
            return _parse_money(merged), min(val_conf, next_conf)

    return _parse_money(val_text), val_conf


def _extract_stamp_duty(results):
    """Extract stamp duty amount from the 'timbre' line."""
    for text, conf in results:
        if "timbre" in text.lower():
            match = re.search(r"(\d+[.,]\d{2})", text)
            if match:
                return float(match.group(1).replace(",", ".")), conf
    return 0.33, 0.0  # fallback


def _parse_money(text: str) -> float:
    """Convert '883,50' or '1.034,48' or '23.419,20' to float."""
    cleaned = text.replace(" ", "")
    # Remove dot thousands separators: '1.034,48' → '1034,48'
    # A dot followed by exactly 3 digits and then a comma or end is a thousands sep
    cleaned = re.sub(r"\.(\d{3})", r"\1", cleaned)
    # Now swap comma for decimal point: '1034,48' → '1034.48'
    cleaned = cleaned.replace(",", ".")
    match = re.search(r"(\d+\.?\d*)", cleaned)
    if match:
        return float(match.group(1))
    return 0.0


def _parse_date(text: str) -> str:
    """Convert '24.05.2026' to ISO '2026-05-24'."""
    match = re.search(r"(\d{2})[./](\d{2})[./](\d{4})", text)
    if match:
        d, m, y = match.groups()
        return f"{y}-{m}-{d}"
    return text


def _extract_previous_unpaid(results):
    """Extract previous unpaid amount (Arriérés) if present."""
    # Look for "Arri" or "rri" — OCR may mangle accents
    for i, (text, conf) in enumerate(results):
        if "rri" in text.lower() or "arri" in text.lower():
            if i + 1 < len(results):
                val_text, val_conf = results[i + 1]
                amount = _parse_money(val_text)
                if amount > 0:
                    return amount, val_conf
    return 0.0, 0.0


def extract_fields(pdf_path: Path) -> dict:
    """Extract the 8 ground-truth fields from an ONEE invoice PDF.

    Returns a dict matching the extracted_invoice.schema.json format.
    """
    results = _ocr_pdf(pdf_path)

    # --- amount_before_tax (after "Total HT")
    amount_before_tax, ht_conf = _collect_money_after(results, "Total HT")

    # --- taxes (after "Taxes")
    taxes, tx_conf = _collect_money_after(results, "Taxes")

    # --- total_check (after "regler")
    total_check, tc_conf = _collect_money_after(results, "regler")

    # --- total_cash = total_check + stamp duty
    stamp, stamp_conf = _extract_stamp_duty(results)
    total_cash = round(total_check + stamp, 2)

    # --- previous_unpaid (after "Arriérés")
    previous_unpaid, pu_conf = _extract_previous_unpaid(results)

    # --- invoice_date (first "Date" that is NOT "echeance")
    inv_date_text, inv_date_conf = "", 0.0
    for i, (text, conf) in enumerate(results):
        if text.lower().strip() in ("date", "date :"):
            if i + 1 < len(results):
                nxt = results[i + 1][0]
                if re.search(r"\d{2}[./]\d{2}[./]\d{4}", nxt):
                    inv_date_text = nxt
                    inv_date_conf = results[i + 1][1]
                    break

    # --- due_date (after "echeance")
    _, dd = _find_value_after(results, "cheance")
    due_date_text = dd[0] if dd else ""
    dd_conf = dd[1] if dd else 0.0

    # --- tariff (after "Tarif")
    _, tf = _find_value_after(results, "Tarif")
    tariff_text = tf[0] if tf else ""
    tf_conf = tf[1] if tf else 0.0
    # EasyOCR may split "MT" and "GENERAL" — merge them
    if tariff_text in ("MT", "BT") and tf:
        idx = next(
            (i for i, (t, _) in enumerate(results) if t == tariff_text),
            None,
        )
        if idx is not None and idx + 1 < len(results):
            tariff_text = f"{tariff_text}_{results[idx + 1][0]}"

    def _field(raw, normalized, conf, method="easyocr"):
        return {
            "raw_value": raw,
            "normalized_value": normalized,
            "confidence": round(conf, 4),
            "provenance": {"page": 1, "bbox": None, "method": method},
        }

    fmt = lambda v: f"{v:.2f}".replace(".", ",")

    return {
        "schema_version": "0.1.0",
        "invoice_id": "",
        "extraction": {"method": "easyocr", "timestamp": None},
        "fields": {
            "total_cash": _field(fmt(total_cash), total_cash, tc_conf),
            "total_check": _field(fmt(total_check), total_check, tc_conf),
            "due_date": _field(
                due_date_text, _parse_date(due_date_text), dd_conf
            ),
            "invoice_date": _field(
                inv_date_text, _parse_date(inv_date_text), inv_date_conf
            ),
            "tariff": _field(tariff_text, tariff_text, tf_conf),
            "amount_before_tax": _field(
                fmt(amount_before_tax), amount_before_tax, ht_conf
            ),
            "taxes": _field(fmt(taxes), taxes, tx_conf),
            "previous_unpaid": _field(
                fmt(previous_unpaid), previous_unpaid, pu_conf
            ),
        },
        "document_confidence": round(
            min(ht_conf, tx_conf, tc_conf, dd_conf, inv_date_conf, tf_conf), 4
        ),
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    path = Path("data/generated/F02916547.pdf")
    result = extract_fields(path)
    print(json.dumps(result, indent=2, ensure_ascii=False))