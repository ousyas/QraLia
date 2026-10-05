"""Extract invoice fields from a PDF using Tesseract."""

import re
import unicodedata
from pathlib import Path

from PIL import Image, ImageOps
from pdf2image import convert_from_path

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _normalize(text: str) -> str:
    """Remove accents for tolerant matching."""
    return unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("ascii").lower()


def _ocr_file(file_path: Path) -> list[tuple[str, float]]:
    """Run OCR on a PDF or image file, return [(text, confidence), ...]."""
    import pytesseract

    if file_path.suffix.lower() in _IMAGE_EXTENSIONS:
        img = Image.open(file_path)
        img = ImageOps.exif_transpose(img)

        best_results = []
        best_conf = 0.0

        for angle in [0, 90, 180, 270]:
            rotated = img.rotate(angle, expand=True) if angle else img
            data = pytesseract.image_to_data(rotated, lang='fra', config='--psm 6', output_type=pytesseract.Output.DICT)
            pairs = []
            for i in range(len(data["text"])):
                if int(data["conf"][i]) >= 0 and data["text"][i].strip():
                    pairs.append((data["text"][i].strip(), float(data["conf"][i]) / 100))
            if pairs:
                avg = sum(c for _, c in pairs) / len(pairs)
                if avg > best_conf:
                    best_conf = avg
                    best_results = pairs
        return best_results
    else:
        images = convert_from_path(file_path)
        results = []
        for image in images:
            data = pytesseract.image_to_data(image, lang='fra', config='--psm 6', output_type=pytesseract.Output.DICT)
            for i in range(len(data["text"])):
                if int(data["conf"][i]) >= 0 and data["text"][i].strip():
                    results.append((data["text"][i].strip(), float(data["conf"][i]) / 100))
        return results


def _find_value_after(results, keyword, start=0):
    """Find keyword in results, return the next non-punctuation item."""
    kw = _normalize(keyword)
    for i in range(start, len(results)):
        text, _conf = results[i]
        if kw in _normalize(text):
            for j in range(i + 1, min(i + 4, len(results))):
                nxt_text, nxt_conf = results[j]
                if len(nxt_text) > 1 or nxt_text.isdigit():
                    return j, (nxt_text, nxt_conf)
            if i + 1 < len(results):
                return i + 1, results[i + 1]
    return None, None


def _collect_money_after(results, keyword):
    """Find a money value after a keyword, scanning forward for a number."""
    kw = _normalize(keyword)
    for i in range(len(results)):
        if kw in _normalize(results[i][0]):
            for j in range(i + 1, min(i + 8, len(results))):
                txt, conf = results[j]
                if re.search(r"\d", txt):
                    if j >= 1:
                        prev_txt, prev_conf = results[j - 1]
                        if re.fullmatch(r"\d{1,2}", prev_txt) and kw not in _normalize(prev_txt):
                            merged = prev_txt + txt
                            if re.search(r"\d+[.,]\d{2}", merged):
                                return _parse_money(merged), min(conf, prev_conf)
                    return _parse_money(txt), conf
            break
    return 0.0, 0.0


def _extract_stamp_duty(results):
    """Extract stamp duty amount from the 'timbre' line."""
    for i, (text, conf) in enumerate(results):
        if "timbre" in text.lower():
            for j in range(i + 1, min(i + 5, len(results))):
                match = re.search(r"(\d+[.,]\d{2})", results[j][0])
                if match:
                    return float(match.group(1).replace(",", ".")), results[j][1]
    return 0.33, 0.0


def _parse_money(text: str) -> float:
    """Convert '883,50' or '1.034,48' or '23.419,20' to float."""
    cleaned = text.replace(" ", "")
    cleaned = re.sub(r"\.(\d{3})", r"\1", cleaned)
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
    for i, (text, conf) in enumerate(results):
        if "rri" in text.lower() or "arri" in text.lower():
            if i + 1 < len(results):
                val_text, val_conf = results[i + 1]
                amount = _parse_money(val_text)
                if amount > 0:
                    return amount, val_conf
    return 0.0, 0.0


def extract_fields(pdf_path: Path) -> dict:
    """Extract the 8 ground-truth fields from an ONEE invoice PDF."""
    results = _ocr_file(pdf_path)

    # --- amount_before_tax (after "Total HT")
    amount_before_tax, ht_conf = _collect_money_after(results, "Total")

    # --- taxes (after "Taxes")
    taxes, tx_conf = 0.0, 0.0
    for keyword in ["Taxes", "axes", "oxes", "Toxes"]:
        taxes, tx_conf = _collect_money_after(results, keyword)
        if taxes > 0:
            break

    # --- total_check (after "regler")
    total_check, tc_conf = _collect_money_after(results, "regler")

    # --- total_cash = total_check + stamp duty
    stamp, stamp_conf = _extract_stamp_duty(results)
    total_cash = round(total_check + stamp, 2)

    # --- previous_unpaid (after "Arriérés")
    previous_unpaid, pu_conf = _extract_previous_unpaid(results)

    # --- invoice_date
    inv_date_text, inv_date_conf = "", 0.0
    for i, (text, conf) in enumerate(results):
        if "date" in _normalize(text) and "che" not in _normalize(text) and "regle" not in _normalize(text):
            for j in range(i + 1, min(i + 3, len(results))):
                nxt = results[j][0]
                if re.search(r"\d{2}[.,/]\d{2}[.,/]\d{4}", nxt):
                    inv_date_text = nxt.replace(",", ".")
                    inv_date_conf = results[j][1]
                    break
            if inv_date_text:
                break

    # --- due_date (after "echeance" / "reglement")
    due_date_text, dd_conf = "", 0.0
    for keyword in ["echeance", "cheance", "reglement"]:
        for i, (text, conf) in enumerate(results):
            if keyword in _normalize(text):
                for j in range(i + 1, min(i + 5, len(results))):
                    nxt = results[j][0].replace(",", ".")
                    if re.search(r"\d{2}[.]\d{2}[.]\d{4}", nxt):
                        due_date_text = nxt
                        dd_conf = results[j][1]
                        break
                if due_date_text:
                    break
        if due_date_text:
            break

    # --- tariff (after "Tarif")
       # --- tariff (after standalone "Tarif")
    tariff_text, tf_conf = "", 0.0
    for i, (text, conf) in enumerate(results):
        if _normalize(text) == "tarif":  # exact match, not "tarif:0,72"
            # Merge next words until we hit a number or unrelated label
            parts = []
            for j in range(i + 1, min(i + 4, len(results))):
                nxt = results[j][0]
                if re.search(r"\d", nxt) or _normalize(nxt) in ("total", "date"):
                    break
                if len(nxt) > 1:
                    parts.append(nxt)
            if parts:
                tariff_text = "_".join(parts)
                tf_conf = conf
            break

    def _field(raw, normalized, conf, method="tesseract"):
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
        "extraction": {"method": "tesseract", "timestamp": None},
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

    path = Path("data/real/facture_2.png")
    result = extract_fields(path)
    print(json.dumps(result, indent=2, ensure_ascii=False))