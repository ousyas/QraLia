"""Render a synthetic ONEE electricity invoice as PDF + ground-truth JSON."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import arabic_reshaper
from bidi.algorithm import get_display
from fpdf import FPDF

from qralia.domain.invoice import InvoiceParams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FONT_DIR = Path(__file__).parent / "assets" / "fonts"
_CAIRO_REGULAR = _FONT_DIR / "Cairo-Regular.ttf"
_CAIRO_BOLD = _FONT_DIR / "Cairo-Bold.ttf"  # optional — falls back to regular


def _ar(text: str) -> str:
    """Reshape + reorder Arabic text for correct RTL rendering in fpdf2."""
    configuration = {
        "delete_harakat": False,
        "support_ligatures": True,
        "RIAL SIGN": True,
        "use_unshaped_instead_of_isolated": True,
    }
    reshaper = arabic_reshaper.ArabicReshaper(configuration=configuration)
    reshaped = reshaper.reshape(text)
    return get_display(reshaped)


def _fmt_money(value: float) -> str:
    """Format a float as '12.345,67' (Moroccan/ONEE style with dot separator)."""
    integer, decimal = f"{value:,.2f}".rsplit(".", 1)
    integer = integer.replace(",", ".")
    return f"{integer},{decimal}"


def _fmt_date(d: date) -> str:
    return d.strftime("%d.%m.%Y")


# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------

def render_invoice(invoice: InvoiceParams, output_dir: Path) -> dict:
    """Produce  <invoice_number>.pdf  +  <invoice_number>.ground_truth.json."""

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = invoice.invoice_number

    pdf = FPDF(unit="mm", format="A4")

    # -- fonts ---------------------------------------------------------------
    pdf.add_font("Cairo", style="", fname=str(_CAIRO_REGULAR))
    if _CAIRO_BOLD.exists():
        pdf.add_font("Cairo", style="B", fname=str(_CAIRO_BOLD))
    else:
        pdf.add_font("Cairo", style="B", fname=str(_CAIRO_REGULAR))

    pdf.set_auto_page_break(auto=False)
    pdf.add_page()

    # ========================================================================
    # BLOC 1 — Header ONEE (left) + Destinataire (right)
    # ========================================================================
    pdf.set_font("Cairo", "B", 11)
    pdf.set_xy(10, 10)
    pdf.cell(95, 6, "Office National de l'Electricité et de l'Eau Potable", ln=True)

    pdf.set_font("Cairo", "", 9)
    pdf.set_x(10)
    pdf.cell(95, 5, "Votre agence :", ln=True)
    pdf.set_x(10)
    pdf.cell(95, 5, "Agence de Services", ln=True)
    pdf.set_x(10)
    pdf.cell(95, 5, "080 200 77 77", ln=True)
    pdf.set_x(10)
    pdf.cell(95, 5, "crc@onee.ma", ln=True)

    # Destinataire — right column
    pdf.set_font("Cairo", "B", 10)
    pdf.set_xy(120, 10)
    pdf.cell(80, 6, "Destinataire :", ln=True)

    pdf.set_font("Cairo", "", 9)
    pdf.set_xy(120, 16)
    pdf.cell(80, 5, invoice.client_name, ln=True)
    pdf.set_xy(120, 21)
    pdf.cell(80, 5, invoice.client_address, ln=True)

    # ========================================================================
    # BLOC 2 — Title
    # ========================================================================
    y_title = 48
    pdf.set_font("Cairo", "B", 14)
    pdf.set_xy(10, y_title)
    pdf.cell(95, 8, "Facture d'électricité", align="C")

    pdf.set_font("Cairo", "B", 14)
    pdf.set_xy(105, y_title)
    pdf.cell(95, 8, _ar("فاتورة الكهرباء"), align="C")

    # ========================================================================
    # BLOC 3 — Identifiers
    # ========================================================================
    y = 62
    labels_fr = [
        ("N° Facture :", invoice.invoice_number),
        ("N° Contrat :", invoice.contract_number),
        ("Date :", _fmt_date(invoice.invoice_date)),
        ("Type :", "Facture"),
        ("Tarif :", invoice.tariff),
    ]

    pdf.set_font("Cairo", "", 10)
    for label, value in labels_fr:
        pdf.set_xy(15, y)
        pdf.cell(45, 6, label)
        pdf.cell(50, 6, str(value))
        y += 7

    # ========================================================================
    # BLOC 4 — Amounts box
    # ========================================================================
    y_box = y + 5
    box_x, box_w = 12, 186
    box_h = 60

    pdf.set_draw_color(0, 51, 153)
    pdf.set_line_width(0.5)
    pdf.rect(box_x, y_box, box_w, box_h)

    row_y = y_box + 5

    def _amount_row(label_fr: str, value: str, bold: bool = False):
        nonlocal row_y
        style = "B" if bold else ""
        pdf.set_font("Cairo", style, 10 if not bold else 12)
        pdf.set_xy(box_x + 5, row_y)
        pdf.cell(70, 7, label_fr)
        pdf.set_xy(box_x + 75, row_y)
        pdf.cell(40, 7, value, align="R")
        row_y += 9

    _amount_row("Total HT :", _fmt_money(invoice.amount_before_tax))
    _amount_row("Taxes :", _fmt_money(invoice.taxes))
    _amount_row("Total à régler (DH TTC) :", _fmt_money(invoice.total_check), bold=True)
    _amount_row("Date d'échéance :", _fmt_date(invoice.due_date))

    # stamp duty line
    row_y += 2
    pdf.set_font("Cairo", "", 8)
    pdf.set_xy(box_x + 5, row_y)
    stamp_duty = invoice.total_cash - invoice.total_check
    pdf.cell(
        0, 5,
        f"En cas de règlement en espèce, le montant de droit de timbre est {_fmt_money(stamp_duty)} Dh",
    )

    # ========================================================================
    # BLOC 5 — Consumption details
    # ========================================================================
    y_cons = y_box + box_h + 10
    pdf.set_font("Cairo", "B", 10)
    pdf.set_xy(10, y_cons)
    pdf.cell(0, 6, "Détails de consommation", ln=True)

    pdf.set_font("Cairo", "", 9)
    details = [
        ("Période :", f"{_fmt_date(invoice.period_start)} - {_fmt_date(invoice.period_end)}"),
        ("Index précédent :", str(invoice.index_previous)),
        ("Index actuel :", str(invoice.index_current)),
        ("Consommation (kWh) :", str(invoice.consumption_kwh)),
        ("Tranche tarifaire :", str(invoice.tariff_tranche)),
    ]

    y_d = y_cons + 8
    for label, value in details:
        pdf.set_xy(15, y_d)
        pdf.cell(55, 5, label)
        pdf.cell(40, 5, value)
        y_d += 6

    # ========================================================================
    # BLOC 6 — Previous unpaid
    # ========================================================================
    if invoice.previous_unpaid > 0:
        y_d += 4
        pdf.set_font("Cairo", "", 9)
        pdf.set_xy(15, y_d)
        pdf.cell(55, 5, "Arriérés :")
        pdf.cell(40, 5, _fmt_money(invoice.previous_unpaid))

    # ========================================================================
    # BLOC 7 — Coupon (bottom)
    # ========================================================================
    y_coupon = 250
    pdf.set_draw_color(150, 150, 150)
    pdf.set_line_width(0.3)
    pdf.dashed_line(10, y_coupon, 200, y_coupon, dash_length=2, space_length=2)

    pdf.set_font("Cairo", "B", 9)
    pdf.set_xy(10, y_coupon + 3)
    pdf.cell(0, 5, f"Facture N° {invoice.invoice_number} du {_fmt_date(invoice.due_date)}", ln=True)

    pdf.set_xy(110, y_coupon + 3)
    pdf.cell(0, 5, f"Montant à régler avant le {_fmt_date(invoice.due_date)} :")

    pdf.set_font("Cairo", "B", 12)
    pdf.set_xy(110, y_coupon + 10)
    pdf.cell(0, 6, f"{_fmt_money(invoice.total_check)} DH TTC")

    pdf.set_font("Cairo", "", 9)
    pdf.set_xy(10, y_coupon + 14)
    pdf.cell(0, 5, invoice.client_name)

    # -- Footer ONEE ---------------------------------------------------------
    pdf.set_font("Cairo", "", 7)
    pdf.set_xy(10, 285)
    pdf.cell(
        0, 4,
        "Office National de l'Electricité et de l'Eau Potable - Rabat",
        align="C",
    )

    # ========================================================================
    # Save PDF
    # ========================================================================
    pdf_path = output_dir / f"{stem}.pdf"
    pdf.output(str(pdf_path))

    # ========================================================================
    # Ground-truth JSON (conforms to extracted_invoice.schema.json)
    # ========================================================================
    ground_truth = {
        "schema_version": "0.1.0",
        "invoice_id": invoice.invoice_number,
        "extraction": {
            "method": "synthetic_generation",
            "timestamp": date.today().isoformat(),
        },
        "fields": {
            "total_cash": {
                "raw_value": _fmt_money(invoice.total_cash),
                "normalized_value": invoice.total_cash,
                "confidence": 1.0,
                "provenance": {"page": 1, "bbox": None, "method": "generated"},
            },
            "total_check": {
                "raw_value": _fmt_money(invoice.total_check),
                "normalized_value": invoice.total_check,
                "confidence": 1.0,
                "provenance": {"page": 1, "bbox": None, "method": "generated"},
            },
            "due_date": {
                "raw_value": _fmt_date(invoice.due_date),
                "normalized_value": invoice.due_date.isoformat(),
                "confidence": 1.0,
                "provenance": {"page": 1, "bbox": None, "method": "generated"},
            },
            "invoice_date": {
                "raw_value": _fmt_date(invoice.invoice_date),
                "normalized_value": invoice.invoice_date.isoformat(),
                "confidence": 1.0,
                "provenance": {"page": 1, "bbox": None, "method": "generated"},
            },
            "tariff": {
                "raw_value": invoice.tariff,
                "normalized_value": invoice.tariff,
                "confidence": 1.0,
                "provenance": {"page": 1, "bbox": None, "method": "generated"},
            },
            "amount_before_tax": {
                "raw_value": _fmt_money(invoice.amount_before_tax),
                "normalized_value": invoice.amount_before_tax,
                "confidence": 1.0,
                "provenance": {"page": 1, "bbox": None, "method": "generated"},
            },
            "taxes": {
                "raw_value": _fmt_money(invoice.taxes),
                "normalized_value": invoice.taxes,
                "confidence": 1.0,
                "provenance": {"page": 1, "bbox": None, "method": "generated"},
            },
            "previous_unpaid": {
                "raw_value": _fmt_money(invoice.previous_unpaid),
                "normalized_value": invoice.previous_unpaid,
                "confidence": 1.0,
                "provenance": {"page": 1, "bbox": None, "method": "generated"},
            },
        },
        "document_confidence": 1.0,
    }

    gt_path = output_dir / f"{stem}.ground_truth.json"
    gt_path.write_text(json.dumps(ground_truth, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"pdf_path": str(pdf_path), "ground_truth_path": str(gt_path)}


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from qralia.generation.factory import random_invoice_params

    out = Path("data/generated")
    params = random_invoice_params()
    result = render_invoice(params, out)
    print(f"PDF          → {result['pdf_path']}")
    print(f"Ground truth → {result['ground_truth_path']}")