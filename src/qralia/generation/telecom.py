"""Factures d'abonnement télécom synthétiques (Maroc Telecom, Orange, inwi).

Chaque opérateur a ses propres libellés et sa mise en page, afin que l'OCR soit
évalué sur sa capacité à généraliser et non sur un gabarit unique. Les
documents ne portent aucun logo et sont marqués « SPÉCIMEN ».
"""

from __future__ import annotations

import argparse
import json
import random
import string
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from fpdf import FPDF

from qralia.generation.renderer import _CAIRO_BOLD, _CAIRO_REGULAR, _fmt_date, _fmt_money

VAT_RATE = 0.20  # TVA des services de télécommunications au Maroc


@dataclass(frozen=True)
class OperatorStyle:
    provider_id: str
    name: str
    accent: tuple[int, int, int]
    invoice_prefix: str
    labels: dict[str, str]
    plans: tuple[tuple[str, float], ...]


OPERATORS = {
    "iam": OperatorStyle(
        "iam", "Maroc Telecom", (0, 86, 150), "M",
        {
            "invoice_number": "N° de facture", "contract_number": "N° client", "line_number": "N° de ligne",
            "invoice_date": "Date de facture", "period": "Période de facturation", "plan_name": "Offre",
            "subscription": "Abonnement mensuel HT", "out_of_plan_amount": "Consommation hors forfait HT",
            "amount_before_tax": "Total HT", "taxes": "TVA 20 %", "previous_unpaid": "Solde précédent",
            "total_check": "Montant à payer", "due_date": "Date limite de paiement",
        },
        (("Forfait Mobile 10 Go", 49.0), ("Forfait Mobile 30 Go", 99.0), ("Forfait Illimité Appels", 149.0)),
    ),
    "orange_ma": OperatorStyle(
        "orange_ma", "Orange Maroc", (255, 121, 0), "OR",
        {
            "invoice_number": "Référence facture", "contract_number": "Code client", "line_number": "Numéro de ligne",
            "invoice_date": "Date d'émission", "period": "Période", "plan_name": "Forfait",
            "subscription": "Abonnement HT", "out_of_plan_amount": "Hors forfait HT",
            "amount_before_tax": "Total HT", "taxes": "TVA (20%)", "previous_unpaid": "Report du solde",
            "total_check": "Total à payer", "due_date": "À payer avant le",
        },
        (("Forfait 20 Go", 79.0), ("Forfait 50 Go", 149.0), ("Box Fibre 100 Méga", 349.0)),
    ),
    "inwi": OperatorStyle(
        "inwi", "inwi", (128, 0, 128), "IN",
        {
            "invoice_number": "N° facture", "contract_number": "N° de compte client", "line_number": "Ligne",
            "invoice_date": "Date de facture", "period": "Période de consommation", "plan_name": "Offre souscrite",
            "subscription": "Montant de l'abonnement HT", "out_of_plan_amount": "Consommation hors forfait HT",
            "amount_before_tax": "Total HT", "taxes": "TVA 20%", "previous_unpaid": "Solde antérieur",
            "total_check": "Net à payer", "due_date": "Date limite de paiement",
        },
        (("Forfait Mobile 15 Go", 59.0), ("Forfait Mobile 40 Go", 119.0), ("Box Internet 4G", 249.0)),
    ),
}


@dataclass(frozen=True)
class TelecomInvoice:
    provider_id: str
    invoice_number: str
    contract_number: str
    line_number: str
    client_name: str
    client_address: str
    invoice_date: date
    due_date: date
    period_start: date
    period_end: date
    plan_name: str
    subscription: float
    out_of_plan_amount: float
    amount_before_tax: float
    taxes: float
    previous_unpaid: float
    total_check: float


_FIRST_NAMES = ("Mohamed", "Fatima", "Youssef", "Khadija", "Omar", "Salma", "Hassan", "Nadia", "Imane", "Karim")
_LAST_NAMES = ("Alaoui", "Bennani", "Idrissi", "Tazi", "Berrada", "Chraibi", "Lahlou", "Ziani", "Rifi", "Kabbaj")
_CITIES = ("Rabat", "Salé", "Casablanca", "Fès", "Tanger", "Agadir")


def random_telecom_invoice(rng: random.Random, provider_id: str, reference: date) -> TelecomInvoice:
    style = OPERATORS[provider_id]
    period_start = (reference - timedelta(days=rng.randint(0, 330))).replace(day=1)
    period_end = (period_start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
    invoice_date = period_end + timedelta(days=rng.randint(2, 6))
    due_date = invoice_date + timedelta(days=rng.randint(15, 25))

    plan_name, subscription = rng.choice(style.plans)
    out_of_plan = round(rng.uniform(3, 120), 2) if rng.random() < 0.35 else 0.0
    amount_before_tax = round(subscription + out_of_plan, 2)
    taxes = round(amount_before_tax * VAT_RATE, 2)
    previous_unpaid = round(rng.uniform(20, 300), 2) if rng.random() < 0.2 else 0.0
    total = round(amount_before_tax + taxes + previous_unpaid, 2)

    # Les numéros de ligne sont masqués pour ne jamais désigner une vraie personne.
    line_number = f"0{rng.choice('67')} ** ** {rng.randint(10, 99)} {rng.randint(10, 99)}"
    return TelecomInvoice(
        provider_id=provider_id,
        invoice_number=style.invoice_prefix + "".join(rng.choices(string.digits, k=9)),
        contract_number="".join(rng.choices(string.digits, k=9)),
        line_number=line_number,
        client_name=f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}",
        client_address=f"Rue {rng.randint(1, 99)}, {rng.choice(_CITIES)}",
        invoice_date=invoice_date, due_date=due_date, period_start=period_start, period_end=period_end,
        plan_name=plan_name, subscription=subscription, out_of_plan_amount=out_of_plan,
        amount_before_tax=amount_before_tax, taxes=taxes, previous_unpaid=previous_unpaid, total_check=total,
    )


def _ground_field(raw: str, value: object) -> dict:
    return {"raw_value": raw, "normalized_value": value, "confidence": 1.0,
            "provenance": {"page": 1, "bbox": None, "method": "generated"}}


def ground_truth(invoice: TelecomInvoice) -> dict:
    return {
        "schema_version": "0.2.0",
        "invoice_id": invoice.invoice_number,
        "provider_id": invoice.provider_id,
        "extraction": {"method": "synthetic_generation", "timestamp": date.today().isoformat()},
        "fields": {
            "invoice_number": _ground_field(invoice.invoice_number, invoice.invoice_number),
            "contract_number": _ground_field(invoice.contract_number, invoice.contract_number),
            "line_number": _ground_field(invoice.line_number, invoice.line_number.replace(" ", "")),
            "plan_name": _ground_field(invoice.plan_name, invoice.plan_name),
            "invoice_date": _ground_field(_fmt_date(invoice.invoice_date), invoice.invoice_date.isoformat()),
            "due_date": _ground_field(_fmt_date(invoice.due_date), invoice.due_date.isoformat()),
            "out_of_plan_amount": _ground_field(_fmt_money(invoice.out_of_plan_amount), invoice.out_of_plan_amount),
            "amount_before_tax": _ground_field(_fmt_money(invoice.amount_before_tax), invoice.amount_before_tax),
            "taxes": _ground_field(_fmt_money(invoice.taxes), invoice.taxes),
            "previous_unpaid": _ground_field(_fmt_money(invoice.previous_unpaid), invoice.previous_unpaid),
            "total_check": _ground_field(_fmt_money(invoice.total_check), invoice.total_check),
        },
        "document_confidence": 1.0,
    }


def render_telecom_invoice(invoice: TelecomInvoice, output_dir: Path) -> dict:
    style = OPERATORS[invoice.provider_id]
    labels = style.labels
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf = FPDF(unit="mm", format="A4")
    pdf.add_font("Cairo", style="", fname=str(_CAIRO_REGULAR))
    pdf.add_font("Cairo", style="B", fname=str(_CAIRO_BOLD if _CAIRO_BOLD.exists() else _CAIRO_REGULAR))
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()

    # Bandeau opérateur (texte seulement, aucun logo) et mention de spécimen.
    pdf.set_fill_color(*style.accent)
    pdf.rect(0, 0, 210, 26, style="F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Cairo", "B", 18)
    pdf.set_xy(12, 7)
    pdf.cell(120, 10, style.name)
    pdf.set_font("Cairo", "", 8)
    pdf.set_xy(130, 9)
    pdf.cell(70, 6, "SPÉCIMEN — facture synthétique", align="R")
    pdf.set_text_color(30, 30, 30)

    pdf.set_font("Cairo", "B", 13)
    pdf.set_xy(12, 34)
    pdf.cell(100, 8, "Facture d'abonnement")
    pdf.set_font("Cairo", "", 9)
    pdf.set_xy(130, 34)
    pdf.multi_cell(70, 5, f"{invoice.client_name}\n{invoice.client_address}", align="R")

    y = 50
    pdf.set_font("Cairo", "", 10)
    for key, value in (
        ("invoice_number", invoice.invoice_number),
        ("contract_number", invoice.contract_number),
        ("line_number", invoice.line_number),
        ("invoice_date", _fmt_date(invoice.invoice_date)),
        ("period", f"du {_fmt_date(invoice.period_start)} au {_fmt_date(invoice.period_end)}"),
        ("plan_name", invoice.plan_name),
    ):
        pdf.set_xy(14, y)
        pdf.cell(62, 6, f"{labels[key]} :")
        pdf.cell(100, 6, value)
        y += 7

    # Détail des montants.
    y += 6
    pdf.set_draw_color(*style.accent)
    pdf.set_line_width(0.4)
    pdf.line(12, y, 198, y)
    y += 4
    rows = [
        ("subscription", invoice.subscription),
        ("out_of_plan_amount", invoice.out_of_plan_amount),
        ("amount_before_tax", invoice.amount_before_tax),
        ("taxes", invoice.taxes),
    ]
    if invoice.previous_unpaid > 0:
        rows.append(("previous_unpaid", invoice.previous_unpaid))
    for key, value in rows:
        pdf.set_font("Cairo", "B" if key == "amount_before_tax" else "", 10)
        pdf.set_xy(14, y)
        pdf.cell(110, 7, f"{labels[key]} :")
        pdf.cell(40, 7, _fmt_money(value), align="R")
        y += 8
    pdf.line(12, y + 1, 198, y + 1)

    # Encadré à payer.
    y += 8
    pdf.set_fill_color(245, 245, 245)
    pdf.rect(12, y, 186, 26, style="F")
    pdf.set_font("Cairo", "B", 13)
    pdf.set_xy(16, y + 3)
    pdf.cell(108, 9, f"{labels['total_check']} (DH TTC) :")
    pdf.cell(40, 9, _fmt_money(invoice.total_check), align="R")
    pdf.set_font("Cairo", "", 10)
    pdf.set_xy(16, y + 14)
    pdf.cell(108, 7, f"{labels['due_date']} :")
    pdf.cell(40, 7, _fmt_date(invoice.due_date), align="R")

    pdf.set_font("Cairo", "", 7)
    pdf.set_text_color(120, 120, 120)
    pdf.set_xy(10, 285)
    pdf.cell(190, 4, f"Document fictif généré pour les tests QraLia — sans valeur contractuelle — {style.name}", align="C")

    stem = invoice.invoice_number
    pdf_path = output_dir / f"{stem}.pdf"
    pdf.output(str(pdf_path))
    gt_path = output_dir / f"{stem}.ground_truth.json"
    gt_path.write_text(json.dumps(ground_truth(invoice), ensure_ascii=False, indent=2), encoding="utf-8")
    return {"pdf_path": str(pdf_path), "ground_truth_path": str(gt_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Génère des factures télécom synthétiques.")
    parser.add_argument("--per-operator", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("data/generated/telecom"))
    args = parser.parse_args()
    rng = random.Random(args.seed)
    reference = date(2026, 11, 1)
    for provider_id in OPERATORS:
        for _ in range(args.per_operator):
            result = render_telecom_invoice(random_telecom_invoice(rng, provider_id, reference), args.output)
            print(result["pdf_path"])


if __name__ == "__main__":
    main()
