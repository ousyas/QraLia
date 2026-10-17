"""Crée une facture ONEE résidentielle fictive pour les tests locaux QraLia."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/pdf/qralia_facture_onee_residentielle_synthetique.pdf"


def draw_label_value(pdf: canvas.Canvas, y: float, label: str, value: str) -> None:
    pdf.setFont("Helvetica", 10)
    pdf.setFillColor(colors.HexColor("#425466"))
    pdf.drawString(30 * mm, y, label)
    pdf.setFont("Helvetica-Bold", 11)
    pdf.setFillColor(colors.HexColor("#12263A"))
    pdf.drawString(105 * mm, y, value)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(OUTPUT), pagesize=A4)
    width, height = A4

    pdf.setFillColor(colors.HexColor("#0C6B68"))
    pdf.rect(0, height - 48 * mm, width, 48 * mm, fill=1, stroke=0)
    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 24)
    pdf.drawString(22 * mm, height - 24 * mm, "ONEE - BRANCHE ELECTRICITE")
    pdf.setFont("Helvetica", 11)
    pdf.drawString(22 * mm, height - 34 * mm, "FACTURE RESIDENTIELLE - DOCUMENT DE TEST")

    pdf.setStrokeColor(colors.HexColor("#D9E2E8"))
    pdf.setFillColor(colors.HexColor("#F7FAFC"))
    pdf.roundRect(20 * mm, height - 137 * mm, 170 * mm, 77 * mm, 4 * mm, fill=1, stroke=1)

    y = height - 73 * mm
    draw_label_value(pdf, y, "N Facture", "20261017001")
    draw_label_value(pdf, y - 10 * mm, "Date", "17.10.2026")
    draw_label_value(pdf, y - 20 * mm, "N Contrat", "1234567")
    draw_label_value(pdf, y - 30 * mm, "Tarif", "BT")
    draw_label_value(pdf, y - 40 * mm, "Periode", "Septembre 2026")
    draw_label_value(pdf, y - 50 * mm, "Consommation", "312 kWh")

    pdf.setFillColor(colors.HexColor("#12263A"))
    pdf.setFont("Helvetica-Bold", 15)
    pdf.drawString(22 * mm, height - 158 * mm, "RECAPITULATIF")
    pdf.setStrokeColor(colors.HexColor("#D9E2E8"))
    pdf.line(22 * mm, height - 163 * mm, 188 * mm, height - 163 * mm)

    y = height - 176 * mm
    draw_label_value(pdf, y, "Total HT", "365,58 DH")
    draw_label_value(pdf, y - 11 * mm, "Total TVA", "73,12 DH")
    draw_label_value(pdf, y - 22 * mm, "Droit de timbre", "0,55 DH")

    pdf.setFillColor(colors.HexColor("#E9F5F3"))
    pdf.roundRect(20 * mm, height - 231 * mm, 170 * mm, 26 * mm, 4 * mm, fill=1, stroke=0)
    draw_label_value(pdf, height - 217 * mm, "Total a regler", "438,70 DH")
    draw_label_value(pdf, height - 227 * mm, "Date d echeance du reglement", "17.11.2026")

    pdf.setFillColor(colors.HexColor("#9B2C2C"))
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawCentredString(width / 2, 34 * mm, "DOCUMENT ENTIEREMENT SYNTHETIQUE - AUCUNE DONNEE PERSONNELLE REELLE")
    pdf.setFillColor(colors.HexColor("#61758A"))
    pdf.setFont("Helvetica", 8)
    pdf.drawCentredString(width / 2, 25 * mm, "Cree uniquement pour tester le pipeline OCR, RAG et LLM de QraLia.")

    pdf.showPage()
    pdf.save()
    print(OUTPUT)


if __name__ == "__main__":
    main()
