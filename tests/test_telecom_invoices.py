"""Factures d'abonnement télécom : génération, lecture et validation."""

import random
from datetime import date

from qralia.domain.extracted_invoice import FieldStatus
from qralia.domain.providers import detect_provider
from qralia.generation.telecom import OPERATORS, ground_truth, random_telecom_invoice
from qralia.ocr.extractor import OcrToken, parse_ocr_tokens
from qralia.validation.validator import validate_invoice


def _lines(*lines: str, confidence: float = 0.95) -> list[OcrToken]:
    """Une ligne de texte par ordonnée, espacées comme une page rendue à 250 dpi."""
    tokens = []
    for row, line in enumerate(lines):
        x = 10
        for word in line.split():
            tokens.append(OcrToken(word, confidence, 1, (x, 20 + row * 80, x + 10 * len(word), 50 + row * 80)))
            x += 10 * len(word) + 12
    return tokens


ORANGE = (
    "Orange Maroc SPÉCIMEN — facture synthétique",
    "Référence facture : OR076693775",
    "Code client : 635179199",
    "Numéro de ligne : 07 ** ** 71 53",
    "Date d'émission : 03.07.2026",
    "Forfait : Box Fibre 100 Méga",
    "Abonnement HT : 349,00",
    "Hors forfait HT : 0,00",
    "Total HT : 349,00",
    "TVA (20%) : 69,80",
    "Total à payer (DH TTC) : 418,80",
    "À payer avant le : 23.07.2026",
)


def test_provider_detection_refuses_ambiguous_documents():
    assert detect_provider("facture orange maroc forfait").provider_id == "orange_ma"
    assert detect_provider("maroc telecom offre").provider_id == "iam"
    assert detect_provider("inwi offre souscrite").provider_id == "inwi"
    assert detect_provider("facture d abonnement") is None
    assert detect_provider("orange maroc et inwi") is None


def test_orange_invoice_is_parsed_with_its_own_labels():
    invoice = parse_ocr_tokens(_lines(*ORANGE), "synthetic.pdf")
    fields = invoice.fields
    assert invoice.provider_id == "orange_ma"
    assert fields.invoice_number.normalized_value == "OR076693775"
    assert fields.contract_number.normalized_value == "635179199"
    assert fields.line_number.normalized_value == "07****7153"
    assert fields.plan_name.normalized_value == "Box Fibre 100 Méga"
    assert fields.total_check.normalized_value == 418.80
    assert fields.due_date.normalized_value == "2026-07-23"
    assert fields.taxes.normalized_value == 69.80
    # Pas de timbre ni de tarif BT/MT sur un abonnement télécom.
    assert fields.total_cash.status == FieldStatus.NOT_PRESENT
    assert fields.tariff.status == FieldStatus.NOT_PRESENT
    validation = validate_invoice(invoice)
    assert validation["valid"] is True
    assert validation["scope_supported"] is True
    assert not any(w.startswith("taxes_differ") for w in validation["warnings"])


def test_two_word_label_is_not_read_as_value():
    invoice = parse_ocr_tokens(_lines("inwi", "Offre souscrite : Box Internet 4G"), "synthetic.pdf")
    assert invoice.fields.plan_name.normalized_value == "Box Internet 4G"


def test_low_confidence_identifier_is_refused_not_altered():
    tokens = _lines(*ORANGE)
    tokens = [
        OcrToken(t.text, 0.0, t.page, t.bbox) if t.text == "635179199" else t
        for t in tokens
    ]
    invoice = parse_ocr_tokens(tokens, "synthetic.pdf")
    assert invoice.fields.contract_number.status == FieldStatus.UNREADABLE
    assert invoice.fields.contract_number.normalized_value is None


def test_electricity_vat_rate_is_not_applied_to_telecom():
    lines = list(ORANGE)
    lines[9] = "TVA (20%) : 48,86"  # 14 % de 349,00 : faux pour un service télécom
    lines[10] = "Total à payer (DH TTC) : 397,86"
    validation = validate_invoice(parse_ocr_tokens(_lines(*lines), "synthetic.pdf"))
    assert any(w.startswith("taxes_differ_from_vat_rate") for w in validation["warnings"])


def test_unknown_provider_is_out_of_scope():
    invoice = parse_ocr_tokens(_lines("Facture", "Total à payer : 100,00"), "synthetic.pdf")
    validation = validate_invoice(invoice)
    assert invoice.provider_id == "unknown"
    assert validation["scope_supported"] is False
    assert "unknown_provider" in validation["warnings"]


def test_generated_telecom_ground_truth_is_arithmetically_consistent():
    rng = random.Random(7)
    for provider_id in OPERATORS:
        for _ in range(20):
            invoice = random_telecom_invoice(rng, provider_id, date(2026, 11, 1))
            truth = ground_truth(invoice)
            fields = {name: value["normalized_value"] for name, value in truth["fields"].items()}
            assert truth["provider_id"] == provider_id
            assert round(fields["amount_before_tax"] + fields["taxes"] + fields["previous_unpaid"], 2) == fields["total_check"]
            assert abs(fields["taxes"] - round(fields["amount_before_tax"] * 0.20, 2)) < 0.01
            assert invoice.due_date > invoice.invoice_date > invoice.period_end
            assert "**" in invoice.line_number


def test_api_routes_official_rules_to_the_invoice_provider(tmp_path):
    import json

    from fastapi.testclient import TestClient

    from qralia.api.main import create_app

    def chunk(chunk_id, provider_id, domain, service_type, zone, text):
        return {
            "schema_version": "0.3.0", "chunk_id": chunk_id, "document_id": chunk_id, "source_id": chunk_id,
            "section_id": "paiement", "text": text,
            "metadata": {
                "title": f"Paiement {provider_id}", "publisher": provider_id, "canonical_url": f"https://example.test/{chunk_id}",
                "languages": ["fr"], "domain": domain, "provider_id": provider_id, "provider_name": provider_id,
                "service_type": service_type, "service_zone_id": zone, "legacy_operator": None,
                "source_role": "operator_primary", "audiences": ["residential"], "validity_status": "active",
                "verified_at": "2026-10-17", "risk_level": "low", "heading_path": ["Paiement"],
            },
        }

    onee = tmp_path / "onee.jsonl"
    onee.write_text(json.dumps(chunk(
        "onee-cheque", "onee", "utilities", "electricity", "onee_be_national",
        "Le paiement par chèque se fait à l'ordre de l'ONEE en agence.",
    ), ensure_ascii=False) + "\n", encoding="utf-8")
    telecom = tmp_path / "telecom.jsonl"
    telecom.write_text(json.dumps(chunk(
        "orange-cheque", "orange_ma", "telecommunications", "consumer_postpaid", "morocco_national",
        "Le paiement par chèque de votre facture Orange se fait en boutique.",
    ), ensure_ascii=False) + "\n", encoding="utf-8")

    orange = parse_ocr_tokens(_lines(*ORANGE), "synthetic.pdf")
    app = create_app(
        extractor=lambda path: orange, corpus_path=[onee, telecom],
        ocr_ready_override=True, ai_enabled_override=False,
    )
    client = TestClient(app)
    uploaded = client.post(
        "/v1/invoices", data={"confirm_synthetic": "true"},
        files={"file": ("orange.pdf", b"%PDF-1.4 synthetic", "application/pdf")},
    ).json()
    answer = client.post(
        f"/v1/invoices/{uploaded['invoice_id']}/questions",
        json={"question": "Puis-je payer par chèque ?", "language": "fr"},
    ).json()
    assert [citation["chunk_id"] for citation in answer["citations"]] == ["orange-cheque"]
