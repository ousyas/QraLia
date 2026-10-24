"""Tests HTTP de la première API, sans appeler Tesseract ni un LLM."""

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from qralia.api.main import create_app
from qralia.domain.extracted_invoice import (
    ExtractedField,
    ExtractedInvoice,
    ExtractionMetadata,
    FieldProvenance,
    FieldStatus,
    InvoiceFields,
)


def _field(value, raw: str, *, confidence: float = 0.96, status=FieldStatus.EXTRACTED):
    return ExtractedField(
        raw_value=raw,
        normalized_value=value,
        confidence=confidence,
        status=status,
        provenance=FieldProvenance(page=1, method="test"),
    )


def _invoice(source_file: str) -> ExtractedInvoice:
    return ExtractedInvoice(
        invoice_id="F00000001",
        extraction=ExtractionMetadata(
            extracted_at=datetime(2026, 10, 17, tzinfo=timezone.utc),
            ocr_engine="fake",
            source_file=source_file,
            source_type="synthetic",
        ),
        fields=InvoiceFields(
            invoice_number=_field("F00000001", "F00000001"),
            contract_number=_field("1234567", "1234567"),
            total_cash=_field(101.5, "", status=FieldStatus.DERIVED),
            total_check=_field(100.0, "100,00"),
            due_date=_field("2026-11-01", "01/11/2026"),
            invoice_date=_field("2026-10-01", "01/10/2026"),
            tariff=_field("BT", "BT"),
            amount_before_tax=_field(90.0, "90,00"),
            taxes=_field(10.0, "10,00"),
            previous_unpaid=_field(None, "", confidence=0, status=FieldStatus.NOT_PRESENT),
        ),
        document_confidence=0.96,
    )


def _corpus(path: Path) -> Path:
    chunk = {
        "schema_version": "0.3.0",
        "chunk_id": "onee-cheque",
        "document_id": "onee-payment",
        "source_id": "onee-payment",
        "section_id": "paiement-cheque",
        "text": "Le chèque bancaire ou CCP doit être établi à l'ordre de l'ONEE et remis en agence.",
        "metadata": {
            "title": "Paiement par chèque",
            "publisher": "ONEE",
            "canonical_url": "https://example.test/onee-cheque",
            "languages": ["fr"],
            "domain": "utilities",
            "provider_id": "onee",
            "provider_name": "ONEE",
            "service_type": "electricity",
            "service_zone_id": "onee_be_national",
            "legacy_operator": None,
            "source_role": "operator_primary",
            "audiences": ["residential"],
            "validity_status": "active",
            "verified_at": "2026-10-17",
            "risk_level": "low",
            "heading_path": ["Paiement", "Chèque"],
        },
    }
    path.write_text(json.dumps(chunk, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def test_health_and_ready(tmp_path):
    app = create_app(extractor=lambda path: _invoice(path.name), corpus_path=_corpus(tmp_path / "chunks.jsonl"), ocr_ready_override=True, ai_enabled_override=False)
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    response = client.get("/ready")
    assert response.status_code == 200
    assert all(response.json()["components"].values())


def test_ready_reports_missing_dependencies(tmp_path):
    app = create_app(extractor=lambda path: _invoice(path.name), corpus_path=tmp_path / "missing.jsonl", ocr_ready_override=False, ai_enabled_override=False)
    response = TestClient(app).get("/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_invoice_question_and_delete_lifecycle(tmp_path):
    observed_paths: list[Path] = []

    def fake_extractor(path: Path):
        observed_paths.append(path)
        assert path.exists()
        return _invoice(path.name)

    app = create_app(extractor=fake_extractor, corpus_path=_corpus(tmp_path / "chunks.jsonl"), ocr_ready_override=True, ai_enabled_override=False)
    client = TestClient(app)
    upload = client.post(
        "/v1/invoices",
        data={"confirm_synthetic": "true"},
        files={"file": ("facture.pdf", b"%PDF-1.4 synthetic", "application/pdf")},
    )
    assert upload.status_code == 201
    invoice_id = upload.json()["invoice_id"]
    assert upload.json()["validation"]["valid"] is True
    assert observed_paths and not observed_paths[0].exists()
    assert client.get(f"/v1/invoices/{invoice_id}").status_code == 200

    invoice_answer = client.post(
        f"/v1/invoices/{invoice_id}/questions",
        json={"question": "Combien dois-je payer ?", "language": "fr"},
    ).json()
    assert invoice_answer["route"] == "invoice"
    assert invoice_answer["evidence"][0]["value"] == 100.0

    rag_answer = client.post(
        f"/v1/invoices/{invoice_id}/questions",
        json={"question": "Comment payer par chèque ?", "language": "fr"},
    ).json()
    assert rag_answer["route"] == "official_rag"
    assert rag_answer["citations"][0]["chunk_id"] == "onee-cheque"

    mixed = client.post(
        f"/v1/invoices/{invoice_id}/questions",
        json={"question": "Combien dois-je payer et comment payer par chèque ?", "language": "fr"},
    ).json()
    assert mixed["route"] == "both"

    assert client.delete(f"/v1/invoices/{invoice_id}").status_code == 204
    assert client.get(f"/v1/invoices/{invoice_id}").status_code == 404


def test_upload_security_checks(tmp_path):
    app = create_app(extractor=lambda path: _invoice(path.name), corpus_path=_corpus(tmp_path / "chunks.jsonl"), ocr_ready_override=True, ai_enabled_override=False)
    client = TestClient(app)
    real_document = client.post(
        "/v1/invoices",
        data={"confirm_synthetic": "false"},
        files={"file": ("facture.pdf", b"%PDF-1.4 data", "application/pdf")},
    )
    assert real_document.status_code == 400
    disguised = client.post(
        "/v1/invoices",
        data={"confirm_synthetic": "true"},
        files={"file": ("facture.pdf", b"not a pdf", "application/pdf")},
    )
    assert disguised.status_code == 415


def test_conversation_rewrites_followup_then_reuses_existing_qa_pipeline(tmp_path):
    app = create_app(
        extractor=lambda path: _invoice(path.name),
        corpus_path=_corpus(tmp_path / "chunks.jsonl"),
        ocr_ready_override=True,
        ai_enabled_override=False,
    )
    client = TestClient(app)
    upload = client.post(
        "/v1/invoices",
        data={"confirm_synthetic": "true"},
        files={"file": ("facture.pdf", b"%PDF-1.4 synthetic", "application/pdf")},
    )
    invoice_id = upload.json()["invoice_id"]
    created = client.post(
        f"/v1/invoices/{invoice_id}/conversations",
        json={"language": "fr"},
    )
    assert created.status_code == 201
    conversation_id = created.json()["conversation_id"]

    first = client.post(
        f"/v1/conversations/{conversation_id}/messages",
        json={"message": "Puis-je payer cette facture par chèque ?", "language": "fr"},
    )
    assert first.status_code == 200
    assert first.json()["resolution"]["action"] == "standalone"
    assert first.json()["answer"]["route"] == "official_rag"

    followup = client.post(
        f"/v1/conversations/{conversation_id}/messages",
        json={"message": "Et où le déposer ?", "language": "fr"},
    )
    body = followup.json()
    assert followup.status_code == 200
    assert body["resolution"]["action"] == "rewrite"
    assert "chèque" in body["resolution"]["standalone_question"]
    assert body["answer"]["route"] == "official_rag"

    history = client.get(f"/v1/conversations/{conversation_id}").json()
    assert len(history["turns"]) == 4
    assert history["turns"][2]["text"] == "Et où le déposer ?"
    assert history["turns"][2]["standalone_question"] == body["resolution"]["standalone_question"]

    assert client.delete(f"/v1/invoices/{invoice_id}").status_code == 204
    assert client.get(f"/v1/conversations/{conversation_id}").status_code == 404

