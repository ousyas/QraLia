from __future__ import annotations

from qralia.rag.telecom_corpus import (
    EXPECTED_CHUNK_COUNTS,
    build_chunks,
    extract_iam,
)


def test_chunk_budget_is_exactly_25_and_distributed_by_source() -> None:
    assert EXPECTED_CHUNK_COUNTS == {
        "iam": 7,
        "orange_payment": 9,
        "orange_consult": 1,
        "inwi_payment": 5,
        "inwi_money": 1,
        "anrt": 2,
    }
    assert sum(EXPECTED_CHUNK_COUNTS.values()) == 25


def test_iam_rendered_snapshot_is_split_into_seven_contractual_rules() -> None:
    raw = """ARTICLE 8.1.1 — FACTURATION MENSUELLE
Texte 1.
ARTICLE 8.1.2 — EXIGIBILITÉ
Texte 2.
ARTICLE 8.1.4 — ERREUR
Texte 3.
ARTICLE 8.3 — PRÉLÈVEMENT
Texte 4.
ARTICLE 8.4 — RETARD
Texte 5.
ARTICLES 17 ET 18 — SUSPENSION
Texte 6.
ARTICLE 19.1 — RÉSILIATION
Texte 7.
""".encode()

    sections = extract_iam(raw)

    assert len(sections) == 7
    assert sections[0] == ("facturation-mensuelle", "Facturation mensuelle", "Texte 1.")
    assert sections[-1][0] == "resiliation-client"


def test_chunk_metadata_separates_domain_provider_service_zone_and_audience() -> None:
    document = {
        "schema_version": "0.3.0",
        "document_id": "telecom-test-20261001t000000z",
        "source": {
            "source_id": "telecom-test",
            "title": "Test",
            "publisher": "Opérateur",
            "canonical_url": "https://example.test",
            "languages": ["fr"],
        },
        "classification": {
            "domain": "telecommunications",
            "provider_id": "orange_ma",
            "provider_name": "Orange Maroc",
            "service_type": "consumer_postpaid",
            "service_zone_id": "morocco_national",
            "legacy_operator": None,
            "source_role": "operator_primary",
            "audiences": ["residential"],
            "validity_status": "active",
            "verified_at": "2026-10-01",
            "risk_level": "medium",
        },
        "sections": [{
            "section_id": "paiement-test",
            "order": 0,
            "heading_path": ["Télécommunications", "Orange Maroc", "Paiement"],
            "text": "Passage officiel.",
        }],
    }

    metadata = build_chunks(document)[0]["metadata"]

    assert metadata["domain"] == "telecommunications"
    assert metadata["provider_id"] == "orange_ma"
    assert metadata["service_type"] == "consumer_postpaid"
    assert metadata["service_zone_id"] == "morocco_national"
    assert metadata["audiences"] == ["residential"]
