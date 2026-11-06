from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from qralia.rag.onee_corpus import validate_with_schema


ROOT = Path(__file__).resolve().parents[1]


def test_telecom_evaluation_is_balanced_and_covers_every_chunk() -> None:
    cases = [
        json.loads(line)
        for line in (ROOT / "data/evaluation/telecom_retrieval_questions.v0.1.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    schema = ROOT / "schemas/retrieval_evaluation_case_v0.3.schema.json"
    for case in cases:
        validate_with_schema(case, schema)

    assert len(cases) == 56
    assert Counter(case["language"] for case in cases) == {
        "fr": 14, "ar": 14, "darija_latn": 14, "darija_arab": 14,
    }
    assert sum(case["should_abstain"] for case in cases) == 16
    expected = {
        *(('iam', 'mobile_postpaid', section) for section in (
            'facturation-mensuelle', 'exigibilite-non-reception', 'erreur-facturation',
            'prelevement-contestation', 'penalites-retard', 'suspension-retablissement',
            'resiliation-client',
        )),
        *(('orange_ma', 'consumer_postpaid', section) for section in (
            'delai-paiement', 'paiement-max-it-carte', 'paiement-espace-client-carte',
            'paiement-orange-money', 'paiement-tassbik', 'paiement-agents',
            'paiement-prelevement', 'paiement-recharge', 'paiement-agence',
            'consulter-facture',
        )),
        *(('inwi', 'consumer_postpaid', section) for section in (
            'paiement-my-inwi', 'paiement-agence', 'paiement-ebanking',
            'paiement-prelevement', 'paiement-agents', 'paiement-inwi-money',
        )),
        ('anrt', 'cross_operator_complaints', 'reclamation-operateur'),
        ('anrt', 'cross_operator_complaints', 'recours-anrt'),
    }
    covered = {
        (case["provider_id"], case["service_type"], section)
        for case in cases for section in case["relevant_section_ids"]
    }
    assert expected <= covered
    test_positive_providers = {
        case["provider_id"] for case in cases
        if case["split"] == "test" and not case["should_abstain"]
    }
    assert test_positive_providers == {"iam", "orange_ma", "inwi", "anrt"}
