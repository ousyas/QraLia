from __future__ import annotations

import math

from langchain_core.embeddings import Embeddings

from qralia.rag.semantic_evaluation import (
    LangChainSemanticIndex,
    build_report,
    chunk_to_document,
    paired_script_metrics,
)


class KeywordEmbeddings(Embeddings):
    """Petits vecteurs déterministes : aucun modèle distant dans les tests."""

    @staticmethod
    def _embed(text: str) -> list[float]:
        lowered = text.casefold()
        vector = [
            float("chèque" in lowered or "cheque" in lowered),
            float("espèces" in lowered or "cash" in lowered),
            float("inconnu" in lowered),
        ]
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else [0.0, 0.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _chunk(
    section_id: str,
    text: str,
    *,
    distributor_id: str = "onee",
    service_zone_id: str = "onee_be_national",
) -> dict:
    return {
        "schema_version": "0.3.0",
        "chunk_id": f"doc--{section_id}",
        "document_id": "doc",
        "source_id": "source",
        "section_id": section_id,
        "text": text,
        "metadata": {
            "title": "Facture",
            "publisher": "Opérateur",
            "canonical_url": "https://example.test/source",
            "languages": ["fr"],
            "domain": "utilities",
            "provider_id": distributor_id,
            "provider_name": distributor_id.upper(),
            "service_type": "electricity",
            "service_zone_id": service_zone_id,
            "legacy_operator": None,
            "source_role": "operator_primary",
            "audiences": ["residential"],
            "validity_status": "active",
            "verified_at": "2026-09-30",
            "risk_level": "medium",
            "heading_path": ["Résidentiel", section_id],
        },
    }


def _case(case_id: str, question: str, relevant: list[str], *, split: str) -> dict:
    return {
        "schema_version": "0.1.0",
        "case_id": case_id,
        "question": question,
        "language": "fr",
        "category": "direct" if relevant else "hard_negative",
        "audience": "residential",
        "distributor_id": "onee",
        "service_zone_id": "onee_be_national",
        "relevant_section_ids": relevant,
        "should_abstain": not relevant,
        "split": split,
    }


def test_chunk_becomes_langchain_document_with_searchable_heading() -> None:
    document = chunk_to_document(_chunk("paiement-cheque", "Paiement par chèque"))

    assert document.id == "doc--paiement-cheque"
    assert document.page_content.startswith("Résidentiel > paiement-cheque")
    assert document.metadata["section_id"] == "paiement-cheque"
    assert document.metadata["canonical_url"] == "https://example.test/source"


def test_semantic_search_filters_route_before_returning_results() -> None:
    index = LangChainSemanticIndex(
        [
            _chunk("onee-cheque", "Paiement par chèque"),
            _chunk(
                "redal-cheque",
                "Paiement par chèque",
                distributor_id="srm_rsk",
                service_zone_id="rabat_sale_ex_redal",
            ),
            _chunk("onee-especes", "Paiement en espèces"),
        ],
        KeywordEmbeddings(),
    )

    results = index.search(
        "Puis-je payer par chèque ?",
        audience="residential",
        distributor_id="onee",
        service_zone_id="onee_be_national",
        top_k=5,
    )

    assert results[0]["section_id"] == "onee-cheque"
    assert all(result["provider_id"] == "onee" for result in results)
    assert all(result["service_zone_id"] == "onee_be_national" for result in results)
    assert results[0]["retriever"] == "semantic_qwen3_langchain"


def test_semantic_report_reuses_ranking_and_abstention_protocol() -> None:
    index = LangChainSemanticIndex(
        [
            _chunk("cheque", "Paiement par chèque"),
            _chunk("especes", "Paiement en espèces"),
        ],
        KeywordEmbeddings(),
    )
    cases = [
        _case("retrieval-001", "chèque", ["cheque"], split="calibration"),
        _case("retrieval-002", "inconnu", [], split="calibration"),
        _case("retrieval-003", "cash", ["especes"], split="test"),
        _case("retrieval-004", "inconnu", [], split="test"),
    ]

    report = build_report(
        index,
        cases,
        top_k=2,
        model_name="fake-test-model",
        model_revision="test",
        device="cpu",
        batch_size=2,
        max_seq_length=128,
    )

    assert report["test"]["ranking_metrics"]["recall_at_1"] == 1.0
    assert report["test"]["ranking_metrics"]["mrr"] == 1.0
    assert report["test"]["abstention_metrics"]["balanced_accuracy"] == 1.0
    assert report["configuration"]["orchestration"].startswith("langchain-core")
    assert report["configuration"]["device"] == "cpu"
    assert report["configuration"]["batch_size"] == 2
    assert report["configuration"]["max_seq_length"] == 128


def test_paired_script_metrics_compare_equivalent_intents() -> None:
    records = [
        {
            "pair_id": "script-pair-001",
            "language": "darija_latn",
            "should_abstain": False,
            "predicted_abstain": False,
            "reciprocal_rank": 0.5,
        },
        {
            "pair_id": "script-pair-001",
            "language": "darija_arab",
            "should_abstain": False,
            "predicted_abstain": False,
            "reciprocal_rank": 1.0,
        },
        {
            "pair_id": "script-pair-002",
            "language": "darija_latn",
            "should_abstain": True,
            "predicted_abstain": True,
            "reciprocal_rank": 0.0,
        },
        {
            "pair_id": "script-pair-002",
            "language": "darija_arab",
            "should_abstain": True,
            "predicted_abstain": False,
            "reciprocal_rank": 0.0,
        },
    ]

    metrics = paired_script_metrics(records)

    assert metrics["complete_pair_count"] == 2
    assert metrics["positive_pair_count"] == 1
    assert metrics["ranking_wins_by_reciprocal_rank"] == {
        "darija_arab": 1,
        "darija_latn": 0,
        "ties": 0,
    }
    assert metrics["mean_reciprocal_rank_difference_arab_minus_latin"] == 0.5
    assert metrics["abstention_accuracy_by_script"] == {
        "darija_latn": 1.0,
        "darija_arab": 0.5,
    }
