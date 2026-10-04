from __future__ import annotations

import math

from langchain_core.embeddings import Embeddings

from qralia.rag.hybrid_evaluation import HybridRetrievalIndex, canonicalize_case


class TinyEmbeddings(Embeddings):
    @staticmethod
    def _embed(text: str) -> list[float]:
        lowered = text.casefold()
        vector = [
            float("wallet" in lowered or "محفظة" in lowered),
            float("agence" in lowered or "wakala" in lowered),
            float("facture" in lowered or "factora" in lowered),
        ]
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else [0.0, 0.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _chunk(section: str, text: str, *, provider: str = "orange_ma") -> dict:
    return {
        "schema_version": "0.3.0",
        "chunk_id": f"{provider}--{section}",
        "document_id": f"doc-{provider}",
        "source_id": f"source-{provider}",
        "section_id": section,
        "text": text,
        "metadata": {
            "title": "Factures",
            "publisher": provider,
            "canonical_url": "https://example.test/source",
            "languages": ["fr"],
            "domain": "telecommunications",
            "provider_id": provider,
            "provider_name": provider,
            "service_type": "consumer_postpaid",
            "service_zone_id": "morocco_national",
            "legacy_operator": None,
            "source_role": "operator_primary",
            "audiences": ["residential"],
            "validity_status": "active",
            "verified_at": "2026-10-01",
            "risk_level": "medium",
            "heading_path": [provider, section],
        },
    }


def test_hybrid_fuses_bm25_and_semantic_rankings_without_crossing_provider() -> None:
    chunks = [
        _chunk("wallet", "Paiement avec un portefeuille wallet"),
        _chunk("agence", "Paiement en agence"),
        _chunk("wallet-inwi", "Paiement wallet", provider="inwi"),
    ]
    index = HybridRetrievalIndex(chunks, TinyEmbeddings(), candidate_k=5)

    result = index.search_components(
        "factora b wallet",
        domain="telecommunications",
        provider_id="orange_ma",
        service_type="consumer_postpaid",
        service_zone_id="morocco_national",
        audience="residential",
        top_k=5,
    )

    assert result["hybrid"][0]["section_id"] == "wallet"
    assert all(item["provider_id"] == "orange_ma" for item in result["hybrid"])
    assert result["hybrid"][0]["bm25_rank"] is not None
    assert result["hybrid"][0]["semantic_rank"] is not None


def test_legacy_case_is_projected_to_domain_and_service_without_mutation() -> None:
    chunks = [_chunk("wallet", "Paiement wallet")]
    legacy = {
        "case_id": "retrieval-001",
        "distributor_id": "orange_ma",
        "service_zone_id": "morocco_national",
        "audience": "residential",
    }

    canonical = canonicalize_case(legacy, chunks)

    assert canonical["provider_id"] == "orange_ma"
    assert canonical["domain"] == "telecommunications"
    assert canonical["service_type"] == "consumer_postpaid"
    assert "provider_id" not in legacy
