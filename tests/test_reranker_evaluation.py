from __future__ import annotations

import pytest

from qralia.rag.reranker_evaluation import candidate_set_metrics, rerank_with_scores


def _candidate(chunk_id: str, section_id: str, score: float) -> dict:
    return {
        "chunk_id": chunk_id,
        "section_id": section_id,
        "score": score,
        "confidence": score,
        "retriever": "weighted_rrf",
    }


def test_reranker_reorders_only_the_existing_candidates() -> None:
    candidates = [
        _candidate("chunk-a", "agence", 0.9),
        _candidate("chunk-b", "wallet", 0.8),
        _candidate("chunk-c", "carte", 0.7),
    ]

    ranking = rerank_with_scores(candidates, [0.10, 0.95, 0.30], top_k=2)

    assert [item["section_id"] for item in ranking] == ["wallet", "carte"]
    assert {item["chunk_id"] for item in ranking}.issubset(
        {item["chunk_id"] for item in candidates}
    )
    assert ranking[0]["hybrid_rank"] == 2
    assert ranking[0]["hybrid_score"] == pytest.approx(0.8)
    assert ranking[0]["confidence"] == pytest.approx(0.95)


def test_reranker_uses_hybrid_rank_as_stable_tie_breaker() -> None:
    candidates = [
        _candidate("chunk-b", "first", 0.9),
        _candidate("chunk-a", "second", 0.8),
    ]

    ranking = rerank_with_scores(candidates, [0.5, 0.5], top_k=2)

    assert [item["section_id"] for item in ranking] == ["first", "second"]


def test_candidate_set_metrics_exposes_the_reranker_ceiling() -> None:
    cases = [
        {"relevant_section_ids": ["a", "b"]},
        {"relevant_section_ids": ["c"]},
        {"relevant_section_ids": []},
    ]
    candidates = [
        [_candidate("one", "a", 0.1)],
        [_candidate("two", "x", 0.1)],
        [],
    ]

    metrics = candidate_set_metrics(cases, candidates)

    assert metrics == {
        "positive_case_count": 2,
        "recall": pytest.approx(0.25),
        "binary_recall": pytest.approx(0.5),
    }


def test_reranker_rejects_a_score_count_mismatch() -> None:
    with pytest.raises(ValueError, match="un score par candidat"):
        rerank_with_scores([_candidate("a", "a", 0.1)], [], top_k=1)
