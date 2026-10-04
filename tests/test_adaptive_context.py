from qralia.rag.adaptive_context import AdaptiveContextPolicy, select_adaptive_context


def candidates(*scores: float):
    return [
        {"chunk_id": f"c{index}", "reranker_score": score}
        for index, score in enumerate(scores, start=1)
    ]


def test_selects_no_chunk_below_abstention_threshold() -> None:
    policy = AdaptiveContextPolicy(0.5, 0.7)
    assert select_adaptive_context(candidates(0.4, 0.3), policy) == []


def test_selects_one_chunk_when_margin_is_large() -> None:
    policy = AdaptiveContextPolicy(0.5, 0.7)
    selected = select_adaptive_context(candidates(0.9, 0.4, 0.3), policy)
    assert [item["chunk_id"] for item in selected] == ["c1"]


def test_selects_up_to_three_close_chunks() -> None:
    policy = AdaptiveContextPolicy(0.5, 0.7)
    selected = select_adaptive_context(candidates(0.9, 0.8, 0.7, 0.69), policy)
    assert [item["chunk_id"] for item in selected] == ["c1", "c2", "c3"]
