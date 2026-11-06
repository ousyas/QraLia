from scripts.prepare_conservative_reranked_sft import conservative_count


def ranking(*scores: float):
    return [{"reranker_score": score} for score in scores]


def choose(*scores: float) -> int:
    return conservative_count(
        ranking(*scores),
        single_min_score=0.995,
        single_second_max_ratio=0.12,
        double_min_score=0.90,
        double_third_max_ratio=0.10,
    )


def test_keeps_one_only_for_exceptional_confidence() -> None:
    assert choose(0.999, 0.05, 0.01, 0.001) == 1


def test_keeps_two_when_third_score_is_negligible() -> None:
    assert choose(0.95, 0.4, 0.05, 0.01) == 2


def test_defaults_to_three_when_uncertain() -> None:
    assert choose(0.6, 0.3, 0.2, 0.1) == 3
