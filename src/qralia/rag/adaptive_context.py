"""Sélection adaptative de 0 à 3 passages après reranking."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from statistics import mean
from typing import Any, Sequence


@dataclass(frozen=True)
class AdaptiveContextPolicy:
    abstention_threshold: float
    relative_ratio: float
    max_chunks: int = 3


def select_adaptive_context(
    ranked_candidates: Sequence[dict[str, Any]],
    policy: AdaptiveContextPolicy,
) -> list[dict[str, Any]]:
    """Conserve zéro passage si la preuve est faible, sinon 1 à 3 passages.

    Le premier passage est conservé lorsqu'il franchit le seuil d'abstention.
    Les suivants ne sont conservés que si leur score reste suffisamment proche
    du meilleur. La règle dépend uniquement des scores, jamais des labels gold.
    """
    if not ranked_candidates:
        return []
    ranked = sorted(
        ranked_candidates,
        key=lambda item: (-float(item["reranker_score"]), item["chunk_id"]),
    )
    best_score = float(ranked[0]["reranker_score"])
    if best_score < policy.abstention_threshold:
        return []
    selected = [ranked[0]]
    for candidate in ranked[1 : policy.max_chunks]:
        score = float(candidate["reranker_score"])
        if score >= best_score * policy.relative_ratio:
            selected.append(candidate)
    return selected


def policy_metrics(
    cases: Sequence[dict[str, Any]],
    rankings: Sequence[Sequence[dict[str, Any]]],
    policy: AdaptiveContextPolicy,
) -> dict[str, float]:
    positive_recalls: list[float] = []
    positive_precisions: list[float] = []
    negative_empty: list[float] = []
    selected_counts: list[float] = []
    for case, ranking in zip(cases, rankings):
        selected = select_adaptive_context(ranking, policy)
        selected_ids = {item["chunk_id"] for item in selected}
        selected_counts.append(float(len(selected)))
        if case["should_abstain"]:
            negative_empty.append(float(not selected))
            continue
        gold = set(case["gold_chunk_ids"])
        positive_recalls.append(len(gold & selected_ids) / len(gold) if gold else 0.0)
        positive_precisions.append(
            len(gold & selected_ids) / len(selected_ids) if selected_ids else 0.0
        )
    recall = mean(positive_recalls) if positive_recalls else 0.0
    precision = mean(positive_precisions) if positive_precisions else 0.0
    negative_accuracy = mean(negative_empty) if negative_empty else 0.0
    # La couverture reste prioritaire, puis la sûreté des abstentions et enfin
    # la réduction du bruit. Les coefficients sont figés avant le test.
    objective = 0.5 * recall + 0.3 * negative_accuracy + 0.2 * precision
    return {
        "positive_gold_recall": recall,
        "positive_context_precision": precision,
        "negative_empty_accuracy": negative_accuracy,
        "mean_selected_chunks": mean(selected_counts) if selected_counts else 0.0,
        "objective": objective,
    }


def calibrate_policy(
    cases: Sequence[dict[str, Any]],
    rankings: Sequence[Sequence[dict[str, Any]]],
    *,
    abstention_thresholds: Sequence[float],
    relative_ratios: Sequence[float],
    max_chunks: int = 3,
) -> tuple[AdaptiveContextPolicy, dict[str, float]]:
    """Grid search déterministe sur la validation uniquement."""
    candidates = []
    for threshold, ratio in product(abstention_thresholds, relative_ratios):
        policy = AdaptiveContextPolicy(threshold, ratio, max_chunks)
        metrics = policy_metrics(cases, rankings, policy)
        candidates.append((policy, metrics))
    # En cas d'égalité : meilleur rappel, moins de chunks, seuil plus élevé et
    # ratio plus élevé. Ce départage est stable et favorise moins de bruit.
    return max(
        candidates,
        key=lambda item: (
            item[1]["objective"],
            item[1]["positive_gold_recall"],
            -item[1]["mean_selected_chunks"],
            item[0].abstention_threshold,
            item[0].relative_ratio,
        ),
    )
