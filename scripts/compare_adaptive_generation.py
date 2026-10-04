"""Compare une génération à contexte fixe avec une génération rerankée adaptative."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRICS = (
    "core_deterministic_pass_rate",
    "strict_grounded_pass_rate",
    "abstention_accuracy",
    "false_abstention_count",
    "unsafe_answer_count",
    "gold_citation_precision",
    "gold_citation_recall",
    "distractor_citation_case_count",
    "critical_value_recall",
    "mean_latency_seconds",
    "mean_prompt_tokens",
)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--adaptive", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    baseline = load(args.baseline)
    adaptive = load(args.adaptive)
    selection = load(args.selection)
    baseline_cases = {row["case_id"]: row for row in baseline["cases"]}
    adaptive_cases = {row["case_id"]: row for row in adaptive["cases"]}
    selection_cases = {row["case_id"]: row for row in selection["cases"]}
    if baseline_cases.keys() != adaptive_cases.keys() or baseline_cases.keys() != selection_cases.keys():
        raise ValueError("Les trois rapports doivent contenir exactement les mêmes case_id.")

    comparisons = []
    for case_id in sorted(baseline_cases):
        before = baseline_cases[case_id]
        after = adaptive_cases[case_id]
        context = selection_cases[case_id]
        selected = set(context["selected_chunk_ids"])
        gold = set(context["gold_chunk_ids"])
        gold_available = bool(selected & gold) if gold else None
        if before["strict_grounded_pass"] == after["strict_grounded_pass"]:
            strict_change = "unchanged"
        elif after["strict_grounded_pass"]:
            strict_change = "improved"
        else:
            strict_change = "regressed"
        comparisons.append({
            "case_id": case_id,
            "intent_id": before["intent_id"],
            "question": before["question"],
            "expected_abstention": before["expected_abstention"],
            "selected_count": context["selected_count"],
            "gold_available_after_reranking": gold_available,
            "baseline_core_pass": before["core_deterministic_pass"],
            "adaptive_core_pass": after["core_deterministic_pass"],
            "baseline_strict_pass": before["strict_grounded_pass"],
            "adaptive_strict_pass": after["strict_grounded_pass"],
            "strict_change": strict_change,
            "adaptive_errors": after["errors"],
        })

    metrics = {}
    for metric in METRICS:
        before = baseline["summary"][metric]
        after = adaptive["summary"][metric]
        metrics[metric] = {"baseline": before, "adaptive": after, "delta": after - before}

    positive = [row for row in comparisons if not row["expected_abstention"]]
    selection_metrics = selection.get("test_selection_metrics", selection)
    report = {
        "schema_version": "0.1.0",
        "protocol": {
            "baseline_context": "4 passages construits: gold + distracteurs",
            "adaptive_context": "Qwen3-Reranker-0.6B; 1 à 3 passages",
            "threshold_calibration": "validation darija_arab uniquement; aucun label test",
        },
        "metrics": metrics,
        "selection": {
            "distribution": selection["selected_count_distribution"],
            "mean_selected_chunks": selection_metrics["mean_selected_chunks"],
            "positive_gold_recall": selection_metrics.get(
                "positive_gold_recall",
                selection_metrics.get("positive_gold_case_recall"),
            ),
            "positive_cases": len(positive),
            "positive_cases_without_any_gold": sum(
                row["gold_available_after_reranking"] is False for row in positive
            ),
        },
        "strict_changes": {
            label: sum(row["strict_change"] == label for row in comparisons)
            for label in ("improved", "regressed", "unchanged")
        },
        "cases": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "selection": report["selection"],
        "strict_changes": report["strict_changes"],
        "metrics": metrics,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
