"""Prépare un contexte reranké conservateur à partir de scores déjà calculés."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PASSAGE = re.compile(
    r'(<passage id="([^"]+)">\s*(.*?)\s*</passage>)',
    re.IGNORECASE | re.DOTALL,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def passage_blocks(prompt: str) -> dict[str, str]:
    return {chunk_id: block for block, chunk_id, _ in PASSAGE.findall(prompt)}


def replace_passages(prompt: str, selected_ids: list[str]) -> str:
    marker = "PASSAGES APPROUVÉS\n"
    prefix, separator, _ = prompt.partition(marker)
    if not separator:
        raise ValueError("Marqueur PASSAGES APPROUVÉS absent du prompt")
    blocks = passage_blocks(prompt)
    body = "\n\n".join(blocks[chunk_id] for chunk_id in selected_ids)
    return prefix + marker + body + "\n"


def conservative_count(
    ranking: list[dict[str, Any]],
    *,
    single_min_score: float,
    single_second_max_ratio: float,
    double_min_score: float,
    double_third_max_ratio: float,
) -> int:
    """Retourne 1, 2 ou 3; l'incertitude conduit toujours à trois passages."""

    scores = [float(item["reranker_score"]) for item in ranking]
    best = max(scores[0], 1e-12)
    if scores[0] >= single_min_score and scores[1] / best <= single_second_max_ratio:
        return 1
    if scores[0] >= double_min_score and scores[2] / best <= double_third_max_ratio:
        return 2
    return 3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking-report", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--language", default="darija_arab")
    parser.add_argument("--single-min-score", type=float, default=0.995)
    parser.add_argument("--single-second-max-ratio", type=float, default=0.12)
    parser.add_argument("--double-min-score", type=float, default=0.90)
    parser.add_argument("--double-third-max-ratio", type=float, default=0.10)
    args = parser.parse_args()

    ranking_report = json.loads(args.ranking_report.read_text(encoding="utf-8"))
    rankings = {row["case_id"]: row for row in ranking_report["cases"]}
    cases = [row for row in read_jsonl(args.cases) if row["language"] == args.language]
    prompts = {row["case_id"]: row for row in read_jsonl(args.prompts)}

    output_cases: list[dict[str, Any]] = []
    output_prompts: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for case in cases:
        scored = rankings[case["case_id"]]
        count = conservative_count(
            scored["ranking"],
            single_min_score=args.single_min_score,
            single_second_max_ratio=args.single_second_max_ratio,
            double_min_score=args.double_min_score,
            double_third_max_ratio=args.double_third_max_ratio,
        )
        selected_ids = [item["chunk_id"] for item in scored["ranking"][:count]]
        updated = dict(case)
        updated["context_chunk_ids"] = selected_ids
        updated["distractor_chunk_ids"] = [
            chunk_id for chunk_id in case["distractor_chunk_ids"] if chunk_id in selected_ids
        ]
        output_cases.append(updated)
        original_prompt = prompts[case["case_id"]]["messages"][0]["content"]
        output_prompts.append({
            "case_id": case["case_id"],
            "messages": [{"role": "user", "content": replace_passages(original_prompt, selected_ids)}],
        })
        gold = set(case["gold_chunk_ids"])
        audit.append({
            "case_id": case["case_id"],
            "selected_count": count,
            "selected_chunk_ids": selected_ids,
            "gold_chunk_ids": case["gold_chunk_ids"],
            "should_abstain": case["should_abstain"],
            "gold_available": bool(gold & set(selected_ids)) if gold else None,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "test_cases.jsonl", output_cases)
    write_jsonl(args.output_dir / "test_prompts.jsonl", output_prompts)
    positives = [row for row in audit if not row["should_abstain"]]
    report = {
        "schema_version": "0.1.0",
        "configuration": {
            "policy": "conservative_default_3",
            "single_min_score": args.single_min_score,
            "single_second_max_ratio": args.single_second_max_ratio,
            "double_min_score": args.double_min_score,
            "double_third_max_ratio": args.double_third_max_ratio,
            "selection_does_not_use_test_labels": True,
            "evaluation_is_exploratory_because_test_was_previously_inspected": True,
        },
        "selected_count_distribution": {
            str(count): sum(row["selected_count"] == count for row in audit)
            for count in (1, 2, 3)
        },
        "mean_selected_chunks": sum(row["selected_count"] for row in audit) / len(audit),
        "positive_gold_case_recall": (
            sum(row["gold_available"] is True for row in positives) / len(positives)
        ),
        "cases": audit,
    }
    (args.output_dir / "selection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in (
        "selected_count_distribution", "mean_selected_chunks", "positive_gold_case_recall"
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
