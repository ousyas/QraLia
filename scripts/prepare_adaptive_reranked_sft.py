"""Calibre le contexte adaptatif et prépare les prompts SFT rerankés."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from time import perf_counter
from typing import Any

from qralia.rag.adaptive_context import calibrate_policy, policy_metrics, select_adaptive_context
from qralia.rag.reranker_evaluation import Qwen3CrossEncoderReranker


ROOT = Path(__file__).resolve().parents[1]
PASSAGE = re.compile(
    r'(<passage id="([^"]+)">\s*(.*?)\s*</passage>)',
    re.IGNORECASE | re.DOTALL,
)
RERANKER_INSTRUCTION = (
    "Given a multilingual customer question about utility or telecom bills, "
    "retrieve passages that directly contain the information needed to answer it."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def passages_from_prompt(prompt: str) -> list[dict[str, str]]:
    return [
        {"block": block, "chunk_id": chunk_id, "content": content.strip()}
        for block, chunk_id, content in PASSAGE.findall(prompt)
    ]


def question_from_prompt(prompt: str) -> str:
    match = re.search(
        r"\nQUESTION\n(.*?)\n\nPASSAGES APPROUVÉS\n",
        prompt,
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError("Question introuvable dans le prompt")
    return match.group(1).strip()


def score_rows(
    rows: list[dict[str, Any]],
    reranker: Qwen3CrossEncoderReranker,
) -> list[list[dict[str, Any]]]:
    pairs: list[tuple[str, str]] = []
    slices: list[tuple[int, int]] = []
    parsed: list[list[dict[str, str]]] = []
    for row in rows:
        prompt = row["messages"][0]["content"]
        passages = passages_from_prompt(prompt)
        parsed.append(passages)
        start = len(pairs)
        question = row.get("question") or question_from_prompt(prompt)
        pairs.extend((question, passage["content"]) for passage in passages)
        slices.append((start, len(pairs)))
    scores = reranker.score(pairs)
    rankings: list[list[dict[str, Any]]] = []
    for passages, (start, end) in zip(parsed, slices):
        ranking = []
        for passage, score in zip(passages, scores[start:end]):
            ranking.append({**passage, "reranker_score": float(score)})
        ranking.sort(key=lambda item: (-item["reranker_score"], item["chunk_id"]))
        rankings.append(ranking)
    return rankings


def replace_passages(prompt: str, selected: list[dict[str, Any]]) -> str:
    marker = "PASSAGES APPROUVÉS\n"
    prefix, separator, _ = prompt.partition(marker)
    if not separator:
        raise ValueError("Marqueur PASSAGES APPROUVÉS absent du prompt")
    body = "\n\n".join(item["block"] for item in selected)
    if not body:
        body = "AUCUN PASSAGE APPROUVÉ"
    return prefix + marker + body + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation",
        type=Path,
        default=ROOT / "data/training/qralia_sft_v0.4/validation.jsonl",
    )
    parser.add_argument(
        "--test-cases",
        type=Path,
        default=ROOT / "data/training/qralia_sft_v0.4/test_cases.jsonl",
    )
    parser.add_argument(
        "--test-prompts",
        type=Path,
        default=ROOT / "data/training/qralia_sft_v0.4/test_prompts.jsonl",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--language", default="darija_arab")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--minimum-chunks",
        type=int,
        choices=(0, 1),
        default=1,
        help="1 conserve toujours le meilleur passage; 0 autorise un contexte vide.",
    )
    args = parser.parse_args()

    # La calibration est faite sur la même famille linguistique que le test,
    # mais sur des intents de validation distincts.
    validation = [
        row for row in read_jsonl(args.validation) if row["language"] == args.language
    ]
    test_cases = [row for row in read_jsonl(args.test_cases) if row["language"] == args.language]
    prompts_by_id = {row["case_id"]: row for row in read_jsonl(args.test_prompts)}
    test_rows = [
        {
            **case,
            "messages": prompts_by_id[case["case_id"]]["messages"],
        }
        for case in test_cases
    ]

    started = perf_counter()
    reranker = Qwen3CrossEncoderReranker(
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        instruction=RERANKER_INSTRUCTION,
    )
    load_seconds = perf_counter() - started
    started = perf_counter()
    validation_rankings = score_rows(validation, reranker)
    test_rankings = score_rows(test_rows, reranker)
    scoring_seconds = perf_counter() - started

    policy, calibration = calibrate_policy(
        validation,
        validation_rankings,
        abstention_thresholds=(
            [0.0] if args.minimum_chunks == 1 else [index / 20 for index in range(1, 20)]
        ),
        relative_ratios=[0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95],
        max_chunks=3,
    )
    test_selection_metrics = policy_metrics(test_cases, test_rankings, policy)

    adaptive_cases: list[dict[str, Any]] = []
    adaptive_prompts: list[dict[str, Any]] = []
    audit_cases: list[dict[str, Any]] = []
    for case, prompt_row, ranking in zip(test_cases, test_rows, test_rankings):
        selected = select_adaptive_context(ranking, policy)
        selected_ids = [item["chunk_id"] for item in selected]
        updated_case = dict(case)
        updated_case["context_chunk_ids"] = selected_ids
        updated_case["distractor_chunk_ids"] = [
            chunk_id for chunk_id in case["distractor_chunk_ids"] if chunk_id in selected_ids
        ]
        adaptive_cases.append(updated_case)
        adaptive_prompts.append({
            "case_id": case["case_id"],
            "messages": [{
                "role": "user",
                "content": replace_passages(prompt_row["messages"][0]["content"], selected),
            }],
        })
        audit_cases.append({
            "case_id": case["case_id"],
            "question": case["question"],
            "should_abstain": case["should_abstain"],
            "gold_chunk_ids": case["gold_chunk_ids"],
            "selected_chunk_ids": selected_ids,
            "selected_count": len(selected_ids),
            "ranking": [
                {"chunk_id": item["chunk_id"], "reranker_score": item["reranker_score"]}
                for item in ranking
            ],
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "test_cases.jsonl", adaptive_cases)
    write_jsonl(args.output_dir / "test_prompts.jsonl", adaptive_prompts)
    report = {
        "schema_version": "0.1.0",
        "configuration": {
            "reranker_model": reranker.model_name,
            "reranker_revision": reranker.revision,
            "device": args.device,
            "max_length": args.max_length,
            "instruction": RERANKER_INSTRUCTION,
            "minimum_chunks": args.minimum_chunks,
            "load_seconds": load_seconds,
            "scoring_seconds": scoring_seconds,
            "calibration_case_count": len(validation),
            "test_case_count": len(test_cases),
            "test_labels_not_used_for_calibration": True,
        },
        "policy": {
            "abstention_threshold": policy.abstention_threshold,
            "relative_ratio": policy.relative_ratio,
            "max_chunks": policy.max_chunks,
        },
        "calibration_metrics": calibration,
        "test_selection_metrics": test_selection_metrics,
        "selected_count_distribution": {
            str(count): sum(item["selected_count"] == count for item in audit_cases)
            for count in range(4)
        },
        "cases": audit_cases,
    }
    (args.output_dir / "selection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "policy": report["policy"],
        "calibration_metrics": calibration,
        "test_selection_metrics": test_selection_metrics,
        "selected_count_distribution": report["selected_count_distribution"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
