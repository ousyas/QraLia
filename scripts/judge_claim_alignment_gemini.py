"""Applique la barrière Gemini question -> claim -> passages à un rapport existant."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Sequence

from qralia.rag.claim_verification import (
    DEFAULT_GEMINI_CLAIM_JUDGE_MODEL,
    GeminiClaimVerifier,
    online_deterministic_gate,
)
from qralia.rag.generation_evaluation import GroundedAnswer


ROOT = Path(__file__).resolve().parents[1]
PASSAGE = re.compile(
    r'<passage id="([^"]+)">\s*(.*?)\s*</passage>',
    re.IGNORECASE | re.DOTALL,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def chunks_from_prompt(prompt: str) -> dict[str, str]:
    return {chunk_id: content.strip() for chunk_id, content in PASSAGE.findall(prompt)}


def batched(values: Sequence[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [list(values[index : index + size]) for index in range(0, len(values), size)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "data/training/qralia_sft_v0.4/test_cases.jsonl",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=ROOT / "data/training/qralia_sft_v0.4/test_prompts.jsonl",
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--model", default=DEFAULT_GEMINI_CLAIM_JUDGE_MODEL)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--finalize-partial",
        action="store_true",
        help="Produit un rapport incomplet à partir du cache sans appeler Gemini.",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > 8:
        raise ValueError("--batch-size doit être compris entre 1 et 8")

    generation = json.loads(args.generation_report.read_text(encoding="utf-8"))
    case_metadata = {row["case_id"]: row for row in read_jsonl(args.cases)}
    prompt_rows = {row["case_id"]: row for row in read_jsonl(args.prompts)}

    inputs: list[dict[str, Any]] = []
    for record in generation["cases"]:
        case_id = record["case_id"]
        case = case_metadata[case_id]
        contexts_by_id = chunks_from_prompt(prompt_rows[case_id]["messages"][0]["content"])
        answer = GroundedAnswer.model_validate(record["answer"])
        gate = online_deterministic_gate(answer, contexts_by_id, expected_language="ar")
        inputs.append({
            "case_id": case_id,
            "provider_id": case["provider_id"],
            "question": case["question"],
            "answer": answer,
            "contexts_by_id": contexts_by_id,
            "expected_should_abstain": case["should_abstain"],
            "online_gate": gate,
        })
    if args.limit is not None:
        inputs = inputs[: args.limit]

    partial_path = args.output.with_suffix(".partial.jsonl")
    output_records = read_jsonl(partial_path) if partial_path.exists() else []
    completed_ids = {record["case_id"] for record in output_records}
    inputs = [item for item in inputs if item["case_id"] not in completed_ids]
    verifier = None if args.finalize_partial else GeminiClaimVerifier(
        model=args.model, env_file=args.env_file
    )
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    pending_batches = [] if args.finalize_partial else batched(inputs, args.batch_size)
    for batch_index, batch in enumerate(pending_batches, start=1):
        started = perf_counter()
        assert verifier is not None
        result, usage = verifier.verify_batch(batch)
        latency = perf_counter() - started
        for key in usage_totals:
            usage_totals[key] += usage[key]
        verdict_by_id = {item.case_id: item for item in result.cases}
        for item in batch:
            verdict = verdict_by_id[item["case_id"]]
            final_safe = bool(item["online_gate"].passed and verdict.safe_to_show)
            output_records.append({
                "case_id": item["case_id"],
                "question": item["question"],
                "expected_should_abstain": item["expected_should_abstain"],
                "answer": item["answer"].model_dump(),
                "online_deterministic_gate": item["online_gate"].model_dump(),
                "gemini_verdict": verdict.model_dump(),
                "final_safe_to_show": final_safe,
                "batch_index": batch_index,
            })
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                for record in output_records
            ),
            encoding="utf-8",
        )
        print(
            f"batch {batch_index}: {len(batch)} cas, {latency:.1f}s, "
            f"safe={sum(record['final_safe_to_show'] for record in output_records[-len(batch):])}",
            flush=True,
        )

    judged_as_safe = [record for record in output_records if record["final_safe_to_show"]]
    unsafe_expected = [record for record in output_records if record["expected_should_abstain"]]
    unsafe_candidates = [
        record for record in unsafe_expected if not record["answer"]["abstain"]
    ]
    correct_abstentions = [
        record for record in unsafe_expected if record["answer"]["abstain"]
    ]
    summary = {
        "case_count": len(output_records),
        "expected_case_count": len(generation["cases"]),
        "complete": len(output_records) == len(generation["cases"]),
        "missing_case_ids": sorted(
            {record["case_id"] for record in generation["cases"]}
            - {record["case_id"] for record in output_records}
        ),
        "gemini_safe_to_show_rate": mean(
            float(record["gemini_verdict"]["safe_to_show"])
            for record in output_records
        ) if output_records else 0.0,
        "final_safe_to_show_rate": len(judged_as_safe) / len(output_records) if output_records else 0.0,
        "expected_abstention_protection_rate": mean(
            float(record["final_safe_to_show"] == record["answer"]["abstain"])
            for record in unsafe_expected
        ) if unsafe_expected else None,
        "unsafe_candidate_block_rate": mean(
            float(not record["final_safe_to_show"])
            for record in unsafe_candidates
        ) if unsafe_candidates else None,
        "correct_abstention_accept_rate": mean(
            float(record["final_safe_to_show"])
            for record in correct_abstentions
        ) if correct_abstentions else None,
        "online_gate_failure_count": sum(
            not record["online_deterministic_gate"]["passed"]
            for record in output_records
        ),
        "usage_current_run": usage_totals,
    }
    report = {
        "schema_version": "0.1.0",
        "protocol": "question_claim_cited_passages_gemini_gate",
        "configuration": {
            "generator": generation["configuration"].get("model"),
            "judge_model": args.model,
            "batch_size": args.batch_size,
            "generation_report": str(args.generation_report),
            "gold_labels_not_sent_to_judge": True,
        },
        "summary": summary,
        "cases": output_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Rapport: {args.output}")


if __name__ == "__main__":
    main()
