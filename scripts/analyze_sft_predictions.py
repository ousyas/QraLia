"""Analyse reproductible des prédictions gelées SFT QraLia v0.4."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Sequence

from pydantic import ValidationError

from qralia.rag.generation_evaluation import GroundedAnswer


ROOT = Path(__file__).resolve().parents[1]
ARABIC = re.compile(r"[\u0600-\u06ff]")
CITATION = re.compile(r"\[([^\]]+)\]")
PASSAGE = re.compile(r"<passage\b[^>]*>(.*?)</passage>", re.IGNORECASE | re.DOTALL)
NUMBER = re.compile(
    r"(?<![A-Za-z0-9])(\d+(?:[.,]\d+)?)(?:h|j|jours?)?(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_predictions(path: Path) -> list[dict[str, Any]]:
    """Lit soit le JSONL Kaggle, soit le rapport JSON de l'évaluateur Ollama."""
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
            raise ValueError("Le rapport JSON de prédictions doit contenir une liste 'cases'")
        return payload["cases"]
    return read_jsonl(path)


def script_is_correct(text: str, language: str) -> bool:
    sample = CITATION.sub("", text)
    alphabetic = [character for character in sample if character.isalpha()]
    if not alphabetic:
        return True
    ratio = sum(bool(ARABIC.fullmatch(character)) for character in alphabetic) / len(alphabetic)
    return ratio >= 0.5 if language in {"ar", "darija_arab"} else ratio < 0.5


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = round((len(ordered) - 1) * fraction)
    return float(ordered[index])


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"case_count": 0}
    valid = [record for record in records if record["schema_valid"]]
    answered = [record for record in valid if not record["expected_abstention"]]
    latencies = [float(record["latency_seconds"]) for record in records]
    truncated = [
        record
        for record in records
        if not record["schema_valid"] and record["completion_tokens"] >= 500
    ]
    return {
        "case_count": len(records),
        "strict_json_rate": mean(float(record["strict_json_valid"]) for record in records),
        "schema_valid_rate": mean(float(record["schema_valid"]) for record in records),
        "valid_output_core_pass_rate": (
            mean(float(record["core_deterministic_pass"]) for record in valid)
            if valid
            else None
        ),
        "valid_output_strict_grounded_pass_rate": (
            mean(float(record["strict_grounded_pass"]) for record in valid)
            if valid
            else None
        ),
        "valid_output_abstention_accuracy": (
            mean(float(record["abstention_correct"]) for record in valid)
            if valid
            else None
        ),
        "core_deterministic_pass_rate": mean(
            float(record["core_deterministic_pass"]) for record in records
        ),
        "strict_grounded_pass_rate": mean(
            float(record["strict_grounded_pass"]) for record in records
        ),
        "abstention_accuracy": mean(
            float(record["abstention_correct"]) for record in records
        ),
        "false_abstention_count": sum(record["false_abstention"] for record in records),
        "unsafe_answer_count": sum(record["unsafe_answer"] for record in records),
        "gold_citation_precision": (
            mean(float(record["gold_citation_precision"]) for record in answered)
            if answered
            else None
        ),
        "gold_citation_recall": (
            mean(float(record["gold_citation_recall"]) for record in answered)
            if answered
            else None
        ),
        "distractor_citation_case_count": sum(record["cites_distractor"] for record in records),
        "critical_value_recall": (
            mean(float(record["critical_values_complete"]) for record in answered)
            if answered
            else None
        ),
        "mean_claim_count": (
            mean(float(record["claim_count"]) for record in valid) if valid else 0.0
        ),
        "mean_latency_seconds": mean(latencies),
        "median_latency_seconds": median(latencies),
        "p95_latency_seconds": percentile(latencies, 0.95),
        "mean_prompt_tokens": mean(float(record["prompt_tokens"]) for record in records),
        "mean_completion_tokens": mean(
            float(record["completion_tokens"]) for record in records
        ),
        "near_generation_limit_count": sum(
            int(record["completion_tokens"] >= 500) for record in records
        ),
        "truncated_invalid_output_count": len(truncated),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "data/training/qralia_sft_v0.4/test_cases.jsonl",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=ROOT / "data/training/sft_chunk_targets.v0.4.jsonl",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=ROOT / "data/training/qralia_sft_v0.4/test_prompts.jsonl",
    )
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument(
        "--answer-language",
        choices=("fr", "ar", "darija_latn", "darija_arab"),
        help="Langue de sortie expérimentale, différente de la langue des questions.",
    )
    parser.add_argument(
        "--use-rendered-answer",
        action="store_true",
        help="Évalue la réponse après ajout déterministe des citations par l'application.",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    cases = {row["case_id"]: row for row in read_jsonl(args.cases)}
    predictions = read_predictions(args.predictions)
    if args.use_rendered_answer:
        for prediction in predictions:
            if isinstance(prediction.get("answer"), dict):
                prediction["model_raw_output"] = prediction["raw_output"]
                prediction["raw_output"] = json.dumps(
                    prediction["answer"], ensure_ascii=False, separators=(",", ":")
                )
    prediction_ids = [row["case_id"] for row in predictions]
    if len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("case_id dupliqué dans les prédictions")
    if not set(prediction_ids).issubset(cases):
        raise ValueError(
            f"case_id inconnus={sorted(set(prediction_ids) - set(cases))}"
        )
    cases = {case_id: cases[case_id] for case_id in prediction_ids}

    targets = {row["chunk_id"]: row for row in read_jsonl(args.targets)}
    prompts = {
        row["case_id"]: row["messages"][0]["content"]
        for row in read_jsonl(args.prompts)
    }
    records: list[dict[str, Any]] = []
    for prediction in predictions:
        case = cases[prediction["case_id"]]
        answer_language = args.answer_language or case["language"]
        raw_output = prediction["raw_output"].strip()
        errors: list[str] = []
        answer: GroundedAnswer | None = None
        strict_json_valid = False
        schema_valid = False
        try:
            json.loads(raw_output)
            strict_json_valid = True
            answer = GroundedAnswer.model_validate_json(raw_output)
            schema_valid = True
        except (json.JSONDecodeError, ValidationError) as exc:
            errors.append(f"invalid_output:{type(exc).__name__}")

        allowed = set(case["context_chunk_ids"])
        gold = set(case["gold_chunk_ids"])
        distractors = set(case["distractor_chunk_ids"])
        cited: set[str] = set()
        abstention_correct = False
        false_abstention = False
        unsafe_answer = False
        language_correct = False
        writing_script_correct = False
        visible_citations_complete = False
        claims_well_formed = False
        critical_values: set[str] = set()
        critical_values_complete = True
        invented_values: set[str] = set()
        claim_count = 0

        if answer is not None:
            claim_count = len(answer.claims)
            cited = {citation for claim in answer.claims for citation in claim.citations}
            language_correct = answer.language == answer_language
            if not language_correct:
                errors.append("declared_language")
            language_sample = answer.abstention_reason or "" if answer.abstain else answer.answer
            writing_script_correct = script_is_correct(language_sample, answer_language)
            if not writing_script_correct:
                errors.append("writing_script")
            abstention_correct = answer.abstain == case["should_abstain"]
            false_abstention = bool(answer.abstain and not case["should_abstain"])
            unsafe_answer = bool(not answer.abstain and case["should_abstain"])
            if not abstention_correct:
                errors.append("abstention_decision")
            claims_well_formed = (
                (answer.abstain and not answer.claims)
                or (
                    not answer.abstain
                    and bool(answer.claims)
                    and all(claim.citations for claim in answer.claims)
                )
            )
            if not claims_well_formed:
                errors.append("claims_structure")
            if cited - allowed:
                errors.append("citation_outside_context")
            visible_citations_complete = all(
                f"[{citation}]" in answer.answer
                for claim in answer.claims
                for citation in claim.citations
            )
            if not visible_citations_complete:
                errors.append("citation_missing_from_answer")

            if not case["should_abstain"]:
                for chunk_id in case["gold_chunk_ids"]:
                    target = targets.get(chunk_id)
                    if target:
                        for claim in target["claims"][answer_language]:
                            critical_values.update(NUMBER.findall(claim))
                critical_values_complete = all(
                    value in answer.answer for value in critical_values
                )
                if not critical_values_complete:
                    errors.append("critical_value_missing")
            answer_without_citations = CITATION.sub("", answer.answer)
            output_values = set(NUMBER.findall(answer_without_citations))
            # Tous les faits numériques autorisés sont visibles dans le prompt
            # source ; les IDs de citation sont retirés avant ce contrôle.
            passage_text = "\n".join(PASSAGE.findall(prompts[case["case_id"]]))
            prompt_values = set(NUMBER.findall(passage_text))
            invented_values = output_values - prompt_values
            if invented_values:
                errors.append("numeric_value_not_in_context")

        gold_precision = len(cited & gold) / len(cited) if cited else float(not gold)
        gold_recall = len(cited & gold) / len(gold) if gold else float(not cited)
        cites_distractor = bool(cited & distractors)
        core_errors = {
            "invalid_output:JSONDecodeError",
            "invalid_output:ValidationError",
            "declared_language",
            "writing_script",
            "abstention_decision",
            "claims_structure",
            "citation_outside_context",
            "citation_missing_from_answer",
            "numeric_value_not_in_context",
        }
        core_pass = schema_valid and not any(error in core_errors for error in errors)
        strict_pass = bool(
            core_pass
            and (
                case["should_abstain"]
                or (
                    gold_precision == 1.0
                    and gold_recall == 1.0
                    and not cites_distractor
                    and critical_values_complete
                )
            )
        )
        records.append({
            "case_id": case["case_id"],
            "intent_id": case["intent_id"],
            "semantic_family_id": case["semantic_family_id"],
            "language": case["language"],
            "answer_language": answer_language,
            "provider_id": case["provider_id"],
            "formulation_kind": case["formulation_kind"],
            "retrieval_outcome": case["retrieval_outcome"],
            "expected_abstention": case["should_abstain"],
            "question": case["question"],
            "raw_output": raw_output,
            "answer": answer.model_dump() if answer else None,
            "strict_json_valid": strict_json_valid,
            "schema_valid": schema_valid,
            "language_correct": language_correct,
            "writing_script_correct": writing_script_correct,
            "abstention_correct": abstention_correct,
            "false_abstention": false_abstention,
            "unsafe_answer": unsafe_answer,
            "claims_well_formed": claims_well_formed,
            "visible_citations_complete": visible_citations_complete,
            "cited_chunk_ids": sorted(cited),
            "gold_chunk_ids": sorted(gold),
            "gold_citation_precision": gold_precision,
            "gold_citation_recall": gold_recall,
            "cites_distractor": cites_distractor,
            "critical_values": sorted(critical_values),
            "critical_values_complete": critical_values_complete,
            "invented_numeric_values": sorted(invented_values),
            "claim_count": claim_count,
            "errors": errors,
            "core_deterministic_pass": core_pass,
            "strict_grounded_pass": strict_pass,
            "prompt_tokens": int(prediction["prompt_tokens"]),
            "completion_tokens": int(prediction["completion_tokens"]),
            "latency_seconds": float(prediction["latency_seconds"]),
        })

    def breakdown(field: str) -> dict[str, Any]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[str(record[field])].append(record)
        return {key: summarize(value) for key, value in sorted(grouped.items())}

    baseline_comparison = None
    if args.baseline_report and args.baseline_report.exists():
        baseline = json.loads(args.baseline_report.read_text(encoding="utf-8"))
        baseline_by_id = {row["case_id"]: row for row in baseline["cases"]}
        lora_by_id = {row["case_id"]: row for row in records}
        common = sorted(set(baseline_by_id) & set(lora_by_id))
        baseline_comparison = {
            "case_count": len(common),
            "languages": sorted({lora_by_id[case_id]["language"] for case_id in common}),
            "base_deterministic_pass_rate": mean(
                float(baseline_by_id[case_id]["deterministic_pass"])
                for case_id in common
            ),
            "lora_core_deterministic_pass_rate": mean(
                float(lora_by_id[case_id]["core_deterministic_pass"])
                for case_id in common
            ),
            "lora_strict_grounded_pass_rate": mean(
                float(lora_by_id[case_id]["strict_grounded_pass"])
                for case_id in common
            ),
            "base_pass_lora_fail": [
                case_id
                for case_id in common
                if baseline_by_id[case_id]["deterministic_pass"]
                and not lora_by_id[case_id]["core_deterministic_pass"]
            ],
            "base_fail_lora_pass": [
                case_id
                for case_id in common
                if not baseline_by_id[case_id]["deterministic_pass"]
                and lora_by_id[case_id]["core_deterministic_pass"]
            ],
        }

    report = {
        "schema_version": "0.1.0",
        "protocol": "sft_v0.4_frozen_test_deterministic_audit",
        "configuration": {
            "predictions": str(args.predictions),
            "case_count": len(records),
            "answer_language": args.answer_language,
            "use_rendered_answer": args.use_rendered_answer,
            "test_used_for_tuning": False,
        },
        "summary": summarize(records),
        "error_counts": Counter(error for record in records for error in record["errors"]),
        "breakdown": {
            "by_language": breakdown("language"),
            "by_intent": breakdown("intent_id"),
            "by_formulation_kind": breakdown("formulation_kind"),
            "by_retrieval_outcome": breakdown("retrieval_outcome"),
            "by_provider": breakdown("provider_id"),
        },
        "baseline_comparison": baseline_comparison,
        "cases": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "summary": report["summary"],
        "error_counts": report["error_counts"],
        "baseline_comparison": baseline_comparison,
    }, ensure_ascii=False, indent=2))
    print(f"Rapport: {args.output}")


if __name__ == "__main__":
    main()
