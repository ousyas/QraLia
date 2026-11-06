"""Evalue un modèle Ollama sur les prompts gelés du SFT QraLia v0.4."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any
from urllib.request import Request, urlopen

from pydantic import ValidationError

from qralia.rag.generation_evaluation import GroundedAnswer, render_answer_from_claims


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS = ROOT / "data/training/qralia_sft_v0.4/test_prompts.jsonl"
DEFAULT_CASES = ROOT / "data/training/qralia_sft_v0.4/test_cases.jsonl"
DEFAULT_OUTPUT = ROOT / "artifacts/evaluation/qwen35-4b-sft-v04-non-latin.json"
ARABIC = re.compile(r"[\u0600-\u06ff]")
CITATION = re.compile(r"\[([^\]]+)\]")
LANGUAGES = ("fr", "ar", "darija_arab")


def adapt_prompt_language(
    prompt: str,
    *,
    input_language: str,
    answer_language: str,
) -> str:
    """Change uniquement la langue de sortie d'un prompt gelé.

    L'expérience darija arabe -> arabe standard conserve la question et les
    passages à l'identique. Elle retire seulement les few-shots de style darija,
    qui contrediraient la nouvelle consigne de sortie.
    """
    if answer_language == input_language:
        return prompt
    if (input_language, answer_language) != ("darija_arab", "ar"):
        raise ValueError(
            "Seule l'expérience darija_arab -> ar est prise en charge pour le moment"
        )
    prompt = re.sub(
        r"- La langue cible est darija_arab\.[^\n]+",
        "- La question est en darija marocaine écrite en caractères arabes. "
        "Comprends-la sans en modifier le sens. Réponds uniquement en arabe "
        "standard simple et accessible.",
        prompt,
        count=1,
    )
    prompt = re.sub(
        r"\nREPÈRES DE STYLE DARIJA EN ÉCRITURE ARABE.*?"
        r"\nAvant de retourner le JSON,",
        "\n\nCONSIGNE DE LANGUE DE SORTIE\n"
        "- Utilise des phrases courtes et un vocabulaire arabe courant.\n"
        "- N'utilise pas de darija, sauf les noms propres présents dans les passages.\n"
        "- Dans le JSON, mets exactement \"language\":\"ar\".\n\n"
        "Avant de retourner le JSON,",
        prompt,
        count=1,
        flags=re.DOTALL,
    )
    return prompt


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def script_is_correct(text: str, language: str) -> bool:
    text = CITATION.sub("", text)
    alphabetic = [character for character in text if character.isalpha()]
    if not alphabetic:
        return True
    ratio = sum(bool(ARABIC.fullmatch(character)) for character in alphabetic) / len(alphabetic)
    return ratio >= 0.5 if language in {"ar", "darija_arab"} else ratio < 0.5


def validate_prediction(
    case: dict[str, Any],
    answer: GroundedAnswer,
    *,
    expected_answer_language: str | None = None,
) -> list[str]:
    errors: list[str] = []
    allowed = set(case["context_chunk_ids"])
    citations = {citation for claim in answer.claims for citation in claim.citations}
    target_language = expected_answer_language or case["language"]
    if answer.language != target_language:
        errors.append("declared_language")
    language_text = answer.abstention_reason or "" if answer.abstain else answer.answer
    if not script_is_correct(language_text, target_language):
        errors.append("writing_script")
    if answer.abstain != case["should_abstain"]:
        errors.append("abstention_decision")
    if answer.abstain and answer.claims:
        errors.append("claims_during_abstention")
    if not answer.abstain and not answer.claims:
        errors.append("missing_claims")
    if citations - allowed:
        errors.append("citation_outside_context")
    for claim in answer.claims:
        if not claim.citations:
            errors.append("claim_without_citation")
        for citation in claim.citations:
            if f"[{citation}]" not in answer.answer:
                errors.append("citation_missing_from_answer")
    return sorted(set(errors))


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"case_count": 0}
    valid = [record for record in records if record["schema_valid"]]
    return {
        "case_count": len(records),
        "strict_json_rate": mean(float(record["strict_json_valid"]) for record in records),
        "schema_valid_rate": mean(float(record["schema_valid"]) for record in records),
        "deterministic_pass_rate": mean(float(record["deterministic_pass"]) for record in records),
        "abstention_accuracy": (
            mean(float("abstention_decision" not in record["errors"]) for record in valid)
            if valid
            else 0.0
        ),
        "mean_latency_seconds": mean(record["latency_seconds"] for record in records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--language", action="append", choices=LANGUAGES)
    parser.add_argument(
        "--answer-language",
        choices=("fr", "ar", "darija_arab"),
        help="Langue de sortie imposée ; l'expérience validée ici est darija_arab -> ar.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--samples-per-intent-language", type=int)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--num-predict", type=int, default=400)
    args = parser.parse_args()

    languages = set(args.language or LANGUAGES)
    if args.answer_language and len(languages) != 1:
        raise ValueError("--answer-language exige exactement un --language")
    prompts = {row["case_id"]: row for row in read_jsonl(args.prompts)}
    cases = [row for row in read_jsonl(args.cases) if row["language"] in languages]
    if args.samples_per_intent_language is not None:
        selected = []
        group_counts: dict[tuple[str, str], int] = defaultdict(int)
        for case in cases:
            key = (case["intent_id"], case["language"])
            if group_counts[key] < args.samples_per_intent_language:
                selected.append(case)
                group_counts[key] += 1
        cases = selected
    if args.limit is not None:
        cases = cases[: args.limit]

    partial_path = args.output.with_suffix(".partial.jsonl")
    records = read_jsonl(partial_path) if partial_path.exists() else []
    completed = {record["case_id"] for record in records}
    for position, case in enumerate(cases, start=1):
        if case["case_id"] in completed:
            continue
        answer_language = args.answer_language or case["language"]
        prompt = adapt_prompt_language(
            prompts[case["case_id"]]["messages"][0]["content"],
            input_language=case["language"],
            answer_language=answer_language,
        )
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0,
                "seed": 42,
                "num_ctx": args.num_ctx,
                "num_predict": args.num_predict,
            },
            "keep_alive": "10m",
        }
        request = Request(
            args.ollama_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = perf_counter()
        with urlopen(request, timeout=300) as response:
            raw = json.loads(response.read().decode("utf-8"))
        latency = perf_counter() - started
        raw_output = raw["message"]["content"].strip()

        strict_json_valid = False
        schema_valid = False
        answer = None
        answer_before_render = None
        renderer_changed_answer = False
        errors: list[str] = []
        try:
            json.loads(raw_output)
            strict_json_valid = True
            answer_before_render = GroundedAnswer.model_validate_json(raw_output)
            answer = render_answer_from_claims(answer_before_render)
            renderer_changed_answer = answer.answer != answer_before_render.answer
            schema_valid = True
            errors = validate_prediction(
                case,
                answer,
                expected_answer_language=answer_language,
            )
        except (json.JSONDecodeError, ValidationError) as exc:
            errors = [f"invalid_output:{type(exc).__name__}"]

        record = {
            "case_id": case["case_id"],
            "intent_id": case["intent_id"],
            "language": case["language"],
            "answer_language": answer_language,
            "question": case["question"],
            "should_abstain": case["should_abstain"],
            "raw_output": raw_output,
            "answer_before_render": (
                answer_before_render.model_dump() if answer_before_render is not None else None
            ),
            "answer": answer.model_dump() if answer is not None else None,
            "citation_renderer_changed_answer": renderer_changed_answer,
            "strict_json_valid": strict_json_valid,
            "schema_valid": schema_valid,
            "errors": errors,
            "deterministic_pass": schema_valid and not errors,
            "latency_seconds": latency,
            "prompt_tokens": int(raw.get("prompt_eval_count", 0)),
            "completion_tokens": int(raw.get("eval_count", 0)),
        }
        records.append(record)
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path.write_text(
            "".join(
                json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
                for item in records
            ),
            encoding="utf-8",
        )
        print(
            f"[{position}/{len(cases)}] {case['case_id']} {case['language']} "
            f"{'PASS' if record['deterministic_pass'] else 'FAIL'} {latency:.1f}s",
            flush=True,
        )

    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_language[record["language"]].append(record)
    report = {
        "schema_version": "0.1.0",
        "protocol": "frozen_sft_v0.4_prompts_without_constrained_decoding",
        "configuration": {
            "model": args.model,
            "temperature": 0,
            "seed": 42,
            "thinking": False,
            "num_ctx": args.num_ctx,
            "num_predict": args.num_predict,
            "samples_per_intent_language": args.samples_per_intent_language,
            "languages": sorted(languages),
            "answer_language": args.answer_language,
            "test_used_for_tuning": False,
        },
        "summary": summarize(records),
        "by_language": {
            language: summarize(language_records)
            for language, language_records in sorted(by_language.items())
        },
        "cases": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Rapport : {args.output}")


if __name__ == "__main__":
    main()
