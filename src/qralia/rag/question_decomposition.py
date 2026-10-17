"""Semantic decomposition of a user utterance into atomic questions.

This module deliberately stops before routing or retrieval. Qwen proposes one
to three independent needs, while deterministic validation prevents malformed
or invented spans from silently entering the rest of the pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Callable
from urllib.error import URLError
from urllib.request import Request, urlopen


DEFAULT_MODEL = "qwen3.5:4b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CASES = ROOT / "data/evaluation/question_decomposition.v0.1.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs/question-decomposition-qwen35-v0.1.json"

NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)*")

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["is_composite", "needs"],
    "properties": {
        "is_composite": {"type": "boolean"},
        "needs": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["source_span", "standalone_question"],
                "properties": {
                    "source_span": {"type": "string", "minLength": 1},
                    "standalone_question": {"type": "string", "minLength": 1},
                },
            },
        },
    },
}


@dataclass(frozen=True)
class AtomicQuestion:
    source_span: str
    standalone_question: str


@dataclass(frozen=True)
class DecompositionResult:
    is_composite: bool
    needs: tuple[AtomicQuestion, ...]


@dataclass(frozen=True)
class DecompositionRun:
    result: DecompositionResult
    valid: bool
    fallback: bool
    attempts: int
    errors: tuple[str, ...]
    latency_ms: float
    model: str


def _search_form(text: str) -> str:
    """Normalize only for validation; never replace the user-visible text."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[؟?!.,;:،؛]+", " ", text)
    return " ".join(text.split())


def _numbers(text: str) -> list[str]:
    return NUMBER_PATTERN.findall(unicodedata.normalize("NFKC", text))


def parse_and_validate(raw_output: str, question: str) -> tuple[DecompositionResult | None, list[str]]:
    """Parse Qwen output and reject structural or factual drift."""
    errors: list[str] = []
    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError:
        return None, ["invalid_json"]

    if not isinstance(payload, dict) or set(payload) != {"is_composite", "needs"}:
        return None, ["invalid_top_level_keys"]
    if not isinstance(payload["is_composite"], bool):
        errors.append("invalid_is_composite")
    needs = payload.get("needs")
    if not isinstance(needs, list) or not 1 <= len(needs) <= 3:
        return None, errors + ["invalid_need_count"]

    original = _search_form(question)
    parsed_needs: list[AtomicQuestion] = []
    for item in needs:
        if not isinstance(item, dict) or set(item) != {"source_span", "standalone_question"}:
            errors.append("invalid_need_shape")
            continue
        source_span = item.get("source_span")
        standalone = item.get("standalone_question")
        if not isinstance(source_span, str) or not source_span.strip():
            errors.append("empty_source_span")
            continue
        if not isinstance(standalone, str) or not standalone.strip():
            errors.append("empty_standalone_question")
            continue
        if _search_form(source_span) not in original:
            errors.append("source_span_not_in_question")
        parsed_needs.append(
            AtomicQuestion(source_span=source_span.strip(), standalone_question=standalone.strip())
        )

    if len(parsed_needs) != len(needs):
        errors.append("unparsed_need")
    if isinstance(payload["is_composite"], bool):
        if payload["is_composite"] != (len(needs) > 1):
            errors.append("composite_count_mismatch")

    # Numbers are critical literals. They may be inherited by only one atomic
    # question, but none may disappear from the decomposition as a whole.
    generated_text = " ".join(need.standalone_question for need in parsed_needs)
    if not set(_numbers(question)).issubset(set(_numbers(generated_text))):
        errors.append("number_lost")

    if errors:
        return None, sorted(set(errors))
    return DecompositionResult(bool(payload["is_composite"]), tuple(parsed_needs)), []


def build_prompt(question: str, language: str) -> str:
    """Few-shot prompt: identify needs only, never answer or route them."""
    return f"""Tu es un décomposeur sémantique de questions pour QraLia.

TÂCHE
Détermine si la question contient une ou plusieurs demandes indépendantes.
Retourne entre 1 et 3 besoins atomiques. Ne réponds jamais à la question et ne
décide pas si elle concerne une facture ou une règle officielle.

RÈGLES
- source_span doit être une citation continue et exacte de la question originale.
- standalone_question reformule seulement si nécessaire pour rendre le besoin autonome.
- Conserve exactement tous les nombres, dates, montants, identifiants et noms propres.
- Deux options d'un même besoin ne sont PAS deux questions.
- Une liste comme « carte et chèque » reste un seul besoin.
- Un montant et un lieu, ou une date et une méthode, sont deux besoins.
- N'invente aucun contexte absent.
- is_composite vaut true si et seulement si needs contient au moins deux éléments.
- Retourne uniquement le JSON conforme au schéma demandé.

EXEMPLES
Question: Quel est le montant et où puis-je payer ?
Sortie: {{"is_composite":true,"needs":[{{"source_span":"Quel est le montant","standalone_question":"Quel est le montant ?"}},{{"source_span":"où puis-je payer","standalone_question":"Où puis-je payer ?"}}]}}

Question: هل يمكنني الدفع نقداً أو بالشيك؟
Sortie: {{"is_composite":false,"needs":[{{"source_span":"هل يمكنني الدفع نقداً أو بالشيك","standalone_question":"هل يمكنني الدفع نقداً أو بالشيك؟"}}]}}

Question: شحال خاصني نخلص وفين نقدر نخلص؟
Sortie: {{"is_composite":true,"needs":[{{"source_span":"شحال خاصني نخلص","standalone_question":"شحال خاصني نخلص؟"}},{{"source_span":"فين نقدر نخلص","standalone_question":"فين نقدر نخلص؟"}}]}}

LANGUE D'ENTRÉE: {language}
QUESTION ORIGINALE: {question}
"""


class OllamaQuestionDecomposer:
    """Small local Qwen client with strict validation and a safe fallback."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        url: str = DEFAULT_OLLAMA_URL,
        timeout_seconds: int = 180,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        self.model = model
        self.url = url
        self.timeout_seconds = timeout_seconds
        self._opener = opener

    def decompose(self, question: str, language: str) -> DecompositionRun:
        question = question.strip()
        if not question:
            raise ValueError("La question ne peut pas être vide.")

        prompt = build_prompt(question, language)
        all_errors: list[str] = []
        started = perf_counter()
        for attempt in range(1, 3):
            if attempt == 2:
                prompt += (
                    "\nCORRECTION: la sortie précédente a été rejetée. "
                    "Recopie chaque source_span exactement depuis QUESTION ORIGINALE, "
                    "conserve tous les nombres et retourne uniquement le JSON."
                )
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
                "format": OUTPUT_SCHEMA,
                "options": {
                    "temperature": 0,
                    "seed": 42 + attempt - 1,
                    "num_ctx": 4096,
                    "num_predict": 350,
                },
                "keep_alive": "10m",
            }
            request = Request(
                self.url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with self._opener(request, timeout=self.timeout_seconds) as response:
                    raw = json.loads(response.read().decode("utf-8"))
            except URLError as exc:
                raise RuntimeError(
                    f"Ollama est inaccessible ou le modèle {self.model!r} est absent."
                ) from exc
            content = raw.get("message", {}).get("content", "")
            result, errors = parse_and_validate(content, question)
            if result is not None:
                return DecompositionRun(
                    result=result,
                    valid=True,
                    fallback=False,
                    attempts=attempt,
                    errors=(),
                    latency_ms=(perf_counter() - started) * 1000,
                    model=str(raw.get("model", self.model)),
                )
            all_errors.extend(errors)

        fallback = DecompositionResult(
            is_composite=False,
            needs=(AtomicQuestion(question, question),),
        )
        return DecompositionRun(
            result=fallback,
            valid=False,
            fallback=True,
            attempts=2,
            errors=tuple(sorted(set(all_errors))),
            latency_ms=(perf_counter() - started) * 1000,
            model=self.model,
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _marker_coverage(markers: list[str], needs: tuple[AtomicQuestion, ...]) -> float:
    """Require each gold marker to occur in a different predicted need."""
    remaining = set(range(len(needs)))
    matched = 0
    for marker in markers:
        normalized_marker = _search_form(marker)
        for index in list(remaining):
            text = _search_form(needs[index].source_span + " " + needs[index].standalone_question)
            if normalized_marker in text:
                matched += 1
                remaining.remove(index)
                break
    return matched / len(markers) if markers else 1.0


def evaluate(
    decomposer: OllamaQuestionDecomposer,
    cases: list[dict[str, Any]],
    *,
    existing_records: list[dict[str, Any]] | None = None,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate cases and optionally checkpoint each expensive Ollama call."""
    records = list(existing_records or [])
    completed = {record["case_id"] for record in records}
    for position, case in enumerate(cases, start=1):
        if case["case_id"] in completed:
            continue
        run = decomposer.decompose(case["question"], case["language"])
        count_correct = len(run.result.needs) == case["expected_need_count"]
        composite_correct = run.result.is_composite == case["expected_is_composite"]
        coverage = _marker_coverage(case["expected_markers"], run.result.needs)
        strict_correct = count_correct and composite_correct and coverage == 1.0 and run.valid
        record = {
                "case_id": case["case_id"],
                "language": case["language"],
                "question": case["question"],
                "expected_need_count": case["expected_need_count"],
                "expected_is_composite": case["expected_is_composite"],
                "expected_markers": case["expected_markers"],
                "prediction": asdict(run.result),
                "valid": run.valid,
                "fallback": run.fallback,
                "errors": list(run.errors),
                "count_correct": count_correct,
                "composite_correct": composite_correct,
                "marker_coverage": coverage,
                "strict_correct": strict_correct,
                "latency_ms": run.latency_ms,
            }
        records.append(record)
        if checkpoint_path is not None:
            with checkpoint_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"[{position:02d}/{len(cases):02d}] {case['case_id']} "
            f"needs={len(run.result.needs)} expected={case['expected_need_count']} "
            f"strict={'PASS' if strict_correct else 'FAIL'}",
            flush=True,
        )

    by_language: dict[str, dict[str, float | int]] = {}
    for language in sorted({record["language"] for record in records}):
        group = [record for record in records if record["language"] == language]
        by_language[language] = {
            "case_count": len(group),
            "count_accuracy": mean(float(item["count_correct"]) for item in group),
            "composite_accuracy": mean(float(item["composite_correct"]) for item in group),
            "marker_coverage": mean(item["marker_coverage"] for item in group),
            "strict_accuracy": mean(float(item["strict_correct"]) for item in group),
        }
    summary = {
        "case_count": len(records),
        "model": decomposer.model,
        "count_accuracy": mean(float(item["count_correct"]) for item in records),
        "composite_accuracy": mean(float(item["composite_correct"]) for item in records),
        "marker_coverage": mean(item["marker_coverage"] for item in records),
        "strict_accuracy": mean(float(item["strict_correct"]) for item in records),
        "fallback_rate": mean(float(item["fallback"]) for item in records),
        "mean_latency_ms": mean(item["latency_ms"] for item in records),
        "by_language": by_language,
    }
    return {"summary": summary, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description="Évalue la décomposition sémantique Qwen3.5.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    cases = _read_jsonl(args.cases)
    if args.limit is not None:
        cases = cases[: args.limit]
    partial_path = args.output.with_suffix(".partial.jsonl")
    existing_records = _read_jsonl(partial_path) if partial_path.exists() else []
    report = evaluate(
        OllamaQuestionDecomposer(model=args.model, url=args.ollama_url),
        cases,
        existing_records=existing_records,
        checkpoint_path=partial_path,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    partial_path.unlink(missing_ok=True)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Rapport écrit dans {args.output}")


if __name__ == "__main__":
    main()
