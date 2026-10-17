"""Joint question decomposition and evidence-source routing with local Qwen.

The model only proposes a source hint.  It never receives invoice values and
never answers the question.  Later pipeline stages must confirm the hint with
validated invoice fields or retrieved official chunks before generation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Callable
from urllib.error import URLError
from urllib.request import Request, urlopen

from qralia.rag.question_decomposition import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    ROOT,
    _numbers,
    _read_jsonl,
    _search_form,
)


SOURCE_HINTS = frozenset({"invoice", "official_rag", "both", "unknown"})
DEFAULT_CASES = ROOT / "data/evaluation/question_routing_invoice.v0.1.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs/question-routing-qwen35-v0.1.json"

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
                "required": ["source_span", "standalone_question", "source_hint"],
                "properties": {
                    "source_span": {"type": "string", "minLength": 1},
                    "standalone_question": {"type": "string", "minLength": 1},
                    "source_hint": {"type": "string", "enum": sorted(SOURCE_HINTS)},
                },
            },
        },
    },
}


@dataclass(frozen=True)
class RoutedNeed:
    source_span: str
    standalone_question: str
    source_hint: str


@dataclass(frozen=True)
class RoutingResult:
    is_composite: bool
    needs: tuple[RoutedNeed, ...]
    route: str


@dataclass(frozen=True)
class RoutingRun:
    result: RoutingResult
    valid: bool
    fallback: bool
    attempts: int
    errors: tuple[str, ...]
    latency_ms: float
    model: str


def derive_route(needs: tuple[RoutedNeed, ...]) -> str:
    """Derive the global route in code rather than trusting the model twice."""
    hints = {need.source_hint for need in needs}
    if hints == {"invoice"}:
        return "invoice"
    if hints == {"official_rag"}:
        return "official_rag"
    if hints == {"unknown"}:
        return "unknown"
    if "both" in hints or {"invoice", "official_rag"}.issubset(hints):
        return "both"
    # A partly answerable request must not be presented as fully answerable.
    return "unknown"


def parse_and_validate(raw_output: str, question: str) -> tuple[RoutingResult | None, list[str]]:
    errors: list[str] = []
    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError:
        return None, ["invalid_json"]
    if not isinstance(payload, dict) or set(payload) != {"is_composite", "needs"}:
        return None, ["invalid_top_level_keys"]
    if not isinstance(payload["is_composite"], bool):
        errors.append("invalid_is_composite")
    items = payload.get("needs")
    if not isinstance(items, list) or not 1 <= len(items) <= 3:
        return None, errors + ["invalid_need_count"]

    original = _search_form(question)
    needs: list[RoutedNeed] = []
    expected_keys = {"source_span", "standalone_question", "source_hint"}
    for item in items:
        if not isinstance(item, dict) or set(item) != expected_keys:
            errors.append("invalid_need_shape")
            continue
        span, standalone, hint = (
            item.get("source_span"),
            item.get("standalone_question"),
            item.get("source_hint"),
        )
        if not isinstance(span, str) or not span.strip():
            errors.append("empty_source_span")
            continue
        if not isinstance(standalone, str) or not standalone.strip():
            errors.append("empty_standalone_question")
            continue
        if hint not in SOURCE_HINTS:
            errors.append("invalid_source_hint")
            continue
        if _search_form(span) not in original:
            errors.append("source_span_not_in_question")
        needs.append(RoutedNeed(span.strip(), standalone.strip(), hint))

    if len(needs) != len(items):
        errors.append("unparsed_need")
    if isinstance(payload["is_composite"], bool) and payload["is_composite"] != (len(items) > 1):
        errors.append("composite_count_mismatch")
    generated = " ".join(need.standalone_question for need in needs)
    if not set(_numbers(question)).issubset(set(_numbers(generated))):
        errors.append("number_lost")
    if errors:
        return None, sorted(set(errors))
    frozen = tuple(needs)
    return RoutingResult(bool(payload["is_composite"]), frozen, derive_route(frozen)), []


def build_prompt(question: str, language: str) -> str:
    return f"""Tu analyses une question pour QraLia sans jamais y répondre.

TÂCHE
1. Décompose la question en 1 à 3 besoins indépendants.
2. Pour chaque besoin, propose seulement la source de preuve probable.

SOURCES AUTORISÉES
- invoice: fait imprimé sur la facture active (montant, date d'échéance,
  numéro de facture ou contrat, tarif, HT, taxes, total, consommation).
- official_rag: règle ou procédure générale issue de documents officiels
  (où/comment payer, moyens de paiement, jour férié, contestation, retard).
- both: le même besoin atomique exige à la fois une valeur de la facture et
  une règle officielle pour conclure.
- unknown: ni la facture ni les règles officielles ne suffisent (prévision,
  diagnostic de fraude, cause non documentée, garantie sur le futur).

RÈGLES
- Ne réponds pas et n'invente aucune valeur de facture.
- source_span est une citation continue et exacte de la question.
- Conserve exactement nombres, dates, montants, identifiants et noms propres.
- Deux options d'un même besoin restent un besoin.
- Un montant et un lieu sont deux besoins.
- is_composite vaut true si et seulement si needs contient plusieurs éléments.
- Retourne uniquement le JSON demandé.

EXEMPLES
Question: Quel montant dois-je payer et où le régler ?
Sortie: {{"is_composite":true,"needs":[{{"source_span":"Quel montant dois-je payer","standalone_question":"Quel montant dois-je payer ?","source_hint":"invoice"}},{{"source_span":"où le régler","standalone_question":"Où dois-je régler cette facture ?","source_hint":"official_rag"}}]}}

Question: هل مبلغ الأداء نقدا مطابق لقاعدة التنبر؟
Sortie: {{"is_composite":false,"needs":[{{"source_span":"هل مبلغ الأداء نقدا مطابق لقاعدة التنبر","standalone_question":"هل مبلغ الأداء نقدا مطابق لقاعدة التنبر؟","source_hint":"both"}}]}}

Question: واش الفاتورة الجاية غادي تكون أقل؟
Sortie: {{"is_composite":false,"needs":[{{"source_span":"واش الفاتورة الجاية غادي تكون أقل","standalone_question":"واش الفاتورة الجاية غادي تكون أقل؟","source_hint":"unknown"}}]}}

LANGUE: {language}
QUESTION ORIGINALE: {question}
"""


class OllamaQuestionRouter:
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        url: str = DEFAULT_OLLAMA_URL,
        timeout_seconds: int = 180,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        self.model, self.url = model, url
        self.timeout_seconds, self._opener = timeout_seconds, opener

    def route(self, question: str, language: str) -> RoutingRun:
        question = question.strip()
        if not question:
            raise ValueError("La question ne peut pas être vide.")
        prompt, errors = build_prompt(question, language), []
        started = perf_counter()
        for attempt in range(1, 3):
            if attempt == 2:
                prompt += "\nCORRECTION: recopie les spans exactement et retourne seulement le JSON."
            body = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
                "format": OUTPUT_SCHEMA,
                "options": {"temperature": 0, "seed": 70 + attempt, "num_ctx": 4096, "num_predict": 450},
                "keep_alive": "10m",
            }
            request = Request(self.url, data=json.dumps(body, ensure_ascii=False).encode(), headers={"Content-Type": "application/json"}, method="POST")
            try:
                with self._opener(request, timeout=self.timeout_seconds) as response:
                    raw = json.loads(response.read().decode())
            except URLError as exc:
                raise RuntimeError(f"Ollama est inaccessible ou le modèle {self.model!r} est absent.") from exc
            result, attempt_errors = parse_and_validate(raw.get("message", {}).get("content", ""), question)
            if result is not None:
                return RoutingRun(result, True, False, attempt, (), (perf_counter() - started) * 1000, str(raw.get("model", self.model)))
            errors.extend(attempt_errors)
        fallback_need = RoutedNeed(question, question, "unknown")
        fallback = RoutingResult(False, (fallback_need,), "unknown")
        return RoutingRun(fallback, False, True, 2, tuple(sorted(set(errors))), (perf_counter() - started) * 1000, self.model)


def _match_expected(expected: list[dict[str, str]], predicted: tuple[RoutedNeed, ...]) -> tuple[float, float]:
    remaining = set(range(len(predicted)))
    marker_hits = source_hits = 0
    for gold in expected:
        marker = _search_form(gold["marker"])
        for index in list(remaining):
            need = predicted[index]
            text = _search_form(need.source_span + " " + need.standalone_question)
            if marker in text:
                marker_hits += 1
                source_hits += int(need.source_hint == gold["source_hint"])
                remaining.remove(index)
                break
    count = len(expected) or 1
    return marker_hits / count, source_hits / count


def evaluate(router: OllamaQuestionRouter, cases: list[dict[str, Any]], *, existing_records: list[dict[str, Any]] | None = None, checkpoint_path: Path | None = None) -> dict[str, Any]:
    records = list(existing_records or [])
    completed = {record["case_id"] for record in records}
    for position, case in enumerate(cases, 1):
        if case["case_id"] in completed:
            continue
        run = router.route(case["question"], case["language"])
        count_correct = len(run.result.needs) == case["expected_need_count"]
        route_correct = run.result.route == case["expected_route"]
        marker_coverage, source_accuracy = _match_expected(case["expected_needs"], run.result.needs)
        strict = run.valid and count_correct and route_correct and marker_coverage == 1 and source_accuracy == 1
        record = {**case, "prediction": asdict(run.result), "valid": run.valid, "fallback": run.fallback, "errors": list(run.errors), "count_correct": count_correct, "route_correct": route_correct, "marker_coverage": marker_coverage, "source_accuracy": source_accuracy, "strict_correct": strict, "latency_ms": run.latency_ms}
        records.append(record)
        if checkpoint_path:
            with checkpoint_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[{position:02d}/{len(cases):02d}] {case['case_id']} route={run.result.route}/{case['expected_route']} strict={'PASS' if strict else 'FAIL'}", flush=True)

    def metrics(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {"case_count": len(group), "count_accuracy": mean(float(x["count_correct"]) for x in group), "route_accuracy": mean(float(x["route_correct"]) for x in group), "marker_coverage": mean(x["marker_coverage"] for x in group), "atomic_source_accuracy": mean(x["source_accuracy"] for x in group), "strict_accuracy": mean(float(x["strict_correct"]) for x in group)}
    summary = metrics(records)
    summary.update({"model": router.model, "fallback_rate": mean(float(x["fallback"]) for x in records), "mean_latency_ms": mean(x["latency_ms"] for x in records), "by_language": {language: metrics([x for x in records if x["language"] == language]) for language in sorted({x["language"] for x in records})}, "by_expected_route": {route: metrics([x for x in records if x["expected_route"] == route]) for route in sorted({x["expected_route"] for x in records})}})
    return {"summary": summary, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description="Évalue décomposition + routage facture/RAG avec Qwen3.5.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    cases = _read_jsonl(args.cases)
    if args.limit is not None:
        cases = cases[:args.limit]
    partial = args.output.with_suffix(".partial.jsonl")
    report = evaluate(OllamaQuestionRouter(model=args.model, url=args.ollama_url), cases, existing_records=_read_jsonl(partial) if partial.exists() else [], checkpoint_path=partial)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    partial.unlink(missing_ok=True)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
