"""Évaluation reproductible de la résolution des questions de suivi."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from qralia.api.conversation import ConversationResolver, new_turn, question_dimensions, semantic_concepts


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CASES = ROOT / "data/evaluation/conversation_followups.v0.1.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs/conversation-followups-v0.1.json"


def _fold(text: str) -> str:
    value = unicodedata.normalize("NFKD", text.casefold())
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"[^\w\u0600-\u06ff]+", " ", value).strip()


def load_cases(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(resolver: ConversationResolver, cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for case in cases:
        turns = tuple(new_turn("user", text, case["language"]) for text in case["history"])
        result = resolver.resolve(
            case["question"], turns,
            requested_language=case["language"], provider="onee",
        )
        candidate = result.standalone_question or result.clarification_question or ""
        folded = _fold(candidate)
        if "required_semantic_concepts" in case:
            expected_concepts = set(case["required_semantic_concepts"])
            predicted_concepts = semantic_concepts(candidate)
            concept_recall = (
                len(expected_concepts & predicted_concepts) / len(expected_concepts)
                if expected_concepts else 1.0
            )
        else:
            concepts = [_fold(value) for value in case["required_concepts"]]
            concept_recall = mean(float(value in folded) for value in concepts) if concepts else 1.0
        expected_dimensions = set(case.get("expected_dimensions", []))
        predicted_dimensions = question_dimensions(candidate)
        dimension_correct = not expected_dimensions or expected_dimensions.issubset(predicted_dimensions)
        action_correct = result.action == case["expected_action"]
        history_expected = case["expected_action"] != "standalone"
        history_correct = result.history_used == history_expected
        records.append({
            "case_id": case["case_id"],
            "language": case["language"],
            "expected_action": case["expected_action"],
            "prediction": result.model_dump(),
            "action_correct": action_correct,
            "history_usage_correct": history_correct,
            "required_concept_recall": concept_recall,
            "dimension_correct": dimension_correct,
            "strict_pass": action_correct and history_correct and concept_recall == 1.0 and dimension_correct,
        })

    def metrics(group: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(group),
            "action_accuracy": mean(float(item["action_correct"]) for item in group),
            "history_usage_accuracy": mean(float(item["history_usage_correct"]) for item in group),
            "required_concept_recall": mean(item["required_concept_recall"] for item in group),
            "dimension_accuracy": mean(float(item["dimension_correct"]) for item in group),
            "strict_accuracy": mean(float(item["strict_pass"]) for item in group),
        }

    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_language[record["language"]].append(record)
    return {
        "protocol": "history_to_standalone_question_without_answering",
        "metrics": metrics(records),
        "method_counts": dict(Counter(record["prediction"]["method"] for record in records)),
        "by_language": {language: metrics(group) for language, group in sorted(by_language.items())},
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Évalue la résolution conversationnelle QraLia.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--use-ollama", action="store_true")
    parser.add_argument(
        "--routing-profile", choices=("conservative", "qwen_first"),
        default="conservative",
        help="conservative filtre d'abord les suivis ; qwen_first laisse Qwen décider dès qu'un historique existe.",
    )
    args = parser.parse_args()
    report = evaluate(
        ConversationResolver(use_llm=args.use_ollama, routing_profile=args.routing_profile),
        load_cases(args.cases),
    )
    report["configuration"] = {
        "use_ollama": args.use_ollama,
        "routing_profile": args.routing_profile,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": report["metrics"], "method_counts": report["method_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
