"""Marque un dataset SFT comme relu après toutes les décisions humaines."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_DATASET = Path("data/training/qralia_sft_v0.4")
DEFAULT_QUESTION_REVIEW = Path("data/evaluation/sft_questions.v0.4.review.jsonl")
DEFAULT_QUESTION_MANIFEST = Path("data/evaluation/sft_questions.v0.4.manifest.json")


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def unresolved_ids(queue: list[dict]) -> list[str]:
    return [
        item.get("review_group_id", item.get("sample_id", "inconnu"))
        for item in queue
        if item.get("human_decision") != "accept"
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--question-review", type=Path, default=DEFAULT_QUESTION_REVIEW)
    parser.add_argument("--question-manifest", type=Path, default=DEFAULT_QUESTION_MANIFEST)
    parser.add_argument("--reviewer", action="append", required=True)
    args = parser.parse_args()

    queue_path = args.dataset / "review_queue.jsonl"
    answer_queue = read_jsonl(queue_path)
    question_queue = read_jsonl(args.question_review)
    unresolved_answers = unresolved_ids(answer_queue)
    unresolved_questions = unresolved_ids(question_queue)
    if unresolved_answers or unresolved_questions:
        preview = ", ".join((unresolved_questions + unresolved_answers)[:10])
        raise SystemExit(
            "Revue incomplète: "
            f"{len(unresolved_questions)} groupes de questions et "
            f"{len(unresolved_answers)} groupes de réponses non acceptés ({preview}). "
            "Corrigez le builder, reconstruisez le dataset et recommencez la revue."
        )

    reviewers = sorted({name.strip() for name in args.reviewer if name.strip()})
    if not reviewers:
        raise SystemExit("Au moins un nom de relecteur non vide est requis.")
    reviewed_at = datetime.now(timezone.utc).isoformat()
    question_manifest = json.loads(args.question_manifest.read_text(encoding="utf-8"))
    question_manifest["files"]["review"]["sha256"] = hashlib.sha256(
        args.question_review.read_bytes()
    ).hexdigest()
    question_manifest["status"] = "human_reviewed"
    question_manifest["human_review"] = {
        "reviewers": reviewers,
        "reviewed_at": reviewed_at,
        "accepted_review_groups": len(question_queue),
    }
    args.question_manifest.write_text(
        json.dumps(question_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest_path = args.dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["review_queue"]["sha256"] = hashlib.sha256(
        queue_path.read_bytes()
    ).hexdigest()
    manifest["status"] = "human_reviewed"
    manifest["question_bank_manifest_sha256"] = hashlib.sha256(
        args.question_manifest.read_bytes()
    ).hexdigest()
    manifest["human_review"] = {
        "reviewers": reviewers,
        "reviewed_at": reviewed_at,
        "accepted_question_groups": len(question_queue),
        "accepted_answer_groups": len(answer_queue),
        "scope": "Toutes les formulations et toutes les réponses candidates.",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Banque de questions et dataset validés par {', '.join(reviewers)}: "
        f"{manifest_path}"
    )


if __name__ == "__main__":
    main()
