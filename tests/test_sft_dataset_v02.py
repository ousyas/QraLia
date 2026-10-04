import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from jsonschema import Draft202012Validator

from qralia.rag.generation_evaluation import GroundedAnswer


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "training" / "qralia_sft_v0.2"
SCHEMA = ROOT / "schemas" / "sft_example_v0.2.schema.json"
ARABIC = re.compile(r"[\u0600-\u06ff]")
PRECISION_CUE = re.compile(r"exact|bddabt|بالضبط|المضبوط", re.IGNORECASE)


def load_jsonl(name: str) -> list[dict]:
    return [json.loads(line) for line in (DATASET / name).read_text(encoding="utf-8").splitlines() if line]


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def test_v02_counts_schema_hashes_and_balance() -> None:
    manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "rejected_split_leakage"
    assert manifest["counts"] == {"questions": 496, "train": 280, "validation": 72, "test": 144, "review_groups": 72}
    assert manifest["language_counts"] == {"ar": 124, "darija_arab": 124, "darija_latn": 124, "fr": 124}
    assert manifest["provider_implicit_count"] >= 150
    for metadata in manifest["files"].values():
        path = DATASET / metadata["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["sha256"]

    validator = Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8")))
    records = load_jsonl("train.jsonl") + load_jsonl("validation.jsonl")
    assert len({row["sample_id"] for row in records}) == len(records)
    for row in records:
        validator.validate(row)
        answer = GroundedAnswer.model_validate_json(row["messages"][1]["content"])
        assert answer.abstain == row["should_abstain"]
        assert answer.language == row["language"]


def test_v02_removes_abstention_shortcuts_and_adds_realistic_styles() -> None:
    questions = load_jsonl("questions.jsonl")
    negatives = [row for row in questions if row["should_abstain"]]
    positives = [row for row in questions if not row["should_abstain"]]
    assert negatives and positives
    assert not any(PRECISION_CUE.search(row["question"]) for row in negatives)
    assert any(re.search(r"\d", row["question"]) for row in positives)
    assert {"natural", "indirect", "contextual", "alias_or_implicit", "asr", "whatsapp"}.issubset({row["style"] for row in questions})
    assert any(not row["provider_mentioned"] for row in positives)
    assert any(not row["provider_mentioned"] for row in negatives)
    assert not any(ARABIC.search(row["question"]) for row in questions if row["language"] == "darija_latn")


def test_v02_splits_are_disjoint_and_test_has_no_assistant_answer() -> None:
    questions = load_jsonl("questions.jsonl")
    by_split = {split: {normalized(row["question"]) for row in questions if row["split"] == split} for split in ("train", "validation", "test")}
    assert by_split["train"].isdisjoint(by_split["validation"])
    assert by_split["train"].isdisjoint(by_split["test"])
    assert by_split["validation"].isdisjoint(by_split["test"])
    test_cases = load_jsonl("test_cases.jsonl")
    test_prompts = load_jsonl("test_prompts.jsonl")
    assert len(test_cases) == len(test_prompts) == 144
    assert Counter(row["language"] for row in test_cases) == Counter({language: 36 for language in ("fr", "ar", "darija_latn", "darija_arab")})
    assert all([message["role"] for message in row["messages"]] == ["user"] for row in test_prompts)


def test_v02_review_groups_expose_every_question() -> None:
    questions = load_jsonl("questions.jsonl")
    groups = load_jsonl("review_queue.jsonl")
    reviewed_ids = {question["question_id"] for group in groups for question in group["questions"]}
    assert len(groups) == 72
    assert reviewed_ids == {row["question_id"] for row in questions}
    assert all(group["human_decision"] is None for group in groups)
