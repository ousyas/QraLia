import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from jsonschema import Draft202012Validator

from qralia.rag.generation_evaluation import GroundedAnswer


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "training" / "qralia_sft_v0.4"
SCHEMA = ROOT / "schemas" / "sft_example_v0.4.schema.json"
TARGET_CACHE = ROOT / "data" / "training" / "sft_chunk_targets.v0.4.jsonl"
ARABIC = re.compile(r"[\u0600-\u06ff]")


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_v04_counts_schema_and_hashes() -> None:
    manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "human_reviewed"
    assert manifest["human_review"]["accepted_question_groups"] == 124
    assert manifest["human_review"]["accepted_answer_groups"] == 168
    assert manifest["counts"] == {
        "train": 360,
        "validation": 100,
        "test": 160,
        "review_groups": 168,
    }
    assert manifest["train_abstention_count"] == 100
    assert manifest["validation_abstention_count"] == 24

    for metadata in manifest["files"].values():
        path = DATASET / metadata["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["sha256"]

    validator = Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8")))
    records = load_jsonl(DATASET / "train.jsonl") + load_jsonl(
        DATASET / "validation.jsonl"
    )
    assert len(records) == 460
    assert len({row["sample_id"] for row in records}) == len(records)
    for row in records:
        validator.validate(row)


def test_v04_answers_are_grounded_and_keep_critical_values() -> None:
    records = load_jsonl(DATASET / "train.jsonl") + load_jsonl(
        DATASET / "validation.jsonl"
    )
    for row in records:
        answer = GroundedAnswer.model_validate_json(row["messages"][1]["content"])
        assert answer.abstain == row["should_abstain"]
        assert answer.language == row["language"]

        if answer.abstain:
            assert not answer.claims
            assert answer.abstention_reason
        else:
            assert answer.claims
            assert answer.abstention_reason is None
            cited = {
                citation
                for claim in answer.claims
                for citation in claim.citations
            }
            assert cited
            assert cited.issubset(set(row["gold_chunk_ids"]))
            assert cited.issubset(set(row["context_chunk_ids"]))
            assert all(f"[{citation}]" in answer.answer for citation in cited)

        assert all(value in answer.answer for value in row["critical_values"])
        if row["language"] == "darija_latn":
            assert not ARABIC.search(answer.answer)


def test_v04_retrieval_misses_abstain_without_leaking_gold() -> None:
    records = load_jsonl(DATASET / "train.jsonl") + load_jsonl(
        DATASET / "validation.jsonl"
    )
    misses = [row for row in records if row["retrieval_outcome"] == "miss"]
    assert misses
    for row in misses:
        assert row["should_abstain"]
        assert set(row["gold_chunk_ids"]).isdisjoint(row["context_chunk_ids"])


def test_v04_test_is_frozen_and_has_no_assistant_answer() -> None:
    test_cases = load_jsonl(DATASET / "test_cases.jsonl")
    test_prompts = load_jsonl(DATASET / "test_prompts.jsonl")
    assert len(test_cases) == len(test_prompts) == 160
    assert Counter(row["language"] for row in test_cases) == Counter(
        {language: 40 for language in ("fr", "ar", "darija_latn", "darija_arab")}
    )
    assert {row["case_id"] for row in test_cases} == {
        row["case_id"] for row in test_prompts
    }
    assert all(
        [message["role"] for message in row["messages"]] == ["user"]
        for row in test_prompts
    )


def test_v04_target_cache_covers_all_gold_chunks_in_four_languages() -> None:
    targets = load_jsonl(TARGET_CACHE)
    all_questions = (
        load_jsonl(DATASET / "train.jsonl")
        + load_jsonl(DATASET / "validation.jsonl")
        + load_jsonl(DATASET / "test_cases.jsonl")
    )
    required = {
        chunk_id
        for row in all_questions
        for chunk_id in row["gold_chunk_ids"]
    }
    assert len(targets) == len(required) == 28
    assert {row["chunk_id"] for row in targets} == required
    for row in targets:
        assert set(row["claims"]) == {"fr", "ar", "darija_latn", "darija_arab"}
        assert all(row["claims"][language] for language in row["claims"])
        assert not any(ARABIC.search(claim) for claim in row["claims"]["darija_latn"])
        assert row["review_status"] == "candidate_human_review"


def test_v04_review_queue_exposes_every_train_and_validation_sample() -> None:
    records = load_jsonl(DATASET / "train.jsonl") + load_jsonl(
        DATASET / "validation.jsonl"
    )
    groups = load_jsonl(DATASET / "review_queue.jsonl")
    reviewed_ids = {
        question["sample_id"]
        for group in groups
        for question in group["questions"]
    }
    assert len(groups) == 168
    assert reviewed_ids == {row["sample_id"] for row in records}
    assert all(group["human_decision"] == "accept" for group in groups)
