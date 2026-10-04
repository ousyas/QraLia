import hashlib
import json
import re
from pathlib import Path

from jsonschema import Draft202012Validator

from qralia.rag.generation_evaluation import GroundedAnswer, load_generation_cases


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "training" / "qralia_sft_v0.1"
SFT_SCHEMA = ROOT / "schemas" / "sft_example.schema.json"
GENERATION_SCHEMA = ROOT / "schemas" / "generation_evaluation_case.schema.json"
ARABIC = re.compile(r"[\u0600-\u06ff]")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def test_sft_dataset_counts_schema_and_hashes() -> None:
    manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
    train = load_jsonl(DATASET / "train.jsonl")
    validation = load_jsonl(DATASET / "validation.jsonl")
    validator = Draft202012Validator(json.loads(SFT_SCHEMA.read_text(encoding="utf-8")))

    assert manifest["status"] == "candidate_human_review"
    assert manifest["counts"] == {
        "train": 478,
        "validation": 100,
        "test": 40,
        "review_queue": 52,
    }
    assert len(train) == 478
    assert len(validation) == 100
    assert len({record["sample_id"] for record in train + validation}) == 578
    for record in train + validation:
        validator.validate(record)
    for metadata in manifest["files"].values():
        path = DATASET / metadata["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["sha256"]


def test_sft_targets_are_structured_grounded_and_script_consistent() -> None:
    records = load_jsonl(DATASET / "train.jsonl") + load_jsonl(DATASET / "validation.jsonl")
    for record in records:
        answer = GroundedAnswer.model_validate_json(record["messages"][1]["content"])
        assert answer.language == record["language"]
        assert answer.abstain == record["should_abstain"]
        if record["should_abstain"]:
            assert answer.claims == []
            assert len(record["source_chunk_ids"]) == 1
        else:
            assert answer.claims
            assert {
                citation for claim in answer.claims for citation in claim.citations
            }.issubset(set(record["source_chunk_ids"]))
            for value in record["critical_values"]:
                assert value in answer.answer
        if record["language"] == "darija_latn":
            assert not ARABIC.search(record["question"])
            assert not ARABIC.search(answer.answer)


def test_sft_train_validation_and_frozen_test_do_not_copy_questions() -> None:
    train = load_jsonl(DATASET / "train.jsonl")
    validation = load_jsonl(DATASET / "validation.jsonl")
    test_cases = load_generation_cases(DATASET / "test_cases.jsonl", GENERATION_SCHEMA)
    test_prompts = load_jsonl(DATASET / "test_prompts.jsonl")

    train_questions = {normalized(record["question"]) for record in train}
    validation_questions = {normalized(record["question"]) for record in validation}
    test_questions = {normalized(case["question"]) for case in test_cases}
    assert train_questions.isdisjoint(validation_questions)
    assert (train_questions | validation_questions).isdisjoint(test_questions)
    assert len(test_cases) == 40
    assert all(case["split"] == "test" for case in test_cases)
    assert len(test_prompts) == 40
    assert all([message["role"] for message in item["messages"]] == ["user"] for item in test_prompts)


def test_sft_uses_only_calibration_source_intents() -> None:
    base_cases = load_generation_cases(
        ROOT / "data" / "evaluation" / "generation_questions.v0.1.jsonl",
        GENERATION_SCHEMA,
    )
    calibration_ids = {case["case_id"] for case in base_cases if case["split"] == "calibration"}
    records = load_jsonl(DATASET / "train.jsonl") + load_jsonl(DATASET / "validation.jsonl")
    used_ids = {record["source_case_id"] for record in records if record["source_case_id"]}

    assert used_ids
    assert used_ids.issubset(calibration_ids)
