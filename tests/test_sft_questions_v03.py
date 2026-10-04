import hashlib
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/evaluation/sft_questions.v0.3.jsonl"
MANIFEST = ROOT / "data/evaluation/sft_questions.v0.3.manifest.json"
SCHEMA = ROOT / "schemas/sft_question_v0.3.schema.json"
ARABIC = re.compile(r"[\u0600-\u06ff]")


def rows() -> list[dict]:
    return [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line]


def test_v03_counts_schema_and_hash() -> None:
    items = rows()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    validator = Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8")))
    assert manifest["counts"] == {"questions": 124, "intents": 31, "positive_intents": 25, "negative_intents": 6}
    assert manifest["split_counts"] == {"test": 20, "train": 84, "validation": 20}
    assert manifest["language_counts"] == {"ar": 31, "darija_arab": 31, "darija_latn": 31, "fr": 31}
    assert manifest["positive_chunk_coverage"] == 25
    assert 0.15 <= manifest["abstention_rate"] <= 0.25
    assert hashlib.sha256(DATA.read_bytes()).hexdigest() == manifest["file"]["sha256"]
    for item in items:
        validator.validate(item)


def test_v03_groups_intents_and_positive_chunks_without_split_leakage() -> None:
    items = rows()
    intent_splits: dict[str, set[str]] = defaultdict(set)
    family_splits: dict[str, set[str]] = defaultdict(set)
    chunk_splits: dict[str, set[str]] = defaultdict(set)
    intent_languages: dict[str, set[str]] = defaultdict(set)
    for item in items:
        intent_splits[item["intent_id"]].add(item["split"])
        family_splits[item["semantic_family_id"]].add(item["split"])
        intent_languages[item["intent_id"]].add(item["language"])
        if item["source_chunk_id"]:
            chunk_splits[item["source_chunk_id"]].add(item["split"])
    assert all(len(splits) == 1 for splits in intent_splits.values())
    assert all(len(splits) == 1 for splits in family_splits.values())
    assert all(len(splits) == 1 for splits in chunk_splits.values())
    assert all(languages == {"fr", "ar", "darija_latn", "darija_arab"} for languages in intent_languages.values())


def test_v03_has_no_fake_asr_or_cross_split_near_duplicates() -> None:
    items = rows()
    assert all(item["channel"] == "typed" for item in items)
    assert not any(ARABIC.search(item["question"]) for item in items if item["language"] == "darija_latn")
    for index, left in enumerate(items):
        for right in items[index + 1 :]:
            if left["language"] != right["language"] or left["split"] == right["split"]:
                continue
            similarity = SequenceMatcher(None, left["question"].casefold(), right["question"].casefold()).ratio()
            assert similarity < 0.85, (left["case_id"], right["case_id"], similarity)


def test_v03_context_and_corpus_answerability_are_separate() -> None:
    items = rows()
    positives = [item for item in items if not item["should_abstain"]]
    negatives = [item for item in items if item["should_abstain"]]
    assert all(item["answerable_in_given_context"] and item["answerable_in_corpus"] for item in positives)
    assert all(not item["answerable_in_given_context"] for item in negatives)
    assert all(item["source_chunk_id"] is None for item in negatives)
    assert all(item["context_chunk_ids"] for item in negatives)
    assert len(negatives) == 24
