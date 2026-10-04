import hashlib
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/evaluation/sft_questions.v0.4.jsonl"
MANIFEST = ROOT / "data/evaluation/sft_questions.v0.4.manifest.json"
SCHEMA = ROOT / "schemas/sft_question_v0.4.schema.json"
ARABIC = re.compile(r"[\u0600-\u06ff]")


def rows() -> list[dict]:
    return [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().casefold())


def test_v04_counts_schema_hashes_and_target_rates() -> None:
    items = rows()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    validator = Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8")))
    assert manifest["status"] == "human_reviewed"
    assert manifest["human_review"]["accepted_review_groups"] == 124
    assert manifest["counts"] == {"questions": 620, "intents": 31, "train": 360, "validation": 100, "test": 160, "review_groups": 124}
    assert manifest["split_intent_counts"] == {"train": 18, "validation": 5, "test": 8}
    assert manifest["language_counts"] == {"ar": 155, "darija_arab": 155, "darija_latn": 155, "fr": 155}
    assert 0.15 <= manifest["abstention_rate"] <= 0.25
    assert 0.04 <= manifest["retrieval_miss_rate"] <= 0.06
    assert 0.10 <= manifest["multi_chunk_rate"] <= 0.15
    assert manifest["long_context_count"] == 124
    assert manifest["implicit_provider_count"] == 124
    assert manifest["context_size_counts"] == {"4": 620}
    for metadata in manifest["files"].values():
        path = DATA.parent / metadata["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["sha256"]
    for item in items:
        validator.validate(item)


def test_v04_group_split_and_formulation_coverage() -> None:
    items = rows()
    intent_splits: dict[str, set[str]] = defaultdict(set)
    family_splits: dict[str, set[str]] = defaultdict(set)
    positive_chunk_splits: dict[str, set[str]] = defaultdict(set)
    variants: dict[tuple[str, str], set[str]] = defaultdict(set)
    for item in items:
        intent_splits[item["intent_id"]].add(item["split"])
        family_splits[item["semantic_family_id"]].add(item["split"])
        variants[(item["intent_id"], item["language"])].add(item["formulation_kind"])
        for chunk_id in item["gold_chunk_ids"]:
            positive_chunk_splits[chunk_id].add(item["split"])
    assert all(len(value) == 1 for value in intent_splits.values())
    assert all(len(value) == 1 for value in family_splits.values())
    assert all(len(value) == 1 for value in positive_chunk_splits.values())
    expected = {"direct", "indirect", "contextual_long", "provider_implicit", "informal_typed"}
    assert all(value == expected for value in variants.values())
    for split in ("train", "validation", "test"):
        assert {item["formulation_kind"] for item in items if item["split"] == split} == expected


def test_v04_context_contract_and_retrieval_misses() -> None:
    for item in rows():
        gold = set(item["gold_chunk_ids"])
        context = set(item["context_chunk_ids"])
        distractors = set(item["distractor_chunk_ids"])
        assert distractors == context - gold
        assert len(context) == 4
        if item["retrieval_outcome"] == "hit":
            assert gold and gold.issubset(context)
            assert not item["should_abstain"]
            assert item["answerable_in_given_context"]
        elif item["retrieval_outcome"] == "miss":
            assert gold and gold.isdisjoint(context)
            assert item["should_abstain"]
            assert item["answerable_in_corpus"]
            assert not item["answerable_in_given_context"]
        else:
            assert not gold
            assert item["should_abstain"]
            assert not item["answerable_in_given_context"]


def test_v04_language_provider_and_duplicate_guards() -> None:
    items = rows()
    assert not any(ARABIC.search(item["question"]) for item in items if item["language"] == "darija_latn")
    aliases = {
        "onee": ("onee", "office", "loffice", "المكتب"),
        "srm_rsk": ("redal", "ridal", "ريضال"),
        "iam": ("iam", "maroc telecom", "ittisalat", "اتصالات المغرب"),
        "orange_ma": ("orange", "أورنج"),
        "inwi": ("inwi",),
        "anrt": ("anrt", "الوكالة الوطنية"),
    }
    for item in items:
        if item["provider_source"] == "synthetic_invoice_metadata":
            folded = item["question"].casefold()
            assert not any(alias in folded for alias in aliases[item["provider_id"]])
    normalized = [norm(item["question"]) for item in items]
    assert len(normalized) == len(set(normalized))
    for index, left in enumerate(items):
        for right in items[index + 1 :]:
            if left["language"] != right["language"] or left["split"] == right["split"]:
                continue
            similarity = SequenceMatcher(None, norm(left["question"]), norm(right["question"])).ratio()
            assert similarity < 0.85, (left["case_id"], right["case_id"], similarity)


def test_v04_register_is_not_determined_by_language() -> None:
    items = rows()
    by_language = {language: Counter(item["register"] for item in items if item["language"] == language) for language in ("fr", "ar", "darija_latn", "darija_arab")}
    assert by_language["fr"]["informal"] > 0
    assert by_language["fr"]["whatsapp"] > 0
    assert by_language["darija_latn"]["neutral"] > 0
    assert by_language["darija_arab"]["neutral"] > 0


def test_v04_review_groups_cover_every_question() -> None:
    groups = [json.loads(line) for line in (DATA.parent / "sft_questions.v0.4.review.jsonl").read_text(encoding="utf-8").splitlines() if line]
    reviewed = {question["case_id"] for group in groups for question in group["questions"]}
    assert len(groups) == 124
    assert reviewed == {item["case_id"] for item in rows()}
    assert all(len(group["questions"]) == 5 for group in groups)
    assert all(group["human_decision"] == "accept" for group in groups)
