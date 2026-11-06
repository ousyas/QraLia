from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "data/evaluation/darija_script_pairs.v0.1.jsonl"
SCHEMA_PATH = ROOT / "schemas/retrieval_evaluation_case.schema.json"


def _load_cases() -> list[dict]:
    return [
        json.loads(line)
        for line in CASES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_darija_script_pairs_are_schema_valid_and_balanced() -> None:
    cases = _load_cases()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)

    assert [error for case in cases for error in validator.iter_errors(case)] == []
    assert len(cases) == 48
    assert Counter(case["language"] for case in cases) == {
        "darija_latn": 24,
        "darija_arab": 24,
    }
    assert Counter(case["split"] for case in cases) == {
        "calibration": 24,
        "test": 24,
    }
    assert Counter(case["should_abstain"] for case in cases) == {
        False: 32,
        True: 16,
    }


def test_each_script_pair_has_identical_targets_and_route() -> None:
    pairs: dict[str, list[dict]] = defaultdict(list)
    for case in _load_cases():
        pairs[case["pair_id"]].append(case)

    assert len(pairs) == 24
    shared_fields = (
        "audience",
        "distributor_id",
        "service_zone_id",
        "relevant_section_ids",
        "should_abstain",
        "split",
    )
    for variants in pairs.values():
        assert len(variants) == 2
        assert {case["language"] for case in variants} == {
            "darija_latn",
            "darija_arab",
        }
        for field in shared_fields:
            assert variants[0][field] == variants[1][field]
