import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "data/evaluation/conversation_followups.v0.2.jsonl"
MANIFEST = ROOT / "data/evaluation/conversation_followups.v0.2.manifest.json"


def _rows() -> list[dict]:
    return [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line]


def test_conversation_holdout_v02_is_balanced_and_frozen() -> None:
    rows = _rows()
    assert len(rows) == 80
    assert len({row["case_id"] for row in rows}) == 80
    assert len({row["scenario_family"] for row in rows}) == 20
    assert Counter(row["language"] for row in rows) == Counter({
        "fr": 20, "ar": 20, "darija_arab": 20, "darija_latn": 20,
    })
    assert Counter(row["expected_action"] for row in rows) == Counter({
        "standalone": 20, "rewrite": 48, "clarify": 12,
    })


def test_conversation_holdout_v02_provenance_and_manifest() -> None:
    rows = _rows()
    assert all(row["split"] == "holdout" for row in rows)
    assert all(row["review_status"] == "candidate_human_review" for row in rows)
    assert all(row["source_type"] == "synthetic_conversation_candidate" for row in rows)
    assert all(row["question"].strip() for row in rows)
    assert all(isinstance(row["history"], list) for row in rows)

    raw = CASES.read_bytes()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["case_count"] == 80
    assert manifest["scenario_family_count"] == 20
    assert manifest["sha256"] == hashlib.sha256(raw).hexdigest()
