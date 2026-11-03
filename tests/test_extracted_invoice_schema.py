"""Tests for the extracted invoice JSON schema."""
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

SCHEMAS_DIR = Path(__file__).resolve().parent.parent / "schemas"
TESTS_DIR = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def schema() -> dict:
    return json.loads((SCHEMAS_DIR / "extracted_invoice.schema.json").read_text())


@pytest.fixture(scope="module")
def valid_example() -> dict:
    return json.loads((TESTS_DIR / "example_extracted_invoice.json").read_text())


def test_schema_itself_is_valid(schema):
    """The schema must respect the JSON Schema 2020-12 metaschema."""
    Draft202012Validator.check_schema(schema)


def test_valid_example_passes(schema, valid_example):
    Draft202012Validator(schema).validate(valid_example)


def test_missing_required_field_is_rejected(schema, valid_example):
    broken = json.loads(json.dumps(valid_example))
    del broken["fields"]["total_cash"]
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(broken)


def test_confidence_out_of_range_is_rejected(schema, valid_example):
    broken = json.loads(json.dumps(valid_example))
    broken["fields"]["total_cash"]["confidence"] = 1.5
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(broken)


def test_unknown_field_is_rejected(schema, valid_example):
    broken = json.loads(json.dumps(valid_example))
    broken["random_field"] = "hello"
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(broken)