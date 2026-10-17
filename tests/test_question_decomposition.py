import json

from qralia.rag.question_decomposition import (
    OllamaQuestionDecomposer,
    parse_and_validate,
)


class FakeResponse:
    def __init__(self, content: str):
        self.content = content

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return json.dumps({"model": "fake", "message": {"content": self.content}}).encode()


def test_valid_composite_decomposition() -> None:
    raw = json.dumps(
        {
            "is_composite": True,
            "needs": [
                {"source_span": "Quel est le montant", "standalone_question": "Quel est le montant ?"},
                {"source_span": "où payer", "standalone_question": "Où payer ?"},
            ],
        }
    )
    result, errors = parse_and_validate(raw, "Quel est le montant et où payer ?")
    assert errors == []
    assert result is not None
    assert result.is_composite is True
    assert len(result.needs) == 2


def test_rejects_span_invented_by_model() -> None:
    raw = json.dumps(
        {
            "is_composite": False,
            "needs": [
                {"source_span": "quelle agence ONEE", "standalone_question": "Quelle agence ONEE ?"}
            ],
        }
    )
    result, errors = parse_and_validate(raw, "Où puis-je payer ?")
    assert result is None
    assert "source_span_not_in_question" in errors


def test_rejects_lost_critical_number() -> None:
    raw = json.dumps(
        {
            "is_composite": False,
            "needs": [
                {"source_span": "Quand payer", "standalone_question": "Quand payer ?"}
            ],
        }
    )
    result, errors = parse_and_validate(raw, "Quand payer les 450,20 DH ?")
    assert result is None
    assert "number_lost" in errors


def test_two_invalid_outputs_trigger_safe_single_question_fallback() -> None:
    responses = iter([FakeResponse("not-json"), FakeResponse("still-not-json")])
    decomposer = OllamaQuestionDecomposer(opener=lambda *_args, **_kwargs: next(responses))
    run = decomposer.decompose("شحال خاصني نخلص وفين نخلص؟", "darija_arab")
    assert run.valid is False
    assert run.fallback is True
    assert run.result.is_composite is False
    assert run.result.needs[0].standalone_question == "شحال خاصني نخلص وفين نخلص؟"
