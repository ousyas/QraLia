import json

from qralia.rag.question_routing import (
    OllamaQuestionRouter,
    RoutedNeed,
    derive_route,
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


def need(hint: str) -> RoutedNeed:
    return RoutedNeed("question", "Question ?", hint)


def test_global_route_is_derived_deterministically() -> None:
    assert derive_route((need("invoice"),)) == "invoice"
    assert derive_route((need("official_rag"),)) == "official_rag"
    assert derive_route((need("invoice"), need("official_rag"))) == "both"
    assert derive_route((need("both"),)) == "both"
    assert derive_route((need("unknown"),)) == "unknown"
    assert derive_route((need("invoice"), need("unknown"))) == "unknown"


def test_valid_mixed_question() -> None:
    raw = json.dumps({
        "is_composite": True,
        "needs": [
            {"source_span": "Combien dois-je payer", "standalone_question": "Combien dois-je payer ?", "source_hint": "invoice"},
            {"source_span": "où le régler", "standalone_question": "Où régler la facture ?", "source_hint": "official_rag"},
        ],
    })
    result, errors = parse_and_validate(raw, "Combien dois-je payer et où le régler ?")
    assert errors == []
    assert result is not None
    assert result.route == "both"
    assert len(result.needs) == 2


def test_rejects_invalid_source_hint() -> None:
    raw = json.dumps({
        "is_composite": False,
        "needs": [{"source_span": "Quel montant", "standalone_question": "Quel montant ?", "source_hint": "internet"}],
    })
    result, errors = parse_and_validate(raw, "Quel montant ?")
    assert result is None
    assert "invalid_source_hint" in errors


def test_fallback_is_unknown_after_two_invalid_outputs() -> None:
    responses = iter([FakeResponse("bad"), FakeResponse("bad again")])
    router = OllamaQuestionRouter(opener=lambda *_args, **_kwargs: next(responses))
    run = router.route("شحال خاصني نخلص؟", "darija_arab")
    assert run.fallback is True
    assert run.valid is False
    assert run.result.route == "unknown"
    assert run.result.needs[0].source_hint == "unknown"


def test_critical_number_must_be_preserved() -> None:
    raw = json.dumps({
        "is_composite": False,
        "needs": [{"source_span": "الشهر القادم", "standalone_question": "ماذا سيحدث الشهر القادم؟", "source_hint": "unknown"}],
    })
    result, errors = parse_and_validate(raw, "هل سأدفع 300 درهم الشهر القادم؟")
    assert result is None
    assert "number_lost" in errors
