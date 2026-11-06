"""Tests de la mémoire courte et des questions conversationnelles."""

import json
from pathlib import Path

from qralia.api.conversation import (
    ConversationResolver,
    InMemoryConversationStore,
    ResolutionResult,
    new_turn,
    validate_semantic_rewrite,
)
from qralia.rag.conversation_evaluation import evaluate, load_cases


ROOT = Path(__file__).resolve().parents[1]


def test_followup_dataset_is_balanced_and_passes_deterministic_baseline() -> None:
    cases = load_cases(ROOT / "data/evaluation/conversation_followups.v0.1.jsonl")
    assert len(cases) == 20
    assert {case["language"] for case in cases} == {"fr", "ar", "darija_arab", "darija_latn"}
    assert all(sum(case["language"] == language for case in cases) == 5 for language in {"fr", "ar", "darija_arab", "darija_latn"})

    report = evaluate(ConversationResolver(use_llm=False), cases)
    assert report["metrics"]["action_accuracy"] == 1.0
    assert report["metrics"]["history_usage_accuracy"] == 1.0
    assert report["metrics"]["required_concept_recall"] == 1.0
    assert report["metrics"]["strict_accuracy"] == 1.0


def test_followup_resolves_pronoun_from_user_history() -> None:
    resolver = ConversationResolver(use_llm=False)
    history = (new_turn("user", "Wach n9der nkhlles had factora b chik?", "darija_latn"),)
    result = resolver.resolve("Fin n3tih?", history, requested_language="darija_latn", provider="onee")
    assert result.action == "rewrite"
    assert result.standalone_question == "Fin khasni n9dem chik bach nkhlles facture ONEE?"
    assert result.history_used is True
    assert result.method == "deterministic_fallback"


def test_ambiguous_followup_requests_clarification() -> None:
    resolver = ConversationResolver(use_llm=False)
    history = (new_turn("user", "Je parle du chèque et du prélèvement.", "fr"),)
    result = resolver.resolve("Et ça ?", history, requested_language="fr", provider="onee")
    assert result.action == "clarify"
    assert result.standalone_question is None
    assert result.clarification_question


def test_conversation_store_is_invoice_scoped_and_cascade_ready() -> None:
    store = InMemoryConversationStore()
    first = store.create("invoice-a", "fr")
    second = store.create("invoice-b", "ar")
    store.append(first.public_id, new_turn("user", "Combien dois-je payer ?", "fr"))
    assert len(store.get(first.public_id).turns) == 1  # type: ignore[union-attr]
    assert store.delete_for_invoice("invoice-a") == 1
    assert store.get(first.public_id) is None
    assert store.get(second.public_id) is not None


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")


def _ollama_opener(content: dict):
    def open_request(request, timeout):
        assert request.full_url.endswith("/api/chat")
        assert timeout == 60
        return _FakeResponse({"model": "qwen3.5:4b", "message": {"content": json.dumps(content, ensure_ascii=False)}})
    return open_request


def test_qwen_rewrite_is_used_after_deterministic_validation() -> None:
    resolver = ConversationResolver(
        use_llm=True,
        opener=_ollama_opener({
            "action": "rewrite",
            "standalone_question": "Où dois-je déposer le chèque pour payer la facture ONEE ?",
            "clarification_question": None,
            "language": "fr",
            "history_used": True,
            "resolved_references": ["le → le chèque"],
        }),
    )
    history = (new_turn("user", "Puis-je payer cette facture par chèque ?", "fr"),)
    result = resolver.resolve("Et où le déposer ?", history, requested_language="fr", provider="onee")
    assert result.action == "rewrite"
    assert result.method == "qwen"
    assert "chèque" in result.standalone_question  # type: ignore[operator]


def test_qwen_first_profile_sends_long_autonomous_candidate_to_qwen() -> None:
    resolver = ConversationResolver(
        use_llm=True,
        routing_profile="qwen_first",
        opener=_ollama_opener({
            "action": "standalone",
            "standalone_question": "Maintenant, quel est le numéro de contrat sur ma facture ?",
            "clarification_question": None,
            "language": "fr",
            "history_used": False,
            "resolved_references": [],
        }),
    )
    history = (new_turn("user", "Comment payer par virement ?", "fr"),)
    result = resolver.resolve(
        "Maintenant, quel est le numéro de contrat sur ma facture ?",
        history, requested_language="fr", provider="onee",
    )
    assert result.action == "standalone"
    assert result.method == "qwen"


def test_qwen_rewrite_inventing_number_is_rejected_and_falls_back() -> None:
    resolver = ConversationResolver(
        use_llm=True,
        opener=_ollama_opener({
            "action": "rewrite",
            "standalone_question": "Où déposer le chèque de 500 DH pour payer la facture ONEE ?",
            "clarification_question": None,
            "language": "fr",
            "history_used": True,
            "resolved_references": ["le → le chèque"],
        }),
    )
    history = (new_turn("user", "Puis-je payer cette facture par chèque ?", "fr"),)
    result = resolver.resolve("Et où le déposer ?", history, requested_language="fr", provider="onee")
    assert result.method == "deterministic_fallback"
    assert "500" not in (result.standalone_question or "")


def test_qwen_timeout_uses_safe_fallback_instead_of_failing_request() -> None:
    def timeout_opener(request, timeout):
        raise TimeoutError("local model did not answer")

    resolver = ConversationResolver(use_llm=True, opener=timeout_opener)
    history = (new_turn("user", "Puis-je payer cette facture par chèque ?", "fr"),)
    result = resolver.resolve("Et où le déposer ?", history, requested_language="fr", provider="onee")
    assert result.action == "rewrite"
    assert result.method == "deterministic_fallback"


def test_qwen_incoherent_history_flag_is_rejected() -> None:
    resolver = ConversationResolver(
        use_llm=True,
        opener=_ollama_opener({
            "action": "rewrite",
            "standalone_question": "Où dois-je déposer le chèque pour payer cette facture ?",
            "clarification_question": None,
            "language": "fr",
            "history_used": False,
            "resolved_references": ["le → le chèque"],
        }),
    )
    history = (new_turn("user", "Puis-je payer cette facture par chèque ?", "fr"),)
    result = resolver.resolve("Et où le déposer ?", history, requested_language="fr", provider="onee")
    assert result.method == "deterministic_fallback"


def test_semantic_validator_preserves_referenced_entity_and_question_dimension() -> None:
    history = (new_turn("user", "Puis-je payer cette facture par chèque ?", "fr"),)
    valid = ResolutionResult(
        action="rewrite",
        standalone_question="Où dois-je déposer le chèque pour payer cette facture ?",
        language="fr", history_used=True,
    )
    assert validate_semantic_rewrite(valid, "Et où le déposer ?", history).valid

    missing_entity = valid.model_copy(update={"standalone_question": "Où dois-je payer cette facture ?"})
    report = validate_semantic_rewrite(missing_entity, "Et où le déposer ?", history)
    assert not report.valid
    assert any("missing_required_concepts:cheque" in error for error in report.errors)

    changed_dimension = valid.model_copy(update={"standalone_question": "Comment payer cette facture par chèque ?"})
    report = validate_semantic_rewrite(changed_dimension, "Et où le déposer ?", history)
    assert not report.valid
    assert "question_dimension_changed" in report.errors


def test_semantic_validator_rejects_new_payment_method_and_lost_darija_register() -> None:
    history = (new_turn("user", "Wach n9der nkhlles had factora b chik?", "darija_latn"),)
    invented = ResolutionResult(
        action="rewrite",
        standalone_question="Fin khasni ndir virman bach nkhlles facture ONEE?",
        language="darija_latn", history_used=True,
    )
    report = validate_semantic_rewrite(invented, "Fin n3tih?", history)
    assert not report.valid
    assert any("introduced_critical_concepts:virement" in error for error in report.errors)

    translated = invented.model_copy(update={"standalone_question": "Où dois-je déposer le chèque ?"})
    report = validate_semantic_rewrite(translated, "Fin n3tih?", history)
    assert not report.valid
    assert "darija_latn_register_lost" in report.errors


def test_ambiguous_payment_reference_is_not_resolved_to_cheque() -> None:
    history = (new_turn("user", "Je parle du chèque et du prélèvement.", "fr"),)
    candidate = ResolutionResult(
        action="rewrite",
        standalone_question="Où dois-je déposer le chèque ?",
        language="fr", history_used=True,
    )
    report = validate_semantic_rewrite(candidate, "Et où le faire ?", history)
    assert not report.valid
    assert "ambiguous_reference_requires_clarification" in report.errors

    resolver = ConversationResolver(use_llm=False)
    result = resolver.resolve("Et où le déposer ?", history, requested_language="fr", provider="onee")
    assert result.action == "clarify"
