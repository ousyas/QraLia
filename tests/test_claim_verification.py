import pytest
from pydantic import ValidationError

from qralia.rag.claim_verification import (
    ClaimVerdict,
    build_claim_judge_prompt,
    online_deterministic_gate,
)
from qralia.rag.generation_evaluation import AnswerClaim, GroundedAnswer


def answer(text: str, citations: list[str]) -> GroundedAnswer:
    return GroundedAnswer(
        answer=text,
        claims=[AnswerClaim(text=text, citations=citations)],
        abstain=False,
        abstention_reason=None,
        language="ar",
    )


def test_online_gate_accepts_number_copied_from_cited_passage() -> None:
    result = online_deterministic_gate(
        answer("لديك 7 أيام للاعتراض.", ["chunk-a"]),
        {"chunk-a": "Le client dispose de 7 jours pour faire opposition."},
        expected_language="ar",
    )
    assert result.passed


def test_online_gate_rejects_changed_number() -> None:
    result = online_deterministic_gate(
        answer("لديك 10 أيام للاعتراض.", ["chunk-a"]),
        {"chunk-a": "Le client dispose de 7 jours pour faire opposition."},
        expected_language="ar",
    )
    assert not result.passed
    assert "unsupported_numbers:['10']" in result.errors[0]


def test_claim_acceptance_is_derived_from_all_checks() -> None:
    with pytest.raises(ValidationError):
        ClaimVerdict(
            claim_index=0,
            answers_question=False,
            entailed_by_citations=True,
            provider_consistent=True,
            critical_values_preserved=True,
            accepted=True,
            reason="hors sujet",
        )


def test_judge_prompt_contains_no_gold_answer_label() -> None:
    candidate = answer("لديك 7 أيام للاعتراض.", ["chunk-a"])
    prompt = build_claim_judge_prompt([{
        "case_id": "case-1",
        "provider_id": "iam",
        "question": "شحال عندي باش نعترض؟",
        "answer": candidate,
        "contexts_by_id": {"chunk-a": "Le délai est de 7 jours."},
    }])
    assert "expected_should_abstain" not in prompt
    assert "gold_chunk_ids" not in prompt
    assert "chunk-a" in prompt
