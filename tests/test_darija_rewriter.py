import json

from qralia.rag.darija_rewriter import (
    AtlasDarijaRewriter,
    OllamaAtlasDarijaRewriter,
    build_rewrite_prompt,
    candidate_quality_issues,
    lock_values,
    restore_values,
    values_to_protect,
)


def test_ollama_atlas_backend_returns_message_content() -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps(
                {"message": {"content": "تقدر تخلص فـ ZXQVAL0QXZ."}}
            ).encode("utf-8")

    rewriter = OllamaAtlasDarijaRewriter(
        model_id="atlas-test",
        opener=lambda *_args, **_kwargs: FakeResponse(),
    )

    rewritten, metadata = rewriter.rewrite("الأداء متاح 24h/24.", ["24h/24"])

    assert rewritten == "تقدر تخلص فـ 24h/24."
    assert not metadata["fallback"]


def test_lock_and_restore_preserve_critical_values_and_numbers() -> None:
    text = "خلص 15 jours قبل الموعد واتصل بـ 5555."
    values = values_to_protect(text, ["15 jours", "5555"])
    locked, mapping = lock_values(text, values)

    assert "15 jours" not in locked
    assert "5555" not in locked
    assert restore_values(locked, mapping) == text


def test_rewriter_accepts_candidate_that_preserves_markers() -> None:
    def backend(prompt: str) -> str:
        assert "ZXQVAL0QXZ" in prompt
        return "تقدر تعيط لـ ZXQVAL0QXZ باش تأكد العملية."

    rewritten, metadata = AtlasDarijaRewriter(backend=backend).rewrite(
        "عيط لـ 5555 لتأكيد العملية.", ["5555"]
    )

    assert rewritten == "تقدر تعيط لـ 5555 باش تأكد العملية."
    assert metadata["changed"]
    assert not metadata["fallback"]


def test_rewriter_falls_back_when_marker_is_lost() -> None:
    original = "عيط لـ 5555 لتأكيد العملية."
    rewritten, metadata = AtlasDarijaRewriter(backend=lambda _prompt: "عيط باش تأكد.").rewrite(
        original, ["5555"]
    )

    assert rewritten == original
    assert metadata["fallback"]


def test_observation_mode_keeps_unsafe_candidate_and_records_would_fallback() -> None:
    original = "عيط لـ 5555 لتأكيد العملية."
    rewriter = AtlasDarijaRewriter(
        backend=lambda _prompt: "عيط باش تأكد.",
        observation_mode=True,
    )

    rewritten, metadata = rewriter.rewrite(original, ["5555"])

    assert rewritten == "عيط باش تأكد."
    assert not metadata["fallback"]
    assert metadata["would_fallback"]
    assert metadata["unsafe_observation"]
    assert metadata["raw_candidate"] == "عيط باش تأكد."


def test_observation_mode_records_empty_candidate_but_keeps_valid_claim_text() -> None:
    original = "عيط لـ 5555 لتأكيد العملية."
    rewriter = AtlasDarijaRewriter(
        backend=lambda _prompt: "",
        observation_mode=True,
    )

    rewritten, metadata = rewriter.rewrite(original, ["5555"])

    assert rewritten == original
    assert metadata["raw_candidate"] == ""
    assert metadata["would_fallback"]
    assert not metadata["observation_applied"]


def test_grounded_few_shot_prompt_teaches_marker_and_target_script() -> None:
    prompt = build_rewrite_prompt(
        "Vous avez ZXQVAL0QXZ jours.",
        target_language="darija_latn",
        prompt_profile="grounded_few_shot",
    )

    assert "3ndek ZXQVAL0QXZ iyam" in prompt
    assert "كل رمز بحال ZXQVAL0QXZ" in prompt
    assert "بالحروف اللاتينية" in prompt


def test_grounded_compact_prompt_is_short_and_targeted_for_latin_script() -> None:
    prompt = build_rewrite_prompt(
        "Vous avez ZXQVAL0QXZ jours.",
        target_language="darija_latn",
        prompt_profile="grounded_compact",
    )

    assert "lettres latines" in prompt
    assert "3ndek ZXQVAL0QXZ iyam" in prompt
    assert "TEXTE:" in prompt
    assert len(prompt) < 500


def test_latin_rewrite_is_not_rejected_for_using_latin_script() -> None:
    rewriter = AtlasDarijaRewriter(
        backend=lambda _prompt: "3ndek ZXQVAL0QXZ iyam bach tkhelles.",
        prompt_profile="grounded_few_shot",
    )

    rewritten, metadata = rewriter.rewrite(
        "Vous avez 10 jours pour payer.",
        ["10"],
        target_language="darija_latn",
    )

    assert rewritten == "3ndek 10 iyam bach tkhelles."
    assert not metadata["fallback"]


def test_quality_gate_rejects_non_arabic_repetitive_candidate() -> None:
    candidate = "Ca va bien. ative ative ative ative ative ative."

    issues = candidate_quality_issues("خلص الفاتورة.", candidate)

    assert "excessive_repetition" in issues
    assert "insufficient_arabic_script" in issues
