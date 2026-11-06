from qralia.rag.question_decomposition import AtomicQuestion, DecompositionResult
from qralia.rag.question_fusion import SemanticQuestionFusion, compare_report


class FakeEmbeddings:
    """Stable tiny vectors: payment channel, amount, and location axes."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "prototype-channel" in lowered or "agence" in lowered or "ligne" in lowered:
                vectors.append([1.0, 0.0, 0.0])
            elif "prototype-amount" in lowered or "combien" in lowered or "montant" in lowered:
                vectors.append([0.0, 1.0, 0.0])
            elif "prototype-location" in lowered or "où" in lowered:
                vectors.append([0.0, 0.0, 1.0])
            else:
                vectors.append([0.5, 0.5, 0.0])
        return vectors


def _fusion(monkeypatch):
    from qralia.rag import question_fusion

    prototypes = {
        "channels": "prototype-channel",
        "amount": "prototype-amount",
        "location": "prototype-location",
    }
    monkeypatch.setattr(question_fusion, "INTENT_PROTOTYPES", prototypes)
    return SemanticQuestionFusion(FakeEmbeddings(), sentence_similarity_threshold=0.7)


def test_merges_two_options_of_one_payment_need(monkeypatch) -> None:
    decomposition = DecompositionResult(
        True,
        (
            AtomicQuestion("Est-ce que je peux payer", "Est-ce que je peux payer ?"),
            AtomicQuestion("en agence ou en ligne", "En agence ou en ligne"),
        ),
    )
    result = _fusion(monkeypatch).fuse(
        "Est-ce que je peux payer en agence ou en ligne ?", decomposition
    )
    assert result.applied is True
    assert len(result.final.needs) == 1
    assert result.pair_signals[0].reason == "same_need_alternatives"


def test_keeps_amount_and_location_as_two_needs(monkeypatch) -> None:
    decomposition = DecompositionResult(
        True,
        (
            AtomicQuestion("Combien dois-je payer", "Combien dois-je payer ?"),
            AtomicQuestion("où payer", "Où payer ?"),
        ),
    )
    result = _fusion(monkeypatch).fuse(
        "Combien dois-je payer et où payer ?", decomposition
    )
    assert result.applied is False
    assert len(result.final.needs) == 2
    assert result.pair_signals[0].reason == "different_question_dimensions"


def test_keeps_identifier_and_cheque_payment_as_two_needs(monkeypatch) -> None:
    decomposition = DecompositionResult(
        True,
        (
            AtomicQuestion("ما رقم العقد", "ما رقم العقد؟"),
            AtomicQuestion("هل أستطيع الدفع بالشيك", "هل أستطيع الدفع بالشيك؟"),
        ),
    )
    result = _fusion(monkeypatch).fuse(
        "ما رقم العقد وهل أستطيع الدفع بالشيك؟", decomposition
    )
    assert result.applied is False
    assert len(result.final.needs) == 2
    assert result.pair_signals[0].reason == "different_question_dimensions"


def test_invalid_baseline_output_cannot_be_counted_as_fusion_success(monkeypatch) -> None:
    baseline = {
        "records": [
            {
                "case_id": "invalid-001",
                "language": "fr",
                "question": "Combien payer ?",
                "expected_need_count": 1,
                "expected_is_composite": False,
                "prediction": {
                    "is_composite": False,
                    "needs": [
                        {
                            "source_span": "Combien payer",
                            "standalone_question": "Combien payer ?",
                        }
                    ],
                },
                "valid": False,
                "strict_correct": False,
                "marker_coverage": 1.0,
            }
        ]
    }
    report = compare_report(baseline, _fusion(monkeypatch))
    assert report["records"][0]["fusion_strict_correct"] is False
    assert report["summary"]["fusion_strict_accuracy"] == 0.0


def test_conservative_policy_merges_only_explicit_alternatives_with_same_intent(
    monkeypatch,
) -> None:
    fusion = _fusion(monkeypatch)
    fusion.policy = "conservative"
    decomposition = DecompositionResult(
        True,
        (
            AtomicQuestion("payer en agence", "Puis-je payer en agence ?"),
            AtomicQuestion("ou en ligne", "Puis-je payer en ligne ?"),
        ),
    )
    result = fusion.fuse("Puis-je payer en agence ou en ligne ?", decomposition)
    assert result.applied is True
    assert result.pair_signals[0].reason == (
        "conservative_explicit_alternative_same_intent"
    )


def test_conservative_policy_does_not_merge_dependent_fragment_without_alternative(
    monkeypatch,
) -> None:
    fusion = _fusion(monkeypatch)
    fusion.policy = "conservative"
    decomposition = DecompositionResult(
        True,
        (
            AtomicQuestion("Qui contacter", "Qui contacter ?"),
            AtomicQuestion("par quel moyen", "Par quel moyen ?"),
        ),
    )
    result = fusion.fuse("Qui contacter et par quel moyen ?", decomposition)
    assert result.applied is False
    assert len(result.final.needs) == 2
