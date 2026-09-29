from __future__ import annotations

from qralia.rag.bm25_evaluation import (
    BM25Index,
    abstention_metrics,
    build_report,
    evaluate_case,
    ranking_metrics,
    tokenize,
)


def _chunk(section_id: str, text: str, audience: str = "residential") -> dict:
    return {
        "schema_version": "0.1.0",
        "chunk_id": f"doc--{section_id}",
        "document_id": "doc",
        "source_id": "source",
        "section_id": section_id,
        "text": text,
        "metadata": {
            "title": "Facture",
            "publisher": "ONEE",
            "canonical_url": "https://example.test/source",
            "languages": ["fr"],
            "audiences": [audience],
            "validity_status": "active",
            "verified_at": "2026-09-29",
            "risk_level": "medium",
            "heading_path": ["Résidentiel", section_id],
        },
    }


def _case(case_id: str, question: str, relevant: list[str], *, abstain: bool, split: str) -> dict:
    return {
        "schema_version": "0.1.0",
        "case_id": case_id,
        "question": question,
        "language": "fr",
        "category": "hard_negative" if abstain else "direct",
        "audience": "residential",
        "relevant_section_ids": relevant,
        "should_abstain": abstain,
        "split": split,
    }


def test_tokenize_handles_accents_and_arabic() -> None:
    assert tokenize("CHÈQUE, règlement!") == ["chèque", "règlement"]
    assert tokenize("أداء الفاتورة") == ["أداء", "الفاتورة"]


def test_bm25_ranks_matching_chunk_first() -> None:
    index = BM25Index(
        [
            _chunk("paiement-cheque", "payer la facture par chèque"),
            _chunk("paiement-especes", "payer la facture en espèces"),
        ]
    )
    results = index.search("paiement par chèque", audience="residential", top_k=2)

    assert results[0]["section_id"] == "paiement-cheque"
    assert results[0]["score"] > results[1]["score"]


def test_bm25_does_not_return_zero_score_chunks() -> None:
    index = BM25Index([_chunk("cheque", "paiement par chèque")])

    assert index.search("قطعت الكهرباء", audience="residential", top_k=5) == []


def test_ranking_metrics_distinguish_recall_and_binary_recall() -> None:
    index = BM25Index(
        [
            _chunk("a", "chèque"),
            _chunk("b", "virement"),
            _chunk("c", "autre"),
        ]
    )
    case = _case("retrieval-001", "chèque", ["a", "b"], abstain=False, split="test")
    record = evaluate_case(index, case, top_k=1)
    metrics = ranking_metrics([record])

    assert metrics["recall_at_1"] == 0.5
    assert metrics["binary_recall_at_1"] == 1.0
    assert metrics["mrr"] == 1.0


def test_report_calibrates_only_on_calibration_split() -> None:
    index = BM25Index(
        [
            _chunk("cheque", "paiement par chèque"),
            _chunk("cash", "paiement en espèces"),
        ]
    )
    cases = [
        _case("retrieval-001", "chèque", ["cheque"], abstain=False, split="calibration"),
        _case("retrieval-002", "coupure", [], abstain=True, split="calibration"),
        _case("retrieval-003", "espèces", ["cash"], abstain=False, split="test"),
        _case("retrieval-004", "tarif", [], abstain=True, split="test"),
    ]

    report = build_report(index, cases, top_k=2)
    metrics = report["test"]["abstention_metrics"]

    assert report["calibration"]["case_count"] == 2
    assert report["test"]["case_count"] == 2
    assert metrics["true_positive"] == 1
    assert metrics["true_negative"] == 1
    assert abstention_metrics(report["cases"], report["calibration"]["selected_threshold"])["case_count"] == 4
