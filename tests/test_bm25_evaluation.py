from __future__ import annotations

from qralia.rag.bm25_evaluation import (
    BM25Index,
    abstention_metrics,
    build_parser,
    build_report,
    evaluate_case,
    ranking_metrics,
    tokenize,
)


def test_bm25_cli_accepts_multiple_chunk_files() -> None:
    args = build_parser().parse_args(
        [
            "--chunks",
            "onee.jsonl",
            "--chunks",
            "redal.jsonl",
            "--evaluation",
            "questions.jsonl",
            "--output",
            "report.json",
        ]
    )

    assert [path.name for path in args.chunks] == ["onee.jsonl", "redal.jsonl"]


def _chunk(
    section_id: str,
    text: str,
    audience: str = "residential",
    distributor_id: str = "onee",
    service_zone_id: str = "onee_be_national",
) -> dict:
    return {
        "schema_version": "0.3.0",
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
            "domain": "utilities",
            "provider_id": distributor_id,
            "provider_name": distributor_id.upper(),
            "service_type": "electricity",
            "service_zone_id": service_zone_id,
            "legacy_operator": None,
            "source_role": "operator_primary",
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
        "distributor_id": "onee",
        "service_zone_id": "onee_be_national",
        "relevant_section_ids": relevant,
        "should_abstain": abstain,
        "split": split,
    }


def test_tokenize_handles_accents_and_arabic() -> None:
    assert tokenize("CHÈQUE, règlement!") == ["chèque", "règlement"]
    assert tokenize("أداء الفاتورة") == ["أداء", "الفاتورة"]


def test_darija_lexicon_normalizes_common_latin_spellings() -> None:
    assert tokenize(
        "Factora b chik, cash, virman, tasbi9 w nhar 3tla",
        normalization="darija_lexicon",
    ) == ["facture", "b", "cheque", "especes", "virement", "acompte", "w", "nhar", "ferie"]
    assert tokenize("chèque espèces", normalization="darija_lexicon") == ["cheque", "especes"]
    assert tokenize("lfactura lflous", normalization="darija_lexicon") == ["facture", "especes"]
    assert tokenize("الفاتورة بالشيك", normalization="darija_lexicon") == ["facture", "cheque"]


def test_bm25_ranks_matching_chunk_first() -> None:
    index = BM25Index(
        [
            _chunk("paiement-cheque", "payer la facture par chèque"),
            _chunk("paiement-especes", "payer la facture en espèces"),
        ]
    )
    results = index.search(
        "paiement par chèque",
        audience="residential",
        distributor_id="onee",
        service_zone_id="onee_be_national",
        top_k=2,
    )

    assert results[0]["section_id"] == "paiement-cheque"
    assert results[0]["score"] > results[1]["score"]


def test_darija_lexicon_ranks_chik_as_cheque() -> None:
    index = BM25Index(
        [
            _chunk("paiement-cheque", "payer la facture par chèque"),
            _chunk("paiement-especes", "payer la facture en espèces"),
        ],
        normalization="darija_lexicon",
    )

    results = index.search(
        "n9der nkhlles factora b chik",
        audience="residential",
        distributor_id="onee",
        service_zone_id="onee_be_national",
        top_k=2,
    )

    assert results[0]["section_id"] == "paiement-cheque"


def test_bm25_filters_distributor_before_scoring() -> None:
    index = BM25Index(
        [
            _chunk("onee-cheque", "paiement par chèque ONEE", distributor_id="onee"),
            _chunk("radeef-cheque", "paiement par chèque RADEEF", distributor_id="radeef", service_zone_id="fes_ex_radeef"),
            _chunk("other-zone", "paiement par chèque", distributor_id="onee", service_zone_id="zone_non_autorisee"),
        ]
    )

    results = index.search(
        "paiement par chèque",
        audience="residential",
        distributor_id="onee",
        service_zone_id="onee_be_national",
        top_k=5,
    )

    assert [result["section_id"] for result in results] == ["onee-cheque"]
    assert results[0]["provider_id"] == "onee"
    assert results[0]["service_zone_id"] == "onee_be_national"


def test_bm25_does_not_return_zero_score_chunks() -> None:
    index = BM25Index([_chunk("cheque", "paiement par chèque")])

    assert index.search(
        "قطعت الكهرباء",
        audience="residential",
        distributor_id="onee",
        service_zone_id="onee_be_national",
        top_k=5,
    ) == []


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
