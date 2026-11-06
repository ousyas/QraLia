"""Smoke test EvidenceRouter with the real ONEE hybrid + reranker pipeline."""

from __future__ import annotations

import gc
import json
from dataclasses import asdict
from pathlib import Path

from qralia.rag.bm25_evaluation import load_active_chunks
from qralia.rag.evidence_router import EvidencePolicy, EvidenceRouter, LiveRagEvidenceProvider
from qralia.rag.hybrid_evaluation import HybridRetrievalIndex
from qralia.rag.reranker_evaluation import Qwen3CrossEncoderReranker
from qralia.rag.semantic_evaluation import Qwen3LangChainEmbeddings


ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("live-fr-cheque", "À quel ordre faut-il écrire le chèque ?", "paiement-cheque"),
    ("live-fr-cash", "Où payer la facture en espèces ?", "paiement-especes"),
    ("live-ar-holiday", "ماذا يحدث إذا صادف تاريخ الأداء يوم عطلة؟", "date-reglement"),
    ("live-ar-transfer", "هل يمكن الأداء بالتحويل البنكي؟", "paiement-virement"),
    ("live-da-cheque", "فسمية شكون نكتب الشيك ديال الضو؟", "paiement-cheque"),
    ("live-da-cash", "فين نخلص الفاكتورة بالكاش؟", "paiement-especes"),
)
CONTEXT = {
    "domain": "utilities", "provider_id": "onee", "service_type": "electricity",
    "service_zone_id": "onee_be_national", "audience": "residential",
}


class NoInvoiceMatcher:
    def match(self, *_args):
        return []


def main() -> None:
    chunks = load_active_chunks(
        ROOT / "data/processed/onee-regler-facture-residentiel/chunks.jsonl",
        ROOT / "schemas/retrieval_chunk.schema.json",
    )
    embeddings = Qwen3LangChainEmbeddings(device="cuda", batch_size=8, max_seq_length=256)
    index = HybridRetrievalIndex(chunks, embeddings, candidate_k=20)
    # Move embeddings off the GPU after all vectors are built. Query encoding
    # remains on CPU, leaving GPU memory for the cross-encoder.
    embeddings._model.to("cpu")
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass
    reranker = Qwen3CrossEncoderReranker(device="cuda", batch_size=8, max_length=512)
    provider = LiveRagEvidenceProvider(index, reranker, rerank_k=5, top_k=3)
    # Frozen on the existing 104-case reranker calibration, not these cases.
    router = EvidenceRouter(
        NoInvoiceMatcher(),
        EvidencePolicy(rag_threshold=0.4503173828125),
    )
    records = []
    for case_id, question, expected_section in CASES:
        candidates = provider.retrieve(question, CONTEXT)
        decision = router.decide(
            question=question, qwen_hint="official_rag", invoice_fields=[],
            rag_candidates=candidates, context=CONTEXT,
        )
        selected_sections = [index.by_id[chunk_id]["section_id"] for chunk_id in decision.selected_rag_chunks]
        records.append({
            "case_id": case_id, "question": question, "expected_section": expected_section,
            "top_section": index.by_id[candidates[0]["chunk_id"]]["section_id"] if candidates else None,
            "selected_sections": selected_sections, "decision": asdict(decision),
            "candidate_scores": [
                {"section_id": index.by_id[item["chunk_id"]]["section_id"], "reranker_score": item["reranker_score"], "hybrid_confidence": item["hybrid_confidence"]}
                for item in candidates
            ],
            "route_correct": decision.route == "official_rag",
            "top1_correct": bool(candidates) and index.by_id[candidates[0]["chunk_id"]]["section_id"] == expected_section,
            "selected_contains_gold": expected_section in selected_sections,
        })
    report = {
        "pipeline": "bm25_normalized+qwen3_embedding+rrf+qwen3_reranker+evidence_router",
        "rag_threshold_source": "outputs/reranker-all-corpora-qwen3.json calibration (104 cases)",
        "rag_threshold": 0.4503173828125,
        "case_count": len(records),
        "route_accuracy": sum(item["route_correct"] for item in records) / len(records),
        "top1_accuracy": sum(item["top1_correct"] for item in records) / len(records),
        "selected_gold_recall": sum(item["selected_contains_gold"] for item in records) / len(records),
        "records": records,
    }
    output = ROOT / "outputs/evidence-routing-live-rag-v0.2.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("case_count", "route_accuracy", "top1_accuracy", "selected_gold_recall")}, indent=2))


if __name__ == "__main__":
    main()
