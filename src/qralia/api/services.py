"""Cas d'usage indépendants de FastAPI pour faciliter les tests et le futur CLI."""

from __future__ import annotations

import re
import unicodedata
import logging
from pathlib import Path
from typing import Callable

from qralia.api.models import CitationView, EvidenceView, FieldView, InvoiceResponse, QuestionResponse, ValidationView
from qralia.api.store import InMemoryInvoiceStore, InvoiceRecord
from qralia.domain.extracted_invoice import ExtractedInvoice
from qralia.domain.invoice_evidence import FIELD_LABELS, build_invoice_evidence
from qralia.rag.bm25_evaluation import BM25Index, load_jsonl
from qralia.validation.validator import validate_invoice


LOGGER = logging.getLogger(__name__)


FIELD_ALIASES = {
    "total_check": ("montant", "total a payer", "combien dois je payer", "combien je dois payer", "شحال خاصني نخلص", "المبلغ", "مجموع"),
    "total_cash": ("en especes", "en liquide", "cash", "بالكاش", "نقدا"),
    "due_date": ("date limite", "date d echeance", "quand dois je payer", "fou9ach nkhlles", "fo9ach nkhlles", "آخر اجل", "تاريخ الاستحقاق"),
    "invoice_date": ("date de facture", "تاريخ الفاتورة"),
    "invoice_number": ("numero de facture", "num facture", "رقم الفاتورة", "نمرة الفاتورة"),
    "contract_number": ("numero de contrat", "num contrat", "رقم العقد", "نمرة الكونطرا"),
    "tariff": ("tarif", "تعريفة", "الطاريف"),
    "amount_before_tax": ("hors taxes", "montant ht", "بلا ضريبة"),
    "taxes": ("taxes", "tva", "الضريبة", "الضرائب"),
    "previous_unpaid": ("impaye precedent", "arrieres", "متأخرات"),
}

RAG_MARKERS = (
    "comment payer", "ou payer", "moyen de paiement", "par cheque", "par virement", "prelevement",
    "agence", "jour ferie", "acompte", "fin nkhlles", "kifach nkhlles", "chik", "virman", "3otla",
    "كيفاش نخلص", "فين نخلص", "بالشيك", "تحويل بنكي", "وكالة", "عطلة",
)


def _fold(text: str) -> str:
    value = unicodedata.normalize("NFKD", text.casefold())
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"[^\w\u0600-\u06ff]+", " ", value).strip()


class InvoiceApplicationService:
    def __init__(
        self,
        *,
        extractor: Callable[[Path], ExtractedInvoice],
        store: InMemoryInvoiceStore | None = None,
    ) -> None:
        self.extractor = extractor
        self.store = store or InMemoryInvoiceStore()

    def ingest(self, path: Path) -> InvoiceRecord:
        invoice = self.extractor(path)
        validation = validate_invoice(invoice)
        evidence = build_invoice_evidence(invoice) if validation["valid"] else []
        return self.store.add(invoice, validation, evidence)

    @staticmethod
    def response(record: InvoiceRecord) -> InvoiceResponse:
        fields = []
        for name, label in FIELD_LABELS.items():
            field = getattr(record.invoice.fields, name)
            fields.append(FieldView(
                name=name, label=label, value=field.normalized_value,
                display_value=field.raw_value or ("" if field.normalized_value is None else str(field.normalized_value)),
                status=field.status.value, confidence=field.confidence, page=field.provenance.page,
            ))
        return InvoiceResponse(
            invoice_id=record.public_id,
            created_at=record.created_at,
            expires_at=record.expires_at,
            document_confidence=record.invoice.document_confidence,
            fields=fields,
            validation=ValidationView.model_validate(record.validation),
        )


class QuestionAnswerService:
    """Baseline sûre : faits exacts de facture + extraits officiels cités."""

    def __init__(self, corpus_path: Path | None, *, ai_enabled: bool = True) -> None:
        self.corpus_path = corpus_path
        self.chunks: list[dict] = []
        self.by_id: dict[str, dict] = {}
        self.index: BM25Index | None = None
        self.ai_enabled = ai_enabled
        self.ai_pipeline = None
        if corpus_path and corpus_path.exists():
            self.chunks = [item for item in load_jsonl(corpus_path) if item["metadata"]["validity_status"] == "active"]
            if self.chunks:
                self.index = BM25Index(self.chunks, normalization="darija_lexicon")
                self.by_id = {item["chunk_id"]: item for item in self.chunks}
        if self.ai_enabled and self.chunks:
            from qralia.api.ai_pipeline import IntegratedAnswerPipeline

            self.ai_pipeline = IntegratedAnswerPipeline(self.chunks)

    @property
    def ready(self) -> bool:
        return self.index is not None

    def _invoice_matches(self, question: str, record: InvoiceRecord):
        folded = _fold(question)
        return [
            item for item in record.evidence
            if any(_fold(alias) in folded for alias in FIELD_ALIASES.get(item.field, ()))
        ]

    def _rag_matches(self, question: str) -> list[dict]:
        if not self.index or not any(_fold(marker) in _fold(question) for marker in RAG_MARKERS):
            return []
        ranked = self.index.search(
            question, audience="residential", distributor_id="onee",
            service_zone_id="onee_be_national", domain="utilities",
            service_type="electricity", top_k=3,
        )
        # La présence d'un marqueur métier explicite est le garde-fou de cette
        # baseline. Un seuil BM25 absolu serait instable lorsque la taille du
        # corpus change, car l'IDF dépend du nombre total de chunks.
        return ranked

    def _deterministic_answer(self, question: str, language: str, record: InvoiceRecord) -> QuestionResponse:
        invoice = self._invoice_matches(question, record)
        rag = self._rag_matches(question) if record.validation.get("scope_supported", False) else []
        route = "both" if invoice and rag else "invoice" if invoice else "official_rag" if rag else "unknown"
        evidence = [
            EvidenceView(source="invoice", evidence_id=item.evidence_id, label=FIELD_LABELS[item.field],
                         value=item.value, confidence=item.confidence)
            for item in invoice
        ]
        citations: list[CitationView] = []
        answer_parts = [item.text for item in invoice]
        for item in rag[:1]:
            chunk = self.by_id[item["chunk_id"]]
            answer_parts.append(chunk["text"])
            evidence.append(EvidenceView(
                source="official_corpus", evidence_id=chunk["chunk_id"],
                label=chunk["metadata"]["title"], confidence=item["score"] / (item["score"] + 1.0),
            ))
            citations.append(CitationView(
                chunk_id=chunk["chunk_id"], title=chunk["metadata"]["title"],
                url=chunk["metadata"]["canonical_url"],
            ))
        abstained = route == "unknown"
        warnings = ["Réponse extractive déterministe : génération LLM désactivée dans cette première API."]
        if not record.validation.get("scope_supported", False):
            warnings.append("Document professionnel hors du périmètre résidentiel : les règles RAG sont désactivées.")
        if language not in {"auto", "fr"}:
            warnings.append("La traduction multilingue n'est pas encore activée ; les preuves sont rendues en français.")
        answer = " ".join(answer_parts) if answer_parts else "Je ne dispose pas de preuves suffisantes pour répondre sans deviner."
        scores = [item.confidence for item in evidence]
        return QuestionResponse(
            answer=answer, route=route, language="fr", abstained=abstained,
            confidence=min(scores) if scores else 0.0, evidence=evidence,
            citations=citations, warnings=warnings,
        )

    def answer(self, question: str, language: str, record: InvoiceRecord) -> QuestionResponse:
        if self.ai_pipeline is not None:
            try:
                return self.ai_pipeline.answer(question, language, record)
            except Exception:
                # Le contenu de la facture n'est jamais journalisé.
                LOGGER.exception("Le pipeline IA local a échoué ; fallback déterministe.")
                fallback = self._deterministic_answer(question, language, record)
                fallback.warnings.append(
                    "Pipeline RAG/LLM indisponible : réponse déterministe de secours utilisée."
                )
                return fallback
        return self._deterministic_answer(question, language, record)

