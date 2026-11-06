"""Stockage éphémère, borné et thread-safe pour le premier MVP."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import RLock
from uuid import uuid4

from qralia.domain.extracted_invoice import ExtractedInvoice
from qralia.domain.invoice_evidence import InvoiceEvidence


@dataclass(frozen=True)
class InvoiceRecord:
    public_id: str
    invoice: ExtractedInvoice
    validation: dict
    evidence: tuple[InvoiceEvidence, ...]
    created_at: datetime
    expires_at: datetime


class InMemoryInvoiceStore:
    """Ne conserve ni octets du document ni fichier temporaire après l'OCR."""

    def __init__(self, *, ttl_minutes: int = 30, max_documents: int = 25) -> None:
        self.ttl = timedelta(minutes=ttl_minutes)
        self.max_documents = max_documents
        self._records: dict[str, InvoiceRecord] = {}
        self._lock = RLock()

    def _purge_expired(self, now: datetime) -> None:
        expired = [key for key, value in self._records.items() if value.expires_at <= now]
        for key in expired:
            del self._records[key]

    def add(self, invoice: ExtractedInvoice, validation: dict, evidence: list[InvoiceEvidence]) -> InvoiceRecord:
        now = datetime.now(timezone.utc)
        with self._lock:
            self._purge_expired(now)
            if len(self._records) >= self.max_documents:
                oldest = min(self._records.values(), key=lambda item: item.created_at)
                del self._records[oldest.public_id]
            public_id = str(uuid4())
            record = InvoiceRecord(public_id, invoice, validation, tuple(evidence), now, now + self.ttl)
            self._records[public_id] = record
            return record

    def get(self, public_id: str) -> InvoiceRecord | None:
        now = datetime.now(timezone.utc)
        with self._lock:
            self._purge_expired(now)
            return self._records.get(public_id)

    def delete(self, public_id: str) -> bool:
        with self._lock:
            return self._records.pop(public_id, None) is not None

