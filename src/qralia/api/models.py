"""Schémas publics de l'API ; ils ne dépendent pas des détails du frontend."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str = "qralia-api"
    version: str


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    components: dict[str, bool]
    details: dict[str, str] = Field(default_factory=dict)


class FieldView(BaseModel):
    name: str
    label: str
    value: Any = None
    display_value: str
    status: str
    confidence: float
    page: int


class ValidationView(BaseModel):
    valid: bool
    scope_supported: bool
    critical_fields_complete: bool
    errors: list[str]
    warnings: list[str]
    low_confidence_fields: list[str]


class InvoiceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invoice_id: str
    created_at: datetime
    expires_at: datetime
    document_confidence: float
    fields: list[FieldView]
    validation: ValidationView


class QuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=2, max_length=500)
    language: Literal["auto", "fr", "ar", "darija_arab", "darija_latn"] = "auto"


class EvidenceView(BaseModel):
    source: Literal["invoice", "official_corpus"]
    evidence_id: str
    label: str
    value: Any = None
    confidence: float


class CitationView(BaseModel):
    chunk_id: str
    title: str
    url: str


class QuestionResponse(BaseModel):
    answer: str
    route: Literal["invoice", "official_rag", "both", "unknown"]
    language: str
    abstained: bool
    confidence: float
    evidence: list[EvidenceView]
    citations: list[CitationView]
    warnings: list[str]

