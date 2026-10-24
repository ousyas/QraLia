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


class ConversationCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: Literal["auto", "fr", "ar", "darija_arab", "darija_latn"] = "auto"


class ConversationMessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=2, max_length=500)
    language: Literal["auto", "fr", "ar", "darija_arab", "darija_latn"] = "auto"


class ConversationTurnView(BaseModel):
    message_id: str
    role: Literal["user", "assistant"]
    text: str
    language: str
    created_at: datetime
    standalone_question: str | None = None
    route: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class ConversationResponse(BaseModel):
    conversation_id: str
    invoice_id: str
    preferred_language: str
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    turns: list[ConversationTurnView]


class ConversationResolutionView(BaseModel):
    action: Literal["standalone", "rewrite", "clarify"]
    standalone_question: str | None = None
    clarification_question: str | None = None
    language: str
    history_used: bool
    resolved_references: list[str]
    method: Literal["direct", "qwen", "deterministic_fallback"]


class ConversationMessageResponse(BaseModel):
    conversation_id: str
    user_message_id: str
    assistant_message_id: str
    resolution: ConversationResolutionView
    answer: QuestionResponse | None = None
    clarification_question: str | None = None

