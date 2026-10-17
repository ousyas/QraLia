"""Contrat typé partagé entre extraction, validation et API."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class FieldStatus(str, Enum):
    EXTRACTED = "extracted"
    DERIVED = "derived"
    NOT_PRESENT = "not_present"
    UNREADABLE = "unreadable"


class FieldProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page: int = Field(ge=1)
    bbox: tuple[float, float, float, float] | None = None
    method: str
    derived_from: list[str] | None = None


class ExtractedField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_value: str
    normalized_value: Any = None
    confidence: float = Field(ge=0.0, le=1.0)
    status: FieldStatus
    provenance: FieldProvenance


class ExtractionMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    extracted_at: datetime
    ocr_engine: str
    ocr_engine_version: str | None = None
    source_file: str
    source_type: Literal["synthetic", "user_document"]


class InvoiceFields(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invoice_number: ExtractedField
    contract_number: ExtractedField
    total_cash: ExtractedField
    total_check: ExtractedField
    due_date: ExtractedField
    invoice_date: ExtractedField
    tariff: ExtractedField
    amount_before_tax: ExtractedField
    taxes: ExtractedField
    previous_unpaid: ExtractedField


class ExtractedInvoice(BaseModel):
    """Une extraction canonique ; les valeurs critiques restent immuables."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1.0"] = "0.1.0"
    invoice_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]+$")
    extraction: ExtractionMetadata
    fields: InvoiceFields
    document_confidence: float = Field(ge=0.0, le=1.0)

