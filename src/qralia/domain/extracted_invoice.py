"""Contrat typé partagé entre extraction, validation et API."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from qralia.domain.providers import ProviderId


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


def _not_present() -> ExtractedField:
    return ExtractedField(
        raw_value="", normalized_value=None, confidence=0, status=FieldStatus.NOT_PRESENT,
        provenance=FieldProvenance(page=1, method="not_applicable"),
    )


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
    # Champs propres aux abonnements télécom ; absents d'une facture ONEE.
    plan_name: ExtractedField = Field(default_factory=_not_present)
    line_number: ExtractedField = Field(default_factory=_not_present)
    out_of_plan_amount: ExtractedField = Field(default_factory=_not_present)


class ExtractedInvoice(BaseModel):
    """Une extraction canonique ; les valeurs critiques restent immuables."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1.0", "0.2.0"] = "0.2.0"
    invoice_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]+$")
    # 0.1.0 ne connaissait que l'ONEE : ces valeurs par défaut gardent les
    # anciennes extractions valides.
    provider_id: ProviderId = "onee"
    extraction: ExtractionMetadata
    fields: InvoiceFields
    document_confidence: float = Field(ge=0.0, le=1.0)

