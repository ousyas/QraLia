"""Construction des preuves temporaires issues d'une facture validée."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from qralia.domain.extracted_invoice import ExtractedInvoice, FieldStatus


class InvoiceEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    source_type: Literal["invoice_fact"] = "invoice_fact"
    invoice_id: str
    field: str
    value: Any
    display_value: str
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    page: int
    bbox: tuple[float, float, float, float] | None = None
    derivation: list[str] | None = None


FIELD_LABELS = {
    "invoice_number": "Numéro de facture",
    "contract_number": "Numéro de contrat",
    "total_cash": "Montant à payer en espèces",
    "total_check": "Montant total à régler",
    "due_date": "Date limite de paiement",
    "invoice_date": "Date de facture",
    "tariff": "Tarif",
    "amount_before_tax": "Montant hors taxes",
    "taxes": "Taxes",
    "previous_unpaid": "Impayé précédent",
}


def build_invoice_evidence(
    invoice: ExtractedInvoice | dict,
    *,
    minimum_confidence: float = 0.70,
) -> list[InvoiceEvidence]:
    """Exclut toujours les champs absents, illisibles ou trop incertains."""
    parsed = invoice if isinstance(invoice, ExtractedInvoice) else ExtractedInvoice.model_validate(invoice)
    evidence: list[InvoiceEvidence] = []
    for field_name, label in FIELD_LABELS.items():
        field = getattr(parsed.fields, field_name)
        if field.status not in {FieldStatus.EXTRACTED, FieldStatus.DERIVED}:
            continue
        if field.confidence < minimum_confidence or field.normalized_value is None:
            continue
        display_value = field.raw_value or str(field.normalized_value)
        evidence.append(
            InvoiceEvidence(
                evidence_id=f"invoice.{field_name}",
                invoice_id=parsed.invoice_id,
                field=field_name,
                value=field.normalized_value,
                display_value=display_value,
                text=f"{label} : {display_value}",
                confidence=field.confidence,
                page=field.provenance.page,
                bbox=field.provenance.bbox,
                derivation=field.provenance.derived_from,
            )
        )
    return evidence

