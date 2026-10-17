"""Validation déterministe des relations entre champs extraits."""

from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation

from qralia.domain.extracted_invoice import ExtractedInvoice, FieldStatus

TOLERANCE = Decimal("0.01")


def _as_invoice(extracted: ExtractedInvoice | dict) -> ExtractedInvoice:
    return extracted if isinstance(extracted, ExtractedInvoice) else ExtractedInvoice.model_validate(extracted)


def _money(invoice: ExtractedInvoice, field_name: str) -> Decimal | None:
    field = getattr(invoice.fields, field_name)
    if field.status in {FieldStatus.UNREADABLE, FieldStatus.NOT_PRESENT}:
        return None
    try:
        return Decimal(str(field.normalized_value))
    except (InvalidOperation, TypeError, ValueError):
        return None


def _date(invoice: ExtractedInvoice, field_name: str) -> date | None:
    field = getattr(invoice.fields, field_name)
    if field.status != FieldStatus.EXTRACTED or not field.normalized_value:
        return None
    try:
        return date.fromisoformat(str(field.normalized_value))
    except ValueError:
        return None


def validate_invoice(extracted: ExtractedInvoice | dict) -> dict:
    """Valide uniquement des relations démontrables depuis le document."""
    invoice = _as_invoice(extracted)
    errors: list[str] = []
    warnings: list[str] = []

    total_check = _money(invoice, "total_check")
    total_cash = _money(invoice, "total_cash")
    amount_ht = _money(invoice, "amount_before_tax")
    taxes = _money(invoice, "taxes")
    previous = _money(invoice, "previous_unpaid")
    if None not in (total_check, amount_ht, taxes):
        expected = amount_ht + taxes + (previous or Decimal("0"))
        if abs(total_check - expected) > TOLERANCE:
            errors.append(f"total_check does not equal components ({total_check} != {expected})")
    else:
        warnings.append("arithmetic_check_skipped_missing_amount")
    if total_cash is not None and total_check is not None and total_cash < total_check:
        errors.append("total_cash cannot be lower than total_check")

    due_date = _date(invoice, "due_date")
    invoice_date = _date(invoice, "invoice_date")
    if due_date is None or invoice_date is None:
        warnings.append("date_check_skipped_missing_or_invalid_date")
    elif due_date <= invoice_date:
        errors.append(f"due_date ({due_date}) must be after invoice_date ({invoice_date})")

    critical = ("invoice_number", "total_check", "due_date", "invoice_date", "tariff")
    low_confidence = [name for name in critical if getattr(invoice.fields, name).confidence < 0.70]
    if low_confidence:
        warnings.append("low_confidence:" + ",".join(low_confidence))
    tariff = str(invoice.fields.tariff.normalized_value or "").upper()
    scope_supported = not (tariff.startswith("MT") or "GENERAL" in tariff)
    if not scope_supported:
        warnings.append("outside_mvp_professional_tariff")
    return {
        "valid": not errors,
        "scope_supported": scope_supported,
        "errors": errors,
        "warnings": warnings,
        "critical_fields_complete": not low_confidence,
        "low_confidence_fields": low_confidence,
    }

