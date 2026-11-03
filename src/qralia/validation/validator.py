"""Validation déterministe des relations entre champs extraits."""

from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation

from qralia.domain.extracted_invoice import ExtractedInvoice, FieldStatus
from qralia.domain.providers import PROVIDERS, critical_fields

TOLERANCE = Decimal("0.01")
# Valeurs observées sur les factures synthétiques ; elles peuvent varier, donc
# un écart produit un avertissement et non une erreur. Le taux de TVA dépend du
# fournisseur (registre des profils).
STAMP_DUTY = Decimal("0.33")
VAT_TOLERANCE = Decimal("1.00")


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
    elif total_cash is not None and total_check is not None and abs(total_cash - total_check - STAMP_DUTY) > TOLERANCE:
        warnings.append(f"stamp_duty_unexpected ({total_cash - total_check} != {STAMP_DUTY})")
    if amount_ht is not None and taxes is not None:
        expected_taxes = (amount_ht * PROVIDERS[invoice.provider_id].vat_rate).quantize(TOLERANCE)
        if abs(taxes - expected_taxes) > VAT_TOLERANCE:
            warnings.append(f"taxes_differ_from_vat_rate ({taxes} != {expected_taxes})")

    due_date = _date(invoice, "due_date")
    invoice_date = _date(invoice, "invoice_date")
    if due_date is None or invoice_date is None:
        warnings.append("date_check_skipped_missing_or_invalid_date")
    elif due_date <= invoice_date:
        errors.append(f"due_date ({due_date}) must be after invoice_date ({invoice_date})")

    critical = critical_fields(invoice.provider_id)
    low_confidence = [name for name in critical if getattr(invoice.fields, name).confidence < 0.70]
    if low_confidence:
        warnings.append("low_confidence:" + ",".join(low_confidence))
    tariff = str(invoice.fields.tariff.normalized_value or "").upper()
    # Seule l'ONEE distingue ici tarifs résidentiels et professionnels.
    scope_supported = invoice.provider_id != "onee" or not (tariff.startswith("MT") or "GENERAL" in tariff)
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

