"""Registre des fournisseurs de factures pris en charge.

Chaque profil relie un document lu à son contexte de routage RAG. Les
identifiants reprennent exactement ceux des chunks officiels, afin qu'une
facture Orange ne puisse jamais récupérer une règle ONEE ou inwi.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

ProviderId = Literal["onee", "iam", "orange_ma", "inwi", "unknown"]


@dataclass(frozen=True)
class ProviderProfile:
    provider_id: ProviderId
    display_name: str
    domain: str
    service_type: str
    service_zone_id: str
    vat_rate: Decimal
    # Marqueurs normalisés (minuscules, sans accents) cherchés dans le texte OCR.
    markers: tuple[str, ...]
    document_kind: str


PROVIDERS: dict[str, ProviderProfile] = {
    "onee": ProviderProfile(
        "onee", "ONEE — Branche Électricité", "utilities", "electricity", "onee_be_national",
        Decimal("0.14"), ("office national de l electricite", "onee"), "Facture d'électricité",
    ),
    "iam": ProviderProfile(
        "iam", "Maroc Telecom", "telecommunications", "mobile_postpaid", "morocco_national",
        Decimal("0.20"), ("maroc telecom", "itissalat al maghrib", "iam ma"), "Facture d'abonnement mobile",
    ),
    "orange_ma": ProviderProfile(
        "orange_ma", "Orange Maroc", "telecommunications", "consumer_postpaid", "morocco_national",
        Decimal("0.20"), ("orange maroc", "orange ma", "medi telecom"), "Facture d'abonnement",
    ),
    "inwi": ProviderProfile(
        "inwi", "inwi", "telecommunications", "consumer_postpaid", "morocco_national",
        Decimal("0.20"), ("inwi", "wana corporate"), "Facture d'abonnement",
    ),
}


def detect_provider(normalized_text: str) -> ProviderProfile | None:
    """Retourne le seul fournisseur reconnu, ou None si aucun ou plusieurs."""
    text = f" {normalized_text} "
    found = [
        profile for profile in PROVIDERS.values()
        if any(f" {marker} " in text for marker in profile.markers)
    ]
    # Deux fournisseurs sur un même document (ex. partenaire de paiement cité)
    # ne doivent pas être départagés au hasard.
    return found[0] if len(found) == 1 else None


COMMON_CRITICAL_FIELDS = ("invoice_number", "total_check", "due_date", "invoice_date")


def critical_fields(provider_id: str) -> tuple[str, ...]:
    """Le tarif ONEE décide du périmètre résidentiel ; il n'existe pas en télécom."""
    return COMMON_CRITICAL_FIELDS + (("tariff",) if provider_id == "onee" else ())
