"""Construit 25 chunks télécom à partir de six archives officielles approuvées.

Le module travaille uniquement sur des fichiers locaux. Chaque passage attendu
doit être retrouvé mot pour mot après normalisation des espaces ; si une page
change, la construction s'arrête au lieu d'inventer ou de conserver un texte
potentiellement obsolète.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Sequence

from qralia.rag.onee_corpus import validate_with_schema


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    title: str
    publisher: str
    provider_id: str
    provider_name: str
    service_type: str
    languages: tuple[str, ...]
    source_role: str = "operator_primary"
    risk_level: str = "medium"


class _VisibleTextParser(HTMLParser):
    """Extrait le texte visible en ignorant scripts, styles et SVG."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "svg", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "svg", "noscript"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and data.strip():
            self.parts.append(data)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _visible_text(raw: bytes) -> str:
    parser = _VisibleTextParser()
    parser.feed(raw.decode("utf-8", errors="replace"))
    parser.close()
    return _clean(" ".join(parser.parts))


def _quotes(text: str, specs: Sequence[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """Vérifie chaque citation dans l'archive avant de la retenir."""

    normalized = _clean(text)
    sections: list[tuple[str, str, str]] = []
    for section_id, heading, quote in specs:
        quote = _clean(quote)
        if quote not in normalized:
            raise ValueError(
                f"Passage attendu absent pour {section_id!r}. "
                "La page a changé : revue humaine requise."
            )
        sections.append((section_id, heading, quote))
    return sections


def extract_iam(raw: bytes) -> list[tuple[str, str, str]]:
    text = raw.decode("utf-8")
    blocks = re.split(r"(?m)(?=^ARTICLE(?:S)? )", text)
    wanted = {
        "ARTICLE 8.1.1": ("facturation-mensuelle", "Facturation mensuelle"),
        "ARTICLE 8.1.2": ("exigibilite-non-reception", "Exigibilité même sans réception"),
        "ARTICLE 8.1.4": ("erreur-facturation", "Erreur ou surfacturation"),
        "ARTICLE 8.3": ("prelevement-contestation", "Contestation d'un prélèvement"),
        "ARTICLE 8.4": ("penalites-retard", "Pénalités de retard"),
        "ARTICLES 17": ("suspension-retablissement", "Suspension et rétablissement"),
        "ARTICLE 19.1": ("resiliation-client", "Résiliation demandée par le client"),
    }
    result: list[tuple[str, str, str]] = []
    for block in blocks:
        lines = [line.strip() for line in block.strip().splitlines() if line.strip()]
        if not lines:
            continue
        label = lines[0]
        for marker, (section_id, heading) in wanted.items():
            if label.startswith(marker):
                body = _clean(" ".join(lines[1:]))
                if not body:
                    raise ValueError(f"Passage IAM vide pour {section_id}.")
                result.append((section_id, heading, body))
                break
    if len(result) != 7:
        raise ValueError(f"Sept passages IAM étaient attendus ; {len(result)} trouvés.")
    return result


def extract_orange_payment(raw: bytes) -> list[tuple[str, str, str]]:
    return _quotes(_visible_text(raw), (
        ("delai-paiement", "Délai minimal de paiement", "Vous disposez d’un délai de minimum 10 jours pour effectuer le règlement votre facture."),
        ("paiement-max-it-carte", "Paiement par Max it", "Payer votre facture par carte bancaire via votre smartphone sur l’application Max it."),
        ("paiement-espace-client-carte", "Paiement dans l'espace client", "Payer vos factures avec votre carte bancaire en toute sécurité en vous connectant sur le site web."),
        ("paiement-orange-money", "Paiement par Orange Money", "Payez vos factures Orange et récupérez 10% du montant payé !"),
        ("paiement-tassbik", "Paiement à l'avance Tassbik", "Vous pouvez payer autant de factures que vous souhaitez à l’avance."),
        ("paiement-agents", "Paiement chez les partenaires", "Payez vos factures en espèce chez nos partenaires Cash plus, Fawatir, Barid cash, MPOST, Damane Cash et WafaCash par la simple présentation de votre numéro de téléphone."),
        ("paiement-prelevement", "Paiement automatique", "Le montant de votre facture sera prélevé automatiquement de votre compte bancaire tous les mois. Pour en profiter il suffit de vous rendre à l'agence Orange la plus proche mené de votre RIB ou spécimen de chèque."),
        ("paiement-recharge", "Paiement par recharge", "Effectuer une recharge d’un montant équivalent ou plus à votre facture, puis appelez le 5555 pour confirmer la transaction. Le paiement via recharge est réservé à tous les clients Orange."),
        ("paiement-agence", "Paiement en agence Orange", "Plus de 400 agences Orange au Maroc sont à votre disposition pour payer vos factures."),
    ))


def extract_orange_consult(raw: bytes) -> list[tuple[str, str, str]]:
    return _quotes(_visible_text(raw), (
        ("consulter-facture", "Consulter et télécharger la facture", "Vous pouvez consulter et télécharger vos factures sur votre espace client Orange et moi. Un SMS de notification sera envoyé sur votre mobile pour vous informer de la disponibilité de votre facture."),
    ))


def extract_inwi_payment(raw: bytes) -> list[tuple[str, str, str]]:
    return _quotes(_visible_text(raw), (
        ("paiement-my-inwi", "Paiement avec My inwi", "Profitez de 1Go d’internet gratuit, valable 7jours, pour le paiement de votre facture sur l’application My inwi."),
        ("paiement-agence", "Paiement en agence inwi", "Nos agences sont ouvertes du lundi au samedi de 9h à 20h00"),
        ("paiement-ebanking", "Paiement par application bancaire", "L’espace client de votre banque vous permet bien de payer toutes vos factures en quelques clics seulement"),
        ("paiement-prelevement", "Prélèvement bancaire", "Choisissez le prélèvement bancaire qui est automatique et 100% sécurisé."),
        ("paiement-agents", "Paiement chez les partenaires", "Wafacash, Tasshilates & Fawatir +de 8000 agences sur le Royaume"),
    ))


def extract_inwi_money(raw: bytes) -> list[tuple[str, str, str]]:
    return _quotes(_visible_text(raw), (
        ("paiement-inwi-money", "Paiement par inwi money", "Votre facture est payée immédiatement : Saisissez la Catégorie de paiement «Télécom», puis le facturier « inwi » Saisissez le numéro mobile inwi Confirmez le numéro mobile inwi Validez avec votre code secret"),
    ))


def extract_anrt(raw: bytes) -> list[tuple[str, str, str]]:
    text = _visible_text(raw)
    return _quotes(text, (
        ("reclamation-operateur", "الشكوى الأولى لدى المتعهد", "ينبغي أن يوجه المستهلك أولا شكوى مباشرة للوكالة التجارية لمتعهد المواصلات أو يرسلها عبر رسالة مضمونة. كما ينبغي أن يستعلم عن الأجل وشروط قبول شكواه، خاصة إذا تعلق الأمر بشكوى تخص الفاتورة."),
        ("recours-anrt", "الطعن بعد استنفاذ مسطرة المتعهد", "بعد استنفاذ وسائل الطعن لدى متعهد المواصلات، إلى من يمكنني أن أتوجه ؟"),
    ))


def _load_manifest(path: Path, raw: bytes) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(raw).hexdigest()
    if manifest["sha256"] != digest:
        raise ValueError(f"SHA-256 incohérent pour {path}.")
    return manifest


def make_document(
    raw_path: Path,
    manifest_path: Path,
    spec: SourceSpec,
    extractor: Callable[[bytes], list[tuple[str, str, str]]],
    *,
    verified_at: str,
    validity_status: str,
) -> dict[str, Any]:
    raw = raw_path.read_bytes()
    manifest = _load_manifest(manifest_path, raw)
    retrieved = datetime.fromisoformat(manifest["retrieved_at"].replace("Z", "+00:00"))
    stamp = retrieved.strftime("%Y%m%dT%H%M%SZ").lower()
    sections = extractor(raw)
    return {
        "schema_version": "0.3.0",
        "document_id": f"{spec.source_id}-{stamp}",
        "source": {
            "source_id": spec.source_id,
            "title": spec.title,
            "publisher": spec.publisher,
            "canonical_url": manifest["final_url"],
            "languages": list(spec.languages),
            "published_at": None,
        },
        "acquisition": {
            "retrieved_at": manifest["retrieved_at"],
            "http_status": manifest["http_status"],
            "media_type": manifest["media_type"],
            "sha256": manifest["sha256"],
            "raw_artifact_path": raw_path.as_posix(),
        },
        "classification": {
            "domain": "telecommunications",
            "provider_id": spec.provider_id,
            "provider_name": spec.provider_name,
            "service_type": spec.service_type,
            "service_zone_id": "morocco_national",
            "legacy_operator": None,
            "source_role": spec.source_role,
            "audiences": ["residential"],
            "risk_level": spec.risk_level,
            "validity_status": validity_status,
            "verified_at": verified_at,
        },
        "sections": [
            {
                "section_id": section_id,
                "order": order,
                "heading_path": ["Télécommunications", spec.provider_name, spec.title, heading],
                "text": text,
            }
            for order, (section_id, heading, text) in enumerate(sections)
        ],
    }


def build_chunks(document: dict[str, Any]) -> list[dict[str, Any]]:
    source, classification = document["source"], document["classification"]
    copied = (
        "domain", "provider_id", "provider_name", "service_type", "service_zone_id",
        "legacy_operator", "source_role", "audiences", "validity_status",
        "verified_at", "risk_level",
    )
    return [
        {
            "schema_version": "0.3.0",
            "chunk_id": f"{document['document_id']}--{section['section_id']}",
            "document_id": document["document_id"],
            "source_id": source["source_id"],
            "section_id": section["section_id"],
            "text": section["text"],
            "metadata": {
                "title": source["title"],
                "publisher": source["publisher"],
                "canonical_url": source["canonical_url"],
                "languages": source["languages"],
                **{key: classification[key] for key in copied},
                "heading_path": section["heading_path"],
            },
        }
        for section in document["sections"]
    ]


SOURCES: tuple[tuple[str, SourceSpec, Callable[[bytes], list[tuple[str, str, str]]]], ...] = (
    ("iam", SourceSpec("telecom-iam-mobile-postpaid", "Conditions générales de vente Téléphonie Mobile Postpayée", "Maroc Telecom", "iam", "Maroc Telecom", "mobile_postpaid", ("fr",), "legal_primary", "high"), extract_iam),
    ("orange_payment", SourceSpec("telecom-orange-payer-facture", "Comment payer ma facture ?", "Orange Maroc", "orange_ma", "Orange Maroc", "consumer_postpaid", ("fr",)), extract_orange_payment),
    ("orange_consult", SourceSpec("telecom-orange-consulter-facture", "Consulter votre facture", "Orange Maroc", "orange_ma", "Orange Maroc", "consumer_postpaid", ("fr",)), extract_orange_consult),
    ("inwi_payment", SourceSpec("telecom-inwi-payer-facture", "Payer sa facture", "inwi", "inwi", "inwi", "consumer_postpaid", ("fr",)), extract_inwi_payment),
    ("inwi_money", SourceSpec("telecom-inwi-money-factures", "Payer vos factures", "inwi money", "inwi", "inwi", "consumer_postpaid", ("fr",), "payment_partner"), extract_inwi_money),
    ("anrt", SourceSpec("telecom-anrt-reclamations", "Réclamations", "ANRT", "anrt", "Agence Nationale de Réglementation des Télécommunications", "cross_operator_complaints", ("ar",), "regulator", "high"), extract_anrt),
)

EXPECTED_CHUNK_COUNTS = {
    "iam": 7,
    "orange_payment": 9,
    "orange_consult": 1,
    "inwi_payment": 5,
    "inwi_money": 1,
    "anrt": 2,
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalise six sources télécom et construit exactement 25 chunks.")
    for key, _, _ in SOURCES:
        parser.add_argument(f"--{key.replace('_', '-')}-raw", required=True, type=Path)
        parser.add_argument(f"--{key.replace('_', '-')}-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--verified-at", required=True)
    parser.add_argument("--validity-status", choices=("active", "restricted", "stale", "unknown", "withdrawn"), default="unknown")
    parser.add_argument("--schemas-dir", type=Path, default=Path("schemas"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    documents: list[dict[str, Any]] = []
    for key, spec, extractor in SOURCES:
        document = make_document(
            getattr(args, f"{key}_raw"), getattr(args, f"{key}_manifest"), spec,
            extractor, verified_at=args.verified_at, validity_status=args.validity_status,
        )
        if len(document["sections"]) != EXPECTED_CHUNK_COUNTS[key]:
            raise ValueError(
                f"{key}: {EXPECTED_CHUNK_COUNTS[key]} sections attendues ; "
                f"{len(document['sections'])} trouvées."
            )
        documents.append(document)
    chunks = [chunk for document in documents for chunk in build_chunks(document)]
    if len(chunks) != 25:
        raise ValueError(f"Le corpus attendu contient 25 chunks ; {len(chunks)} produits.")
    for document in documents:
        validate_with_schema(document, args.schemas_dir / "normalized_document.schema.json")
        _write_json(args.output_dir / document["source"]["source_id"] / "document.json", document)
    for chunk in chunks:
        validate_with_schema(chunk, args.schemas_dir / "retrieval_chunk.schema.json")
    output = args.output_dir / "telecom" / "chunks.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(c, ensure_ascii=False, separators=(",", ":")) for c in chunks) + "\n", encoding="utf-8")
    print(f"Documents normalisés : {len(documents)} ; chunks télécom : {len(chunks)}")
    print(f"Sortie agrégée : {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
