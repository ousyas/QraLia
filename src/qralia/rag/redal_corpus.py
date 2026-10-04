"""Construit un petit corpus Redal à partir de deux archives HTML officielles.

Le module ne contacte jamais Internet et n'utilise aucun LLM. Il sélectionne
des onglets/accordéons connus, normalise seulement les espaces, puis crée un
chunk sémantique autonome par sujet.
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
from typing import Any, Sequence

from qralia.rag.onee_corpus import validate_with_schema


PROVIDER_ID = "srm_rsk"
PROVIDER_NAME = "Société Régionale Multiservices Rabat-Salé-Kénitra"
SERVICE_ZONE_ID = "rabat_sale_ex_redal"
LEGACY_OPERATOR = "Redal"
PUBLISHER = "Redal"


@dataclass(frozen=True)
class HtmlItem:
    title: str
    content: str


def _clean(parts: Sequence[str]) -> str:
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def _without_external_urls(text: str) -> str:
    """Retire les URL de connexion affichées sans supprimer le nom des banques."""

    return re.sub(r"\s+", " ", re.sub(r"https?://\S+", "", text)).strip()


class _ComponentParser(HTMLParser):
    """Extrait les composants Drupal désignés par leurs classes CSS."""

    _VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self, *, root_class: str, title_class: str, content_class: str) -> None:
        super().__init__(convert_charrefs=True)
        self.root_class = root_class
        self.title_class = title_class
        self.content_class = content_class
        self.depth = 0
        self.root_depth: int | None = None
        self.title_depth: int | None = None
        self.content_depth: int | None = None
        self.skip_depth: int | None = None
        self.title_parts: list[str] = []
        self.content_parts: list[str] = []
        self.items: list[HtmlItem] = []

    @staticmethod
    def _classes(attrs: list[tuple[str, str | None]]) -> set[str]:
        values = dict(attrs).get("class") or ""
        return set(values.split())

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag not in self._VOID:
            self.depth += 1
        classes = self._classes(attrs)
        if tag in {"script", "style", "svg"} and self.skip_depth is None:
            self.skip_depth = self.depth
        if tag == "div" and self.root_class in classes and self.root_depth is None:
            self.root_depth = self.depth
            self.title_parts = []
            self.content_parts = []
        if self.root_depth is not None:
            if self.title_class in classes:
                self.title_depth = self.depth
            if self.content_class in classes:
                self.content_depth = self.depth

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self.skip_depth == self.depth:
            self.skip_depth = None
        if self.title_depth == self.depth:
            self.title_depth = None
        if self.content_depth == self.depth:
            self.content_depth = None
        if self.root_depth == self.depth:
            title, content = _clean(self.title_parts), _clean(self.content_parts)
            if title and content:
                self.items.append(HtmlItem(title, content))
            self.root_depth = None
        if tag not in self._VOID:
            self.depth -= 1

    def handle_data(self, data: str) -> None:
        if self.skip_depth is not None or self.root_depth is None or not data.strip():
            return
        if self.title_depth is not None:
            self.title_parts.append(data)
        elif self.content_depth is not None:
            self.content_parts.append(data)


def _extract_components(
    html_bytes: bytes, *, root_class: str, title_class: str, content_class: str
) -> dict[str, str]:
    parser = _ComponentParser(
        root_class=root_class, title_class=title_class, content_class=content_class
    )
    parser.feed(html_bytes.decode("utf-8"))
    parser.close()
    return {item.title: item.content for item in parser.items}


def extract_payment_sections(html_bytes: bytes) -> list[tuple[str, str, str]]:
    items = _extract_components(
        html_bytes,
        root_class="tabs-item",
        title_class="tabs-item__title",
        content_class="tabs-item__content",
    )
    required = (
        "Via le net",
        "JIWAR MOBIL",
        "Prélèvement Automatique Bancaire",
        "Guichets Automatiques Bancaires",
    )
    missing = [title for title in required if title not in items]
    if missing:
        raise ValueError(f"Onglets Redal manquants : {missing}. Revue humaine requise.")

    online = items["Via le net"]
    marker = online.find("Autres sites")
    if marker < 0:
        raise ValueError("Le séparateur « Autres sites » est absent de l'onglet Via le net.")
    direct = online[:marker].strip()
    partners = _without_external_urls(online[marker:].strip())
    if not direct or not partners:
        raise ValueError("Impossible de séparer paiement Redal et banques partenaires.")

    return [
        ("paiement-en-ligne-carte", "Paiement en ligne par carte", direct),
        ("paiement-ebanking", "Paiement par banque en ligne", partners),
        ("paiement-mobile-jiwar", "Paiement mobile JIWAR", items["JIWAR MOBIL"]),
        ("paiement-prelevement", "Prélèvement automatique bancaire", items["Prélèvement Automatique Bancaire"]),
        ("paiement-gab", "Guichets automatiques bancaires", items["Guichets Automatiques Bancaires"]),
    ]


def extract_faq_sections(html_bytes: bytes) -> list[tuple[str, str, str]]:
    items = _extract_components(
        html_bytes,
        root_class="accordion-item",
        title_class="accordion-item__title",
        content_class="accordion-item__content",
    )
    wanted = (
        "Comment se passe le relevé des compteurs ?",
        "Comment faire si mon compteur est inaccessible à la relève par Redal ?",
        "Comment lire ma facture ?",
        "Combien de facture je reçois mensuellement ?",
        "Quel est le délai de paiement ?",
        "Comment s’effectue la facturation ?",
    )
    missing = [title for title in wanted if title not in items]
    if missing:
        raise ValueError(f"Questions Redal manquantes : {missing}. Revue humaine requise.")

    def joined(*titles: str) -> str:
        return " ".join(f"{title} {items[title]}" for title in titles)

    return [
        (
            "releve-et-auto-releve",
            "Relève et auto-relève du compteur",
            joined(wanted[0], wanted[1]),
        ),
        ("lecture-facture", "Lecture de la facture", joined(wanted[2])),
        (
            "facturation-mensuelle",
            "Facturation mensuelle",
            joined(wanted[3], wanted[5]),
        ),
        ("date-limite-suspension", "Délai de paiement et suspension", joined(wanted[4])),
    ]


def _load_manifest(path: Path, raw_bytes: bytes) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(raw_bytes).hexdigest()
    if manifest["sha256"] != digest:
        raise ValueError(f"SHA-256 incohérent entre {path} et l'archive HTML.")
    return manifest


def _make_document(
    *, raw_path: Path, manifest_path: Path, source_id: str, title: str,
    sections: Sequence[tuple[str, str, str]], verified_at: str, validity_status: str,
) -> dict[str, Any]:
    raw_bytes = raw_path.read_bytes()
    manifest = _load_manifest(manifest_path, raw_bytes)
    retrieved = datetime.fromisoformat(manifest["retrieved_at"].replace("Z", "+00:00"))
    stamp = retrieved.strftime("%Y%m%dT%H%M%SZ").lower()
    return {
        "schema_version": "0.3.0",
        "document_id": f"{source_id}-{stamp}",
        "source": {
            "source_id": source_id,
            "title": title,
            "publisher": PUBLISHER,
            "canonical_url": manifest["final_url"],
            "languages": ["fr"],
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
            "domain": "utilities",
            "provider_id": PROVIDER_ID,
            "provider_name": PROVIDER_NAME,
            "service_type": "multi_utility",
            "service_zone_id": SERVICE_ZONE_ID,
            "legacy_operator": LEGACY_OPERATOR,
            "source_role": "operator_primary",
            "audiences": ["residential"],
            "risk_level": "medium",
            "validity_status": validity_status,
            "verified_at": verified_at,
        },
        "sections": [
            {
                "section_id": section_id,
                "order": order,
                "heading_path": ["Client particulier", title, heading],
                "text": text,
            }
            for order, (section_id, heading, text) in enumerate(sections)
        ],
    }


def build_chunks(document: dict[str, Any]) -> list[dict[str, Any]]:
    source, classification = document["source"], document["classification"]
    copied = (
        "domain", "provider_id", "provider_name", "service_type", "service_zone_id", "legacy_operator",
        "source_role", "audiences", "validity_status", "verified_at", "risk_level",
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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(v, ensure_ascii=False, separators=(",", ":")) for v in values) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalise les pages officielles Redal et construit 9 chunks RAG.")
    parser.add_argument("--payment-html", required=True, type=Path)
    parser.add_argument("--payment-manifest", required=True, type=Path)
    parser.add_argument("--faq-html", required=True, type=Path)
    parser.add_argument("--faq-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--verified-at", required=True)
    parser.add_argument("--validity-status", choices=("active", "restricted", "stale", "unknown", "withdrawn"), default="unknown")
    parser.add_argument("--schemas-dir", type=Path, default=Path("schemas"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payment_bytes, faq_bytes = args.payment_html.read_bytes(), args.faq_html.read_bytes()
    payment = _make_document(
        raw_path=args.payment_html, manifest_path=args.payment_manifest,
        source_id="redal-e-paiement", title="E-paiement",
        sections=extract_payment_sections(payment_bytes), verified_at=args.verified_at,
        validity_status=args.validity_status,
    )
    faq = _make_document(
        raw_path=args.faq_html, manifest_path=args.faq_manifest,
        source_id="redal-questions-frequentes", title="Questions fréquentes",
        sections=extract_faq_sections(faq_bytes), verified_at=args.verified_at,
        validity_status=args.validity_status,
    )
    chunks = build_chunks(payment) + build_chunks(faq)
    document_schema = args.schemas_dir / "normalized_document.schema.json"
    chunk_schema = args.schemas_dir / "retrieval_chunk.schema.json"
    for document in (payment, faq):
        validate_with_schema(document, document_schema)
    for chunk in chunks:
        validate_with_schema(chunk, chunk_schema)

    _write_json(args.output_dir / "redal-e-paiement" / "document.json", payment)
    _write_json(args.output_dir / "redal-questions-frequentes" / "document.json", faq)
    _write_jsonl(args.output_dir / "redal" / "chunks.jsonl", chunks)
    print(f"Documents normalisés : 2 ; chunks Redal : {len(chunks)}")
    print(f"Sortie agrégée : {args.output_dir / 'redal' / 'chunks.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
