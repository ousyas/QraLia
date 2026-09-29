"""Normalise une page ONEE approuvée et construit ses chunks RAG.

Le module traite une copie HTML locale. Il ne contacte jamais le site ONEE et
n'utilise aucun LLM : le texte officiel est extrait de façon déterministe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Sequence


SOURCE_ID = "onee-regler-facture-residentiel"
CANONICAL_URL = (
    "https://www.one.org.ma/fr/pages/interne.asp?esp=1&id1=3&id2=46&t2=1"
)
PUBLISHER = "ONEE - Branche Electricité"


@dataclass(frozen=True)
class ExtractedPage:
    """Contenu utile extrait de la zone HTML ``td#contenu``."""

    title: str
    introduction: str
    payment_methods: tuple[str, ...]


def _clean_text(parts: Sequence[str]) -> str:
    """Réunit des fragments HTML sans réécrire le contenu lexical."""

    return re.sub(r"\s+", " ", "".join(parts)).strip()


class _OneeContentParser(HTMLParser):
    """Parse uniquement la cellule principale de cette page ONEE historique."""

    _VOID_ELEMENTS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._content_depth = 0
        self._heading_depth = 0
        self._in_list = False
        self._in_list_item = False
        self._heading_parts: list[str] = []
        self._introduction_parts: list[str] = []
        self._current_item_parts: list[str] = []
        self.payment_methods: list[str] = []

    @property
    def in_content(self) -> bool:
        return self._content_depth > 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attributes = {name.lower(): value for name, value in attrs}

        if not self.in_content:
            if tag == "td" and attributes.get("id", "").lower() == "contenu":
                self._content_depth = 1
            return

        if tag not in self._VOID_ELEMENTS:
            self._content_depth += 1

        if tag == "p" and attributes.get("align", "").lower() == "center":
            self._heading_depth = self._content_depth
        elif tag == "ul":
            self._in_list = True
        elif tag == "li" and self._in_list:
            self._in_list_item = True
            self._current_item_parts = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if not self.in_content:
            return

        if tag == "li" and self._in_list_item:
            item = _clean_text(self._current_item_parts)
            if item:
                self.payment_methods.append(item)
            self._current_item_parts = []
            self._in_list_item = False
        elif tag == "ul":
            self._in_list = False

        if self._heading_depth == self._content_depth and tag == "p":
            self._heading_depth = 0

        self._content_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.in_content or not data.strip():
            return

        if self._heading_depth:
            self._heading_parts.append(data)
        elif self._in_list_item:
            self._current_item_parts.append(data)
        elif not self._in_list:
            self._introduction_parts.append(data)

    def result(self) -> ExtractedPage:
        title = _clean_text(self._heading_parts)
        introduction = _clean_text(self._introduction_parts)

        if not title:
            raise ValueError("Titre introuvable dans td#contenu.")
        if not introduction:
            raise ValueError("Introduction introuvable dans td#contenu.")
        if len(self.payment_methods) != 4:
            raise ValueError(
                "La page devait contenir exactement quatre moyens de paiement ; "
                f"{len(self.payment_methods)} ont été trouvés. Revue humaine requise."
            )

        expected_markers = ("espèces", "chèque", "virement", "acompte")
        for item, marker in zip(self.payment_methods, expected_markers, strict=True):
            if marker.casefold() not in item.casefold():
                raise ValueError(
                    f"Le moyen de paiement attendu « {marker} » est absent. "
                    "La structure de la page a peut-être changé."
                )

        return ExtractedPage(title, introduction, tuple(self.payment_methods))


def extract_page(html_bytes: bytes) -> ExtractedPage:
    """Décode l'HTML ONEE historique puis extrait sa zone principale."""

    parser = _OneeContentParser()
    parser.feed(html_bytes.decode("iso-8859-1"))
    parser.close()
    return parser.result()


def _sha256(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest()


def _document_id(retrieved_at: datetime) -> str:
    stamp = retrieved_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ").lower()
    return f"{SOURCE_ID}-{stamp}"


def normalize_html(
    raw_path: Path,
    *,
    retrieved_at: datetime,
    http_status: int,
    media_type: str,
    verified_at: str,
    validity_status: str = "unknown",
) -> dict[str, Any]:
    """Transforme l'archive HTML en document normalisé conforme au schéma."""

    raw_bytes = raw_path.read_bytes()
    page = extract_page(raw_bytes)
    section_specs = (
        ("date-reglement", "Date de règlement", page.introduction),
        ("paiement-especes", "Paiement en espèces", page.payment_methods[0]),
        ("paiement-cheque", "Paiement par chèque", page.payment_methods[1]),
        ("paiement-virement", "Virement bancaire et prélèvement", page.payment_methods[2]),
        ("paiement-acompte", "Paiement par acompte", page.payment_methods[3]),
    )

    return {
        "schema_version": "0.1.0",
        "document_id": _document_id(retrieved_at),
        "source": {
            "source_id": SOURCE_ID,
            "title": page.title,
            "publisher": PUBLISHER,
            "canonical_url": CANONICAL_URL,
            "languages": ["fr"],
            "published_at": None,
        },
        "acquisition": {
            "retrieved_at": retrieved_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "http_status": http_status,
            "media_type": media_type,
            "sha256": _sha256(raw_bytes),
            "raw_artifact_path": raw_path.as_posix(),
        },
        "classification": {
            "domain": "onee_electricity_invoice",
            "audiences": ["residential"],
            "risk_level": "medium",
            "validity_status": validity_status,
            "verified_at": verified_at,
        },
        "sections": [
            {
                "section_id": section_id,
                "order": order,
                "heading_path": ["Résidentiel", page.title, heading],
                "text": text,
            }
            for order, (section_id, heading, text) in enumerate(section_specs)
        ],
    }


def build_chunks(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Crée un chunk autonome par section du document normalisé."""

    source = document["source"]
    classification = document["classification"]
    chunks: list[dict[str, Any]] = []

    for section in document["sections"]:
        chunks.append(
            {
                "schema_version": "0.1.0",
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
                    "audiences": classification["audiences"],
                    "validity_status": classification["validity_status"],
                    "verified_at": classification["verified_at"],
                    "risk_level": classification["risk_level"],
                    "heading_path": section["heading_path"],
                },
            }
        )

    return chunks


def validate_with_schema(instance: Any, schema_path: Path) -> None:
    """Valide une donnée avec le schéma JSON indiqué."""

    try:
        import jsonschema
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "La dépendance jsonschema manque. Installe le projet avec : "
            "py -m pip install -e .[dev]"
        ) from exc

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(instance)


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("L'horodatage doit inclure un fuseau, par exemple Z.")
    return parsed


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = (json.dumps(value, ensure_ascii=False, separators=(",", ":")) for value in values)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalise le HTML ONEE résidentiel puis construit ses chunks RAG."
    )
    parser.add_argument("--input", required=True, type=Path, help="Archive HTML brute locale.")
    parser.add_argument("--document-output", required=True, type=Path, help="JSON normalisé à créer.")
    parser.add_argument("--chunks-output", required=True, type=Path, help="JSONL de chunks à créer.")
    parser.add_argument("--retrieved-at", required=True, type=_parse_datetime, help="Date UTC d'acquisition, format ISO 8601.")
    parser.add_argument("--http-status", required=True, type=int, help="Statut HTTP observé à l'acquisition.")
    parser.add_argument("--media-type", default="text/html; charset=iso-8859-1", help="Type MIME observé.")
    parser.add_argument("--verified-at", required=True, help="Date de revue humaine, format AAAA-MM-JJ.")
    parser.add_argument(
        "--validity-status",
        choices=("active", "restricted", "stale", "unknown", "withdrawn"),
        default="unknown",
        help="Utiliser unknown avant la comparaison humaine, puis active après validation.",
    )
    parser.add_argument("--schemas-dir", type=Path, default=Path("schemas"), help="Dossier contenant les schémas JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    document = normalize_html(
        args.input,
        retrieved_at=args.retrieved_at,
        http_status=args.http_status,
        media_type=args.media_type,
        verified_at=args.verified_at,
        validity_status=args.validity_status,
    )
    chunks = build_chunks(document)

    validate_with_schema(document, args.schemas_dir / "normalized_document.schema.json")
    for chunk in chunks:
        validate_with_schema(chunk, args.schemas_dir / "retrieval_chunk.schema.json")

    _write_json(args.document_output, document)
    _write_jsonl(args.chunks_output, chunks)
    print(f"Document normalisé : {args.document_output}")
    print(f"Chunks créés : {len(chunks)} dans {args.chunks_output}")
    print(f"SHA-256 brut : {document['acquisition']['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
