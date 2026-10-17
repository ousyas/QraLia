"""Extraction ONEE par Tesseract suivie d'un parsing déterministe."""

from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Callable, Iterable

from qralia.domain.extracted_invoice import (
    ExtractedField,
    ExtractedInvoice,
    ExtractionMetadata,
    FieldProvenance,
    FieldStatus,
    InvoiceFields,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
CRITICAL_FIELDS = ("invoice_number", "total_check", "due_date", "invoice_date", "tariff")


@dataclass(frozen=True)
class OcrToken:
    text: str
    confidence: float
    page: int = 1
    bbox: tuple[int, int, int, int] | None = None


def _configure_tesseract(pytesseract) -> None:
    configured = os.getenv("QRALIA_TESSERACT_CMD")
    standard_windows = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if configured:
        pytesseract.pytesseract.tesseract_cmd = configured
    elif standard_windows.exists():
        pytesseract.pytesseract.tesseract_cmd = str(standard_windows)


def _ocr_language(pytesseract) -> str:
    requested = os.getenv("QRALIA_OCR_LANGUAGE", "fra")
    available = set(pytesseract.get_languages(config=""))
    return requested if requested in available else "eng"


def _normalize(text: str) -> str:
    value = unicodedata.normalize("NFD", text)
    return value.encode("ascii", "ignore").decode("ascii").lower()


def _label_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _normalize(text)).strip()


def _tokens_from_tesseract(image, page: int) -> list[OcrToken]:
    import pytesseract

    _configure_tesseract(pytesseract)

    data = pytesseract.image_to_data(
        image,
        lang=_ocr_language(pytesseract),
        config="--psm 6",
        output_type=pytesseract.Output.DICT,
    )
    result: list[OcrToken] = []
    for index, raw_text in enumerate(data["text"]):
        text = raw_text.strip()
        confidence = float(data["conf"][index])
        if not text or confidence < 0:
            continue
        left = int(data["left"][index])
        top = int(data["top"][index])
        width = int(data["width"][index])
        height = int(data["height"][index])
        result.append(OcrToken(text, confidence / 100, page, (left, top, left + width, top + height)))
    return result


def _candidate_quality(tokens: list[OcrToken]) -> float:
    if not tokens:
        return 0.0
    joined = " ".join(_label_text(token.text) for token in tokens)
    coverage = sum(word in joined for word in ("facture", "total", "regler", "echeance", "tarif")) / 5
    return sum(token.confidence for token in tokens) / len(tokens) + 0.15 * coverage


def _anchor_coverage(tokens: list[OcrToken]) -> float:
    joined = " ".join(_label_text(token.text) for token in tokens)
    anchors = ("facture", "total", "regler", "echeance", "tarif")
    return sum(anchor in joined for anchor in anchors) / len(anchors)


def _critical_parse_count(tokens: list[OcrToken]) -> int:
    parsed = parse_ocr_tokens(tokens, "candidate.pdf")
    return sum(getattr(parsed.fields, name).status == FieldStatus.EXTRACTED for name in CRITICAL_FIELDS)


def _best_page_tokens(image, page: int) -> list[OcrToken]:
    from PIL import ImageOps

    original = ImageOps.exif_transpose(image).convert("RGB")
    variants = [original]
    try:
        from qralia.ocr.preprocess import preprocess_for_ocr

        variants.append(preprocess_for_ocr(original))
    except (ImportError, ValueError):
        pass
    upright = [_tokens_from_tesseract(variant, page) for variant in variants]
    # Une page droite et structurée ne nécessite pas six rotations de plus.
    # Le choix favorise d'abord les champs réellement parsables, puis la qualité OCR.
    if max((_anchor_coverage(tokens) for tokens in upright), default=0.0) >= 0.4:
        return max(upright, key=lambda tokens: (_critical_parse_count(tokens), _candidate_quality(tokens)), default=[])
    rotated = [
        _tokens_from_tesseract(variant.rotate(angle, expand=True), page)
        for variant in variants
        for angle in (90, 180, 270)
    ]
    candidates = [*upright, *rotated]
    return max(candidates, key=lambda tokens: (_critical_parse_count(tokens), _candidate_quality(tokens)), default=[])


def _ocr_file(file_path: Path, page_limit: int) -> tuple[list[OcrToken], str | None]:
    import pytesseract

    _configure_tesseract(pytesseract)

    if page_limit < 1:
        raise ValueError("page_limit must be at least 1")
    if file_path.suffix.lower() in IMAGE_EXTENSIONS:
        from PIL import Image

        images = [Image.open(file_path)]
    elif file_path.suffix.lower() == ".pdf":
        from pdf2image import convert_from_path

        images = convert_from_path(
            file_path, dpi=250, first_page=1, last_page=page_limit,
            poppler_path=os.getenv("QRALIA_POPPLER_PATH") or None,
        )
    else:
        raise ValueError(f"Unsupported invoice format: {file_path.suffix}")
    tokens = [token for page, image in enumerate(images, 1) for token in _best_page_tokens(image, page)]
    try:
        version = str(pytesseract.get_tesseract_version())
    except pytesseract.TesseractNotFoundError:
        version = None
    return tokens, version


def _find_label_ends(tokens: list[OcrToken], aliases: Iterable[str]) -> list[int]:
    aliases = tuple(_label_text(alias) for alias in aliases)
    positions: list[int] = []
    for start in range(len(tokens)):
        combined = ""
        for end in range(start, min(start + 6, len(tokens))):
            combined = f"{combined} {_label_text(tokens[end].text)}".strip()
            if any(alias == combined or alias in combined for alias in aliases):
                positions.append(end)
                break
    return list(dict.fromkeys(positions))


def _find_label_end(tokens: list[OcrToken], aliases: Iterable[str]) -> int | None:
    positions = _find_label_ends(tokens, aliases)
    return positions[0] if positions else None


def _find_after(
    tokens: list[OcrToken], aliases: Iterable[str], predicate: Callable[[str], bool], scan: int = 12
) -> OcrToken | None:
    end = _find_label_end(tokens, aliases)
    if end is None:
        return None
    return next((token for token in tokens[end + 1 : end + 1 + scan] if predicate(token.text)), None)


def _same_line_tokens(tokens: list[OcrToken], label_end: int, *, tolerance: int = 45) -> list[OcrToken]:
    """Return values visually aligned to the right of a French label."""
    label = tokens[label_end]
    if label.bbox is None:
        return []
    label_y = (label.bbox[1] + label.bbox[3]) / 2
    aligned = [
        token for token in tokens
        if token.page == label.page
        and token.bbox is not None
        and token.bbox[0] >= label.bbox[2] - 5
        and abs(((token.bbox[1] + token.bbox[3]) / 2) - label_y) <= tolerance
    ]
    return sorted(aligned, key=lambda token: token.bbox[0] if token.bbox else 0)


def _parse_money(text: str) -> float | None:
    # Une date comme 01.03.2017 ne doit jamais devenir le montant 1 032 017.
    if re.fullmatch(r"\s*\d{1,2}[./-]\d{1,2}[./-]\d{4}\s*", text):
        return None
    cleaned = re.sub(r"[^0-9,.-]", "", text).replace("-", "")
    if not re.search(r"\d", cleaned):
        return None
    separators = [position for position, char in enumerate(cleaned) if char in ",."]
    decimal_separator = cleaned[separators[-1]] if separators and len(cleaned) - separators[-1] - 1 == 2 else None
    if decimal_separator:
        integer, decimals = cleaned.rsplit(decimal_separator, 1)
        canonical = f"{integer.replace(',', '').replace('.', '')}.{decimals}"
    else:
        canonical = cleaned.replace(",", "").replace(".", "")
    try:
        return float(Decimal(canonical))
    except InvalidOperation:
        return None


def _parse_date(text: str) -> str | None:
    match = re.search(r"(\d{1,2})[./-](\d{1,2})[./-](\d{4})", text)
    if not match:
        return None
    day, month, year = map(int, match.groups())
    try:
        return datetime(year, month, day).date().isoformat()
    except ValueError:
        return None


def _missing(status: FieldStatus = FieldStatus.UNREADABLE) -> ExtractedField:
    return ExtractedField(
        raw_value="", normalized_value=None, confidence=0, status=status,
        provenance=FieldProvenance(page=1, method="tesseract"),
    )


def _field(token: OcrToken, value: object) -> ExtractedField:
    return ExtractedField(
        raw_value=token.text,
        normalized_value=value,
        confidence=round(token.confidence, 4),
        status=FieldStatus.EXTRACTED,
        provenance=FieldProvenance(page=token.page, bbox=token.bbox, method="tesseract"),
    )


def _money_field(tokens: list[OcrToken], aliases: Iterable[str]) -> ExtractedField:
    label_ends = _find_label_ends(tokens, aliases)
    if not label_ends:
        return _missing()
    extracted: list[ExtractedField] = []
    for end in label_ends:
        aligned = _same_line_tokens(tokens, end)
        linear = tokens[end + 1 : end + 13]
        candidates = list(dict.fromkeys([*aligned, *linear]))
        for index, token in enumerate(candidates):
            has_numeric_prefix = bool(
                index and re.fullmatch(r"\d{1,3}[.,]?", candidates[index - 1].text.strip())
            )
            options = [] if has_numeric_prefix else [(token.text, token.confidence, token.bbox)]
            if has_numeric_prefix:
                previous = candidates[index - 1]
                combined = (
                    f"{previous.text}{token.text}"
                    if previous.text.rstrip().endswith((".", ","))
                    else f"{previous.text} {token.text}"
                )
                options.append((combined, min(previous.confidence, token.confidence), token.bbox))
            if (
                index + 1 < len(candidates)
                and token.text.rstrip().endswith((".", ","))
                and re.fullmatch(r"\d{2}", candidates[index + 1].text.strip())
            ):
                following = candidates[index + 1]
                options.append((f"{token.text}{following.text}", min(token.confidence, following.confidence), following.bbox))
            accepted = False
            for raw, confidence, bbox in options:
                if _parse_date(raw) is not None:
                    continue
                if not re.search(r"\d[\d .]*[,.]\d{2}(?:\s*(?:DH|DHS|MAD|TTC|HT))?\s*$", raw, re.IGNORECASE):
                    continue
                value = _parse_money(raw)
                if value is None:
                    continue
                extracted.append(ExtractedField(
                    raw_value=raw, normalized_value=value, confidence=round(confidence, 4),
                    status=FieldStatus.EXTRACTED,
                    provenance=FieldProvenance(page=token.page, bbox=bbox, method="tesseract"),
                ))
                accepted = True
                break
            if accepted:
                break
    return max(extracted, key=lambda field: field.confidence, default=_missing())


def _date_field(tokens: list[OcrToken], aliases: Iterable[str]) -> ExtractedField:
    candidates: list[OcrToken] = []
    for end in _find_label_ends(tokens, aliases):
        aligned = _same_line_tokens(tokens, end, tolerance=55)
        linear = tokens[end + 1 : end + 13]
        search = list(dict.fromkeys([*aligned, *linear]))
        token = next((item for item in search if _parse_date(item.text)), None)
        if token:
            candidates.append(token)
    token = max(candidates, key=lambda item: item.confidence, default=None)
    return _field(token, _parse_date(token.text)) if token else _missing()


def _identifier_field(tokens: list[OcrToken], aliases: Iterable[str]) -> ExtractedField:
    token = _find_after(tokens, aliases, lambda value: len(re.sub(r"\D", "", value)) >= 4, scan=8)
    if not token:
        return _missing()
    value = re.sub(r"[^A-Za-z0-9._-]", "", token.text)
    return _field(token, value) if value else _missing()


def _invoice_date(tokens: list[OcrToken], invoice_number: ExtractedField) -> ExtractedField:
    start = 0
    if invoice_number.status == FieldStatus.EXTRACTED:
        start = next((i + 1 for i, token in enumerate(tokens) if token.text == invoice_number.raw_value), 0)
    for index in range(start, len(tokens)):
        if _label_text(tokens[index].text) == "date":
            for token in tokens[index + 1 : index + 6]:
                if _parse_date(token.text):
                    return _field(token, _parse_date(token.text))
    return _missing()


def _tariff(tokens: list[OcrToken]) -> ExtractedField:
    for index, token in enumerate(tokens):
        if _label_text(token.text) != "tarif":
            continue
        candidates = []
        for item in tokens[index + 1 : index + 7]:
            value = _label_text(item.text)
            if re.search(r"\d", value):
                break
            if value and value not in {"type", "total", "ht", "taxes"}:
                candidates.append(item)
                # Les codes résidentiels courts sont complets en un token.
                # Continuer absorberait le libellé de la ligne suivante.
                if value in {"bt"}:
                    break
            if len(candidates) == 2:
                break
        if candidates:
            raw = " ".join(item.text for item in candidates)
            return ExtractedField(
                raw_value=raw,
                normalized_value="_".join(_label_text(item.text).upper() for item in candidates),
                confidence=round(min(item.confidence for item in candidates), 4),
                status=FieldStatus.EXTRACTED,
                provenance=FieldProvenance(page=candidates[0].page, method="tesseract"),
            )
    return _missing()


def parse_ocr_tokens(
    tokens: list[OcrToken], source_file: str, *, content_digest: str | None = None,
    extracted_at: datetime | None = None, ocr_engine_version: str | None = None,
) -> ExtractedInvoice:
    invoice_number = _identifier_field(tokens, ("n facture", "no facture", "numero facture"))
    contract_number = _identifier_field(tokens, ("n contrat", "no contrat", "numero contrat"))
    total_check = _money_field(tokens, ("total a regler", "total regler"))
    amount_before_tax = _money_field(tokens, ("total ht",))
    taxes = _money_field(tokens, ("total tva", "taxes"))
    due_date = _date_field(tokens, ("date d echeance du reglement", "date d echeance", "echeance"))
    invoice_date = _invoice_date(tokens, invoice_number)
    tariff = _tariff(tokens)
    previous = _money_field(tokens, ("arrieres", "ancien impaye", "impaye precedent"))
    if previous.status == FieldStatus.UNREADABLE:
        previous = _missing(FieldStatus.NOT_PRESENT)
    stamp = _money_field(tokens, ("droit de timbre", "timbre"))
    if stamp.status == total_check.status == FieldStatus.EXTRACTED:
        total_cash = ExtractedField(
            raw_value="", normalized_value=round(total_check.normalized_value + stamp.normalized_value, 2),
            confidence=min(total_check.confidence, stamp.confidence), status=FieldStatus.DERIVED,
            provenance=FieldProvenance(page=total_check.provenance.page, method="deterministic_derivation", derived_from=["total_check", "stamp_duty"]),
        )
    else:
        total_cash = _missing()
    fields = InvoiceFields(
        invoice_number=invoice_number, contract_number=contract_number, total_cash=total_cash,
        total_check=total_check, due_date=due_date, invoice_date=invoice_date, tariff=tariff,
        amount_before_tax=amount_before_tax, taxes=taxes, previous_unpaid=previous,
    )
    digest = content_digest or hashlib.sha256(source_file.encode()).hexdigest()
    invoice_id = str(invoice_number.normalized_value) if invoice_number.status == FieldStatus.EXTRACTED else f"document-{digest[:16]}"
    confidence = min(getattr(fields, name).confidence for name in CRITICAL_FIELDS)
    return ExtractedInvoice(
        invoice_id=invoice_id,
        extraction=ExtractionMetadata(
            extracted_at=extracted_at or datetime.now(timezone.utc), ocr_engine="tesseract",
            ocr_engine_version=ocr_engine_version, source_file=Path(source_file).name,
            source_type="synthetic",
        ),
        fields=fields,
        document_confidence=round(confidence, 4),
    )


def extract_invoice(file_path: Path, *, page_limit: int = 2) -> ExtractedInvoice:
    path = Path(file_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    tokens, version = _ocr_file(path, page_limit)
    return parse_ocr_tokens(tokens, path.name, content_digest=digest, ocr_engine_version=version)

