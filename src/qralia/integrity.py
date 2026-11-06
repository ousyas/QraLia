"""Utilitaires d'intégrité reproductibles entre Windows et Linux."""

from __future__ import annotations

import hashlib
from pathlib import Path


def text_sha256_variants(path: Path) -> set[str]:
    """Calcule les empreintes acceptables d'un texte selon ses fins de ligne.

    Git peut matérialiser un même fichier texte avec ``LF`` ou ``CRLF`` selon
    la plateforme. Le contenu logique reste identique ; seules les fins de
    ligne changent. Nous acceptons donc l'empreinte brute et celles des deux
    représentations normalisées, sans tolérer d'autre modification.
    """

    raw = path.read_bytes()
    text = raw.decode("utf-8")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    variants = (
        raw,
        normalized.encode("utf-8"),
        normalized.replace("\n", "\r\n").encode("utf-8"),
    )
    return {hashlib.sha256(content).hexdigest() for content in variants}


def text_sha256_matches(path: Path, expected: str) -> bool:
    """Vérifie un SHA-256 textuel sans dépendre de LF ou CRLF."""

    return expected in text_sha256_variants(path)
