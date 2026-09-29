from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from qralia.rag.onee_corpus import build_chunks, extract_page, normalize_html


SYNTHETIC_HTML = b"""
<html><body>
  <td id="menu">Actualites sans rapport</td>
  <td id="contenu">
    <p align="center"><b>Reglement de votre facture</b></p>
    <div>Une date de reglement figure sur la facture.</div>
    <p>Vous pouvez regler votre facture soit :</p>
    <ul>
      <li>en esp\xe8ces aupr\xe8s de votre agence ;</li>
      <li>par ch\xe8que aupr\xe8s de votre agence ;</li>
      <li>par virement bancaire avec domiciliation ;</li>
      <li>par acompte aupr\xe8s de votre agence.</li>
    </ul>
  </td>
</body></html>
"""


def test_extract_page_ignores_navigation_and_preserves_four_methods() -> None:
    page = extract_page(SYNTHETIC_HTML)

    assert page.title == "Reglement de votre facture"
    assert "Actualites" not in page.introduction
    assert len(page.payment_methods) == 4
    assert page.payment_methods[0].startswith("en espèces")


def test_normalize_then_build_one_chunk_per_section(tmp_path: Path) -> None:
    raw_path = tmp_path / "page.html"
    raw_path.write_bytes(SYNTHETIC_HTML)
    document = normalize_html(
        raw_path,
        retrieved_at=datetime(2026, 9, 29, 19, 27, 57, tzinfo=timezone.utc),
        http_status=200,
        media_type="text/html; charset=iso-8859-1",
        verified_at="2026-09-29",
    )
    chunks = build_chunks(document)

    assert document["classification"]["audiences"] == ["residential"]
    assert document["classification"]["validity_status"] == "unknown"
    assert len(document["acquisition"]["sha256"]) == 64
    assert len(document["sections"]) == 5
    assert len(chunks) == 5
    assert chunks[0]["metadata"]["canonical_url"].startswith("https://www.one.org.ma/")
    assert all(chunk["document_id"] == document["document_id"] for chunk in chunks)
