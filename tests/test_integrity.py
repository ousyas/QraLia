from pathlib import Path

from qralia.integrity import text_sha256_matches, text_sha256_variants


def test_text_hash_accepts_lf_and_crlf_representations(tmp_path: Path) -> None:
    lf_path = tmp_path / "lf.jsonl"
    crlf_path = tmp_path / "crlf.jsonl"
    lf_path.write_bytes(b'{"id":1}\n{"id":2}\n')
    crlf_path.write_bytes(b'{"id":1}\r\n{"id":2}\r\n')

    lf_hash = next(
        digest
        for digest in text_sha256_variants(lf_path)
        if text_sha256_matches(lf_path, digest)
    )

    assert text_sha256_matches(crlf_path, lf_hash)


def test_text_hash_rejects_a_content_change(tmp_path: Path) -> None:
    original = tmp_path / "original.jsonl"
    changed = tmp_path / "changed.jsonl"
    original.write_text('{"id":1}\n', encoding="utf-8")
    changed.write_text('{"id":2}\n', encoding="utf-8")
    expected = next(iter(text_sha256_variants(original)))

    assert not text_sha256_matches(changed, expected)
