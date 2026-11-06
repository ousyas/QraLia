"""Acquisition contrôlée d'une source publique approuvée.

Le collecteur enregistre les octets reçus et un manifeste séparé. Il ne parse
pas le contenu et ne décide pas si la source doit devenir active.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Sequence
from urllib.request import Request, urlopen


USER_AGENT = "QraLia/0.1 educational-source-collector"


def acquire(
    url: str, *, timeout: float = 30.0, user_agent: str = USER_AGENT
) -> tuple[bytes, dict[str, object]]:
    """Télécharge une URL publique et retourne les octets et leur manifeste."""

    request = Request(url, headers={"User-Agent": user_agent})
    retrieved_at = datetime.now(timezone.utc)
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - URL approuvée en entrée CLI
        raw_bytes = response.read()
        manifest: dict[str, object] = {
            "schema_version": "0.1.0",
            "requested_url": url,
            "final_url": response.geturl(),
            "retrieved_at": retrieved_at.isoformat().replace("+00:00", "Z"),
            "http_status": response.status,
            "media_type": response.headers.get("Content-Type", "application/octet-stream"),
            "content_length": len(raw_bytes),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "user_agent": user_agent,
        }
    return raw_bytes, manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Archive une source publique approuvée.")
    parser.add_argument("--url", required=True, help="URL publique exacte à télécharger.")
    parser.add_argument("--output", required=True, type=Path, help="Fichier brut à écrire.")
    parser.add_argument("--manifest-output", required=True, type=Path, help="Manifeste JSON à écrire.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Délai réseau maximal en secondes.")
    parser.add_argument(
        "--user-agent",
        default=USER_AGENT,
        help="Identité HTTP explicite utilisée pour l'acquisition.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw_bytes, manifest = acquire(
        args.url,
        timeout=args.timeout,
        user_agent=args.user_agent,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw_bytes)
    args.manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Archive : {args.output}")
    print(f"Manifeste : {args.manifest_output}")
    print(f"HTTP : {manifest['http_status']}")
    print(f"URL finale : {manifest['final_url']}")
    print(f"Type MIME : {manifest['media_type']}")
    print(f"SHA-256 : {manifest['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
