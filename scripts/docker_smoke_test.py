"""Teste une pile QraLia conteneurisée avec une facture synthétique locale."""

from __future__ import annotations

import argparse
import json
import mimetypes
import uuid
from pathlib import Path
from urllib.request import Request, urlopen


def read_json(url: str) -> dict:
    with urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def multipart_body(path: Path) -> tuple[bytes, str]:
    boundary = f"----qralia-{uuid.uuid4().hex}"
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    parts = [
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="confirm_synthetic"\r\n\r\n'
        "true\r\n",
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        f"Content-Type: {mime}\r\n\r\n",
    ]
    body = parts[0].encode() + parts[1].encode() + path.read_bytes()
    body += f"\r\n--{boundary}--\r\n".encode()
    return body, boundary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("invoice", type=Path, help="PDF, PNG ou JPEG entièrement synthétique")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/api")
    args = parser.parse_args()
    if not args.invoice.is_file():
        raise SystemExit(f"Facture introuvable : {args.invoice}")

    health = read_json(f"{args.base_url}/health")
    ready = read_json(f"{args.base_url}/ready")
    if health.get("status") != "ok" or ready.get("status") != "ready":
        raise SystemExit(f"Pile non prête : health={health}, ready={ready}")

    body, boundary = multipart_body(args.invoice)
    upload = Request(
        f"{args.base_url}/v1/invoices",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urlopen(upload, timeout=180) as response:
        invoice = json.loads(response.read().decode("utf-8"))

    invoice_id = invoice["invoice_id"]
    print(json.dumps({
        "health": health["status"],
        "ready": ready["status"],
        "invoice_id": invoice_id,
        "validation": invoice["validation"],
    }, ensure_ascii=False, indent=2))

    delete = Request(
        f"{args.base_url}/v1/invoices/{invoice_id}",
        method="DELETE",
    )
    with urlopen(delete, timeout=10) as response:
        if response.status != 204:
            raise SystemExit(f"Suppression inattendue : HTTP {response.status}")


if __name__ == "__main__":
    main()

