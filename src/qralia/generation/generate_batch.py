"""Generate a batch of synthetic ONEE invoices (PDF + ground-truth JSON)."""

from pathlib import Path

from qralia.generation.factory import random_invoice_params
from qralia.generation.renderer import render_invoice

BATCH_SIZE = 30
OUTPUT_DIR = Path("data/generated")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating {BATCH_SIZE} invoices into {OUTPUT_DIR}/\n")

    for i in range(1, BATCH_SIZE + 1):
        params = random_invoice_params()
        result = render_invoice(params, OUTPUT_DIR)
        print(f"[{i:02d}/{BATCH_SIZE}] {result['pdf_path']}")

    print(f"\nDone — {BATCH_SIZE} invoices generated.")


if __name__ == "__main__":
    main()