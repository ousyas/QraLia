"""Evaluate OCR extraction accuracy against ground-truth on all generated invoices."""

import json
from pathlib import Path

from qralia.ocr.extractor import extract_fields

DATA_DIR = Path("data/generated")
FIELDS = [
    "total_cash", "total_check", "due_date", "invoice_date",
    "tariff", "amount_before_tax", "taxes", "previous_unpaid",
]


def _values_match(field_name: str, extracted, expected) -> bool:
    """Compare two normalized values — numeric tolerance for money, exact for strings."""
    if field_name in ("due_date", "invoice_date", "tariff"):
        return str(extracted).strip() == str(expected).strip()
    try:
        return abs(float(extracted) - float(expected)) < 0.01
    except (ValueError, TypeError):
        return str(extracted).strip() == str(expected).strip()


def evaluate():
    gt_files = sorted(DATA_DIR.glob("*.ground_truth.json"))
    if not gt_files:
        print("No ground truth files found in", DATA_DIR)
        return

    total_correct = 0
    total_fields = 0
    per_field_correct = {f: 0 for f in FIELDS}
    per_field_total = {f: 0 for f in FIELDS}
    results = []

    for gt_path in gt_files:
        invoice_id = gt_path.name.replace(".ground_truth.json", "")
        pdf_path = DATA_DIR / f"{invoice_id}.pdf"

        if not pdf_path.exists():
            print(f"  SKIP {invoice_id} — PDF not found")
            continue

        print(f"  Processing {invoice_id}...", end=" ")

        # Load ground truth
        gt = json.loads(gt_path.read_text(encoding="utf-8"))

        # Run extraction
        extracted = extract_fields(pdf_path)

        # Compare field by field
        invoice_correct = 0
        invoice_details = {}
        for field in FIELDS:
            gt_val = gt["fields"][field]["normalized_value"]
            ex_val = extracted["fields"][field]["normalized_value"]
            match = _values_match(field, ex_val, gt_val)

            per_field_correct[field] += int(match)
            per_field_total[field] += 1
            total_correct += int(match)
            total_fields += 1
            invoice_correct += int(match)

            if not match:
                invoice_details[field] = {"expected": gt_val, "got": ex_val}

        status = "OK" if invoice_correct == len(FIELDS) else f"{invoice_correct}/{len(FIELDS)}"
        print(status)

        if invoice_details:
            for field, detail in invoice_details.items():
                print(f"    {field}: expected {detail['expected']}, got {detail['got']}")

        results.append({"invoice_id": invoice_id, "correct": invoice_correct, "total": len(FIELDS)})

    # --- Summary ---
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    overall_accuracy = (total_correct / total_fields * 100) if total_fields else 0
    print(f"\nOverall accuracy: {total_correct}/{total_fields} ({overall_accuracy:.1f}%)\n")

    print("Per-field accuracy:")
    for field in FIELDS:
        n = per_field_total[field]
        c = per_field_correct[field]
        pct = (c / n * 100) if n else 0
        bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
        print(f"  {field:20s}  {bar}  {c}/{n}  ({pct:.1f}%)")

    # Invoices with errors
    errors = [r for r in results if r["correct"] < r["total"]]
    if errors:
        print(f"\n{len(errors)} invoice(s) with errors:")
        for r in errors:
            print(f"  {r['invoice_id']}: {r['correct']}/{r['total']}")
    else:
        print("\nAll invoices extracted perfectly.")


if __name__ == "__main__":
    print(f"Evaluating {len(list(DATA_DIR.glob('*.pdf')))} invoices...\n")
    evaluate()