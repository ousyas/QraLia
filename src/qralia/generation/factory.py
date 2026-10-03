"""Random generator for InvoiceParams used by the synthetic invoice pipeline."""

import random
import string
from datetime import date, timedelta

from qralia.domain.invoice import InvoiceParams

# Approximate values — to replace with real ONEE grid later (TODO weekend talk with Oussama)
VAT_RATE = 0.14          # Moroccan electricity VAT
STAMP_DUTY = 0.33        # Droit de timbre on cash payments (approximate)
PRICE_PER_KWH = 1.0      # Approximate average BT Domestique price (DH/kWh)

_FIRST_NAMES = [
    "Mohamed", "Fatima", "Youssef", "Khadija", "Ahmed", "Amina", "Omar",
    "Salma", "Hassan", "Nadia", "Sanae", "Rachid", "Imane", "Karim",
    "Souad", "Abdel", "Leila", "Hicham", "Samira", "Nabil",
]
_LAST_NAMES = [
    "Alaoui", "Bennani", "El Fassi", "Idrissi", "Tazi", "Berrada",
    "Chraibi", "Moussaoui", "Lahlou", "Ziani", "Bouhajeb", "El Amrani",
    "Rifi", "Ouazzani", "Bouzid", "El Ghazali", "Haddad", "Jabri",
    "Kabbaj", "Mouline",
]


def random_invoice_params() -> InvoiceParams:
    """Generate a plausible, arithmetically consistent InvoiceParams.

    Primary parameters are drawn at random; derived ones are computed
    so that totals, taxes and date orderings remain coherent.
    """
    # --- Dates (chronology: period_start < period_end < invoice_date < due_date)
    invoice_date = date.today() - timedelta(days=random.randint(0, 365))
    period_end = invoice_date - timedelta(days=random.randint(1, 5))
    period_start = period_end - timedelta(days=random.randint(27, 31))
    due_date = invoice_date + timedelta(days=random.randint(10, 20))

    # --- Consumption and compteur index (integers, like a real meter)
    consumption_kwh = random.randint(100, 1000)
    index_previous = random.randint(0, 10000)
    index_current = index_previous + consumption_kwh

    # --- Tariff type (string from the schema's expected values)
    tariff = random.choice(["BT_DOMESTIQUE", "MT_GENERAL"])
    tariff_tranche = random.choice(["1", "2", "3"])

    # --- Amounts (derived from consumption)
    amount_before_tax = round(consumption_kwh * PRICE_PER_KWH, 2)
    taxes = round(amount_before_tax * VAT_RATE, 2)
    previous_unpaid = round(random.uniform(0, 500), 2) if random.random() < 0.2 else 0.0
    total_ttc = round(amount_before_tax + taxes + previous_unpaid, 2)
    total_check = total_ttc
    total_cash = round(total_ttc + STAMP_DUTY, 2)

    # --- Identity
    invoice_number = "F" + "".join(random.choices(string.digits, k=8))
    contract_number = "".join(random.choices(string.digits, k=10))
    client_name = f"{random.choice(_FIRST_NAMES)} {random.choice(_LAST_NAMES)}"
    client_address = "Rue " + "".join(random.choices(string.digits, k=2)) + ", Rabat"

    return InvoiceParams(
        invoice_number=invoice_number,
        contract_number=contract_number,
        client_name=client_name,
        client_address=client_address,
        invoice_date=invoice_date,
        due_date=due_date,
        period_start=period_start,
        period_end=period_end,
        consumption_kwh=consumption_kwh,
        amount_before_tax=amount_before_tax,
        taxes=taxes,
        total_ttc=total_ttc,
        previous_unpaid=previous_unpaid,
        total_cash=total_cash,
        total_check=total_check,
        tariff=tariff,
        tariff_tranche=tariff_tranche,
        index_previous=index_previous,
        index_current=index_current,
    )