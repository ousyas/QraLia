import type { Strings } from "./i18n";
import type { FieldView } from "./types";

// Mise en forme d'affichage seulement : la valeur normalisée validée par l'API
// n'est jamais recalculée, uniquement écrite à la marocaine (virgule, DH).
const money = new Intl.NumberFormat("fr-MA", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

const MONEY_FIELDS = new Set(["total_cash", "total_check", "amount_before_tax", "taxes", "previous_unpaid", "out_of_plan_amount"]);
const DATE_FIELDS = new Set(["due_date", "invoice_date"]);

export function isReadable(field: FieldView | undefined): field is FieldView {
  return !!field && (field.status === "extracted" || field.status === "derived") && field.value !== null && field.value !== undefined;
}

export function formatDate(iso: string, s: Strings): string {
  // Chiffres latins dans toutes les langues, comme sur les factures marocaines.
  return new Intl.DateTimeFormat(s.locale, { day: "numeric", month: "long", year: "numeric", timeZone: "UTC", numberingSystem: "latn" })
    .format(new Date(`${iso}T00:00:00Z`));
}

export function formatField(field: FieldView, s: Strings): string {
  if (!isReadable(field)) return s.confidence.notReadable;
  if (MONEY_FIELDS.has(field.name) && typeof field.value === "number") return `${money.format(field.value)} ${s.currency}`;
  if (DATE_FIELDS.has(field.name) && typeof field.value === "string" && /^\d{4}-\d{2}-\d{2}$/.test(field.value)) {
    return formatDate(field.value, s);
  }
  if (field.name === "tariff") return s.tariffs[String(field.value)] ?? String(field.value);
  return field.display_value || String(field.value);
}

export function fieldLabel(field: FieldView, s: Strings): string {
  return s.fields[field.name] ?? field.label;
}

// Les identifiants de preuve ([invoice.due_date], [onee-…]) servent à l'audit ;
// les sources restent affichées sous la réponse, avec leur lien.
export function withoutCitationIds(answer: string): string {
  return answer.replace(/\s*\[(?:invoice\.[a-z_]+|[a-z0-9]+(?:-+[a-z0-9]+)+)\]/g, "").trim();
}

export function confidenceLabel(value: number, s: Strings) {
  if (value >= 0.85) return s.confidence.high;
  if (value >= 0.7) return s.confidence.check;
  return s.confidence.uncertain;
}
