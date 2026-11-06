import type { FieldView } from "./types";

// Mise en forme d'affichage seulement : la valeur normalisée validée par l'API
// n'est jamais recalculée, uniquement écrite à la marocaine (virgule, DH).
const money = new Intl.NumberFormat("fr-MA", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const longDate = new Intl.DateTimeFormat("fr-FR", { day: "numeric", month: "long", year: "numeric", timeZone: "UTC" });

const MONEY_FIELDS = new Set(["total_cash", "total_check", "amount_before_tax", "taxes", "previous_unpaid", "out_of_plan_amount"]);
const DATE_FIELDS = new Set(["due_date", "invoice_date"]);

const TARIFF_LABELS: Record<string, string> = {
  BT: "Basse tension",
  BT_DOMESTIQUE: "Basse tension — usage domestique",
  MT_GENERAL: "Moyenne tension — usage général (professionnel)",
};

export function isReadable(field: FieldView | undefined): field is FieldView {
  return !!field && (field.status === "extracted" || field.status === "derived") && field.value !== null && field.value !== undefined;
}

export function formatField(field: FieldView): string {
  if (!isReadable(field)) return "Non lisible";
  if (MONEY_FIELDS.has(field.name) && typeof field.value === "number") return `${money.format(field.value)} DH`;
  if (DATE_FIELDS.has(field.name) && typeof field.value === "string" && /^\d{4}-\d{2}-\d{2}$/.test(field.value)) {
    return longDate.format(new Date(`${field.value}T00:00:00Z`));
  }
  if (field.name === "tariff") return TARIFF_LABELS[String(field.value)] ?? String(field.value);
  return field.display_value || String(field.value);
}

// Les identifiants de preuve ([invoice.due_date], [onee-…]) servent à l'audit ;
// les sources restent affichées sous la réponse, avec leur lien.
export function withoutCitationIds(answer: string): string {
  return answer.replace(/\s*\[(?:invoice\.[a-z_]+|[a-z0-9]+(?:-+[a-z0-9]+)+)\]/g, "").trim();
}

export function confidenceLabel(value: number) {
  if (value >= 0.85) return "Confiance élevée";
  if (value >= 0.7) return "À vérifier";
  return "Lecture incertaine";
}
