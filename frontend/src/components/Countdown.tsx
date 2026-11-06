import type { Strings } from "../i18n";

const DAY = 24 * 60 * 60 * 1000;

/** Jours entre aujourd'hui (date locale) et la date limite ISO, sans heure. */
export function daysUntil(iso: string, today = new Date()): number {
  const [year, month, day] = iso.split("-").map(Number);
  const due = Date.UTC(year, month - 1, day);
  const now = Date.UTC(today.getFullYear(), today.getMonth(), today.getDate());
  return Math.round((due - now) / DAY);
}

// Dans un texte iCalendar, virgules et points-virgules sont des séparateurs (RFC 5545).
const icsText = (value: string) => value.replace(/\\/g, "\\\\").replace(/([,;])/g, "\\$1").replace(/\n/g, "\\n");

function reminderFile(iso: string, rawTitle: string, rawDescription: string): string {
  const title = icsText(rawTitle);
  const description = icsText(rawDescription);
  const date = iso.replaceAll("-", "");
  const stamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d{3}/, "");
  // Fichier calendrier standard, généré dans le navigateur : rien n'est envoyé.
  return [
    "BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//QraLia//Rappel//FR", "BEGIN:VEVENT",
    `UID:qralia-${date}-${Math.random().toString(36).slice(2)}@qralia.local`, `DTSTAMP:${stamp}`,
    `DTSTART;VALUE=DATE:${date}`, `SUMMARY:${title}`, `DESCRIPTION:${description}`,
    "BEGIN:VALARM", "TRIGGER:-P2D", "ACTION:DISPLAY", `DESCRIPTION:${title}`, "END:VALARM",
    "END:VEVENT", "END:VCALENDAR",
  ].join("\r\n");
}

export default function Countdown({ dueIso, s, provider, amount }: { dueIso: string; s: Strings; provider: string; amount: string | null }) {
  const days = daysUntil(dueIso);
  const tone = days < 0 ? "late" : days <= 7 ? "soon" : "ok";
  const text = days < 0 ? s.countdown.late(-days) : days === 0 ? s.countdown.today : days === 1 ? s.countdown.tomorrow : s.countdown.left(days);

  function remind() {
    const title = s.countdown.reminderTitle(provider, amount);
    const blob = new Blob([reminderFile(dueIso, title, s.countdown.reminderNote)], { type: "text/calendar;charset=utf-8" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `qralia-rappel-${dueIso}.ics`;
    link.click();
    URL.revokeObjectURL(link.href);
  }

  return (
    <div className={`countdown ${tone}`} role="status">
      <span className="countdown-dot" aria-hidden="true" />
      <strong>{text}</strong>
      {days >= 0 && <button type="button" className="remind" onClick={remind}>📅 {s.countdown.remind}</button>}
    </div>
  );
}
