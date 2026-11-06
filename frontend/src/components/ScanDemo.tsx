// Facture « lue » en boucle dans l'en-tête. Avant le dépôt, c'est un exemple
// fictif ; après la lecture, les lignes et les étiquettes reprennent les
// valeurs réellement extraites de la facture de l'utilisateur.
export type ScanView = {
  kind: string;
  provider: string;
  icon: string;
  rows: Array<[string, string]>;
  total: string;
  due: string;
  amountChip: string;
  dateChip: string;
  amountSure: boolean;
  dateSure: boolean;
};

export const EXAMPLE_SCAN: Omit<ScanView, "dateChip"> = {
  kind: "Facture d'électricité",
  provider: "ONEE",
  icon: "⚡",
  rows: [["Total HT", "780,00"], ["Taxes", "109,20"]],
  total: "889,20",
  due: "27.08.2026",
  amountChip: "889,20 DH",
  amountSure: true,
  dateSure: true,
};

export default function ScanDemo({ view, toPay, before, caption }: { view: ScanView; toPay: string; before: string; caption: string }) {
  return (
    <figure className="scan-demo" aria-hidden="true">
      {/* La facture est en français : elle reste de gauche à droite. */}
      <div className="paper" dir="ltr">
        <div className="paper-head"><span className="paper-logo" /><span className="paper-title">{view.kind}</span><span className="paper-ar">فاتورة</span></div>
        <div className="paper-lines">
          <span className="line w60" /><span className="line w40" /><span className="line w70" />
        </div>
        <div className="paper-box">
          {view.rows.map(([label, value]) => <div className="paper-row" key={label}><span>{label}</span><span>{value}</span></div>)}
          <div className="paper-row hit hit-amount"><span>Total à régler</span><strong>{view.total}</strong></div>
          <div className="paper-row hit hit-date"><span>Date d'échéance</span><strong>{view.due}</strong></div>
        </div>
        <div className="paper-lines"><span className="line w80" /><span className="line w50" /></div>
        <div className="scan-beam" />
      </div>
      <div className="chip chip-provider"><i>{view.icon}</i> <bdi>{view.provider}</bdi></div>
      <div className="chip chip-amount"><small>{toPay}</small><bdi>{view.amountChip}</bdi> <b className={view.amountSure ? "" : "unsure"}>{view.amountSure ? "✓" : "!"}</b></div>
      <div className="chip chip-date"><small>{before}</small>{view.dateChip} <b className={view.dateSure ? "" : "unsure"}>{view.dateSure ? "✓" : "!"}</b></div>
      <figcaption>{caption}</figcaption>
    </figure>
  );
}
