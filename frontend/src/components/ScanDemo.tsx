// Démonstration décorative : une facture fictive est « lue » en boucle et ses
// informations clés en ressortent. Aucune donnée réelle, purement visuel.
export default function ScanDemo({ toPay, before, date, caption }: { toPay: string; before: string; date: string; caption: string }) {
  return (
    <figure className="scan-demo" aria-hidden="true">
      {/* La facture d'exemple est en français : elle reste de gauche à droite. */}
      <div className="paper" dir="ltr">
        <div className="paper-head"><span className="paper-logo" /><span className="paper-title">Facture d'électricité</span><span className="paper-ar">فاتورة</span></div>
        <div className="paper-lines">
          <span className="line w60" /><span className="line w40" /><span className="line w70" />
        </div>
        <div className="paper-box">
          <div className="paper-row"><span>Total HT</span><span>780,00</span></div>
          <div className="paper-row"><span>Taxes</span><span>109,20</span></div>
          <div className="paper-row hit hit-amount"><span>Total à régler</span><strong>889,20</strong></div>
          <div className="paper-row hit hit-date"><span>Date d'échéance</span><strong>27.08.2026</strong></div>
        </div>
        <div className="paper-lines"><span className="line w80" /><span className="line w50" /></div>
        <div className="scan-beam" />
      </div>
      <div className="chip chip-provider" aria-hidden="true"><i>⚡</i> ONEE</div>
      <div className="chip chip-amount" aria-hidden="true"><small>{toPay}</small><bdi>889,20 DH</bdi> <b>✓</b></div>
      <div className="chip chip-date" aria-hidden="true"><small>{before}</small>{date} <b>✓</b></div>
      <figcaption>{caption}</figcaption>
    </figure>
  );
}
