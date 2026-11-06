import { useEffect, useState } from "react";
import type { PagePreview } from "../types";

export type Spot = { key: string; label: string; value: string; tone: "amount" | "date"; bbox: [number, number, number, number] };

// Les bbox viennent de l'OCR, dans les pixels de la page lue : on les ramène en
// pourcentages pour qu'elles suivent l'image quelle que soit sa taille affichée.
const pct = (value: number, total: number) => `${(value / total) * 100}%`;

function useCompact() {
  const query = "(max-width: 650px)";
  const [compact, setCompact] = useState(() => window.matchMedia(query).matches);
  useEffect(() => {
    const media = window.matchMedia(query);
    const update = () => setCompact(media.matches);
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);
  return compact;
}

function Loupe({ spot, preview, compact }: { spot: Spot; preview: PagePreview; compact: boolean }) {
  const [x1, y1, x2, y2] = spot.bbox;
  const boxHeight = Math.max(y2 - y1, 20);
  // Sur grand écran, la bande part de la marge gauche pour inclure le libellé
  // imprimé sur la même ligne. Sur téléphone, elle se resserre autour de la
  // valeur, sinon le texte deviendrait trop petit pour être lu.
  const compactW = Math.min(preview.width, Math.max((x2 - x1) * 3.2, preview.width * 0.3));
  const cropX = compact
    ? Math.max(0, Math.min(x2 + (x2 - x1) * 0.4 - compactW, preview.width - compactW))
    : Math.max(0, Math.min(x1 - 40, preview.width * 0.05));
  const cropW = compact ? compactW : Math.min(preview.width - cropX, x2 + preview.width * 0.03 - cropX);
  const cropH = boxHeight * 3.2;
  const cropY = Math.max(0, Math.min(y1 - boxHeight * 1.1, preview.height - cropH));
  const position = (offset: number, span: number, total: number) => (total === span ? 0 : (offset / (total - span)) * 100);
  return (
    <figure className={`loupe ${spot.tone}`}>
      <figcaption><span>{spot.label}</span><strong>{spot.value}</strong></figcaption>
      <div
        className="loupe-strip"
        role="img"
        aria-label={`${spot.label} : ${spot.value}`}
        style={{
          aspectRatio: `${cropW} / ${cropH}`,
          backgroundImage: `url(${preview.data_url})`,
          backgroundSize: `${(preview.width / cropW) * 100}% auto`,
          backgroundPosition: `${position(cropX, cropW, preview.width)}% ${position(cropY, cropH, preview.height)}%`,
        }}
      >
        <span
          className="loupe-mark"
          style={{
            left: pct(x1 - cropX - 10, cropW), top: pct(y1 - cropY - 8, cropH),
            width: pct(x2 - x1 + 20, cropW), height: pct(y2 - y1 + 16, cropH),
          }}
        />
      </div>
    </figure>
  );
}

export default function ShowOnInvoice({ preview, spots, title, hint, fullPage }: {
  preview: PagePreview; spots: Spot[]; title: string; hint: string; fullPage: string;
}) {
  const compact = useCompact();
  if (!spots.length) return null;
  return (
    <section className="show-card" aria-label={title}>
      <h2>{title}</h2>
      <p className="show-hint">{hint}</p>
      <div className="loupes">{spots.map((spot) => <Loupe key={spot.key} spot={spot} preview={preview} compact={compact} />)}</div>
      <details className="show-page">
        <summary>{fullPage}</summary>
        <div className="page-frame" style={{ aspectRatio: `${preview.width} / ${preview.height}` }}>
          <img src={preview.data_url} alt="" />
          {spots.map((spot, index) => {
            const [x1, y1, x2, y2] = spot.bbox;
            return <span key={spot.key} className={`page-mark ${spot.tone}`} style={{
              left: pct(x1 - 12, preview.width), top: pct(y1 - 10, preview.height),
              width: pct(x2 - x1 + 24, preview.width), height: pct(y2 - y1 + 20, preview.height),
            }}><i>{index + 1}</i></span>;
          })}
        </div>
      </details>
    </section>
  );
}
