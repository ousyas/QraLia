// Emblème « Tuile zellige » : un carreau bleu-vert portant l'étoile marocaine
// à huit branches en safran ; au centre, un Q en forme de loupe, dont l'œil
// terracotta rappelle que QraLia lit la facture pour vous.
export function Emblem({ size = 40, className = "" }: { size?: number; className?: string }) {
  return (
    <svg className={`emblem ${className}`} width={size} height={size} viewBox="0 0 120 120" aria-hidden="true">
      <rect className="emblem-tile" x="4" y="4" width="112" height="112" rx="28" />
      <g className="emblem-star">
        <rect x="30" y="30" width="60" height="60" rx="8" />
        <rect x="30" y="30" width="60" height="60" rx="8" transform="rotate(45 60 60)" />
      </g>
      <circle className="emblem-q" cx="57" cy="57" r="17" />
      <path className="emblem-q" d="M68 68 82 82" />
      <circle className="emblem-eye" cx="57" cy="57" r="5.5" />
    </svg>
  );
}

export default function Logo() {
  return (
    <a className="brand" href="#top" aria-label="QraLia — قرا ليا" dir="ltr">
      <Emblem />
      <span className="brand-text">
        <span className="brand-name">Qra<em>Lia</em></span>
        <span className="brand-ar" lang="ary">قرا ليا</span>
      </span>
    </a>
  );
}
