// Emblème : étoile zellige à huit branches, et au centre un « Q » qui se lit
// aussi comme une loupe posée sur un œil — QraLia lit pour vous.
export function Emblem({ size = 52 }: { size?: number }) {
  return (
    <svg className="emblem" width={size} height={size} viewBox="0 0 48 48" aria-hidden="true">
      <g className="emblem-star">
        <rect x="9" y="9" width="30" height="30" rx="4" />
        <rect x="9" y="9" width="30" height="30" rx="4" transform="rotate(45 24 24)" />
      </g>
      <circle className="emblem-ring" cx="23" cy="23" r="8.6" />
      <path className="emblem-ring" d="M29.2 29.2 35 35" />
      <circle className="emblem-eye" cx="23" cy="23" r="3.1" />
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
