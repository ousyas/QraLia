import { KeyboardEvent, ReactNode, useEffect, useRef, useState } from "react";
import type { Strings } from "../i18n";
import { Emblem } from "./Logo";

const STORAGE_KEY = "qralia.guide.seen";

export function guideAlreadySeen(): boolean {
  try { return window.localStorage.getItem(STORAGE_KEY) === "1"; } catch { return false; }
}

function rememberGuideSeen() {
  try { window.localStorage.setItem(STORAGE_KEY, "1"); } catch { /* navigation privée : on ignore */ }
}

// Les illustrations restent communes ; titres et textes viennent des traductions.
const ART: ReactNode[] = [
  <div className="art art-welcome"><Emblem size={112} /></div>,
  <div className="art art-upload"><span className="art-phone"><span className="art-paper small"><i /><i /></span></span><span className="art-arrow">↑</span></div>,
  <div className="art art-summary"><span className="art-tile dark">889,20 DH</span><span className="art-tile">🔊</span></div>,
  <div className="art art-chat"><span className="bubble me">شحال خاصني نخلص؟</span><span className="bubble bot">889,20 DH · 27/08</span></div>,
  <div className="art art-privacy"><span className="art-lock">🔒</span></div>,
];

export default function Guide({ onClose, s }: { onClose: () => void; s: Strings }) {
  const [index, setIndex] = useState(0);
  const dialog = useRef<HTMLDivElement>(null);
  const primary = useRef<HTMLButtonElement>(null);
  const slides = s.guide.slides;
  const last = index === slides.length - 1;
  const slide = slides[index];
  const rtl = document.documentElement.dir === "rtl";

  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = ""; previous?.focus(); };
  }, []);
  useEffect(() => { primary.current?.focus(); }, [index]);

  function close() { rememberGuideSeen(); onClose(); }

  function handleKey(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === "Escape") close();
    // En arabe, la flèche gauche fait avancer.
    const forward = rtl ? "ArrowLeft" : "ArrowRight";
    const back = rtl ? "ArrowRight" : "ArrowLeft";
    if (event.key === forward && !last) setIndex(index + 1);
    if (event.key === back && index > 0) setIndex(index - 1);
    if (event.key === "Tab") {
      // Garde le focus dans la fenêtre du guide.
      const focusable = dialog.current?.querySelectorAll<HTMLElement>("button");
      if (!focusable?.length) return;
      const first = focusable[0], end = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); end.focus(); }
      else if (!event.shiftKey && document.activeElement === end) { event.preventDefault(); first.focus(); }
    }
  }

  return (
    <div className="guide-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) close(); }}>
      <div className="guide" role="dialog" aria-modal="true" aria-labelledby="guide-title" ref={dialog} onKeyDown={handleKey}>
        <button className="guide-skip" onClick={close}>{s.guide.skip}</button>
        <div className="guide-art" key={`art-${index}`}>{ART[index]}</div>
        <div className="guide-body" key={`body-${index}`}>
          <p className="guide-count">{s.guide.stepOf(index + 1, slides.length)}</p>
          <h2 id="guide-title">{slide.title}</h2>
          <p>{slide.text}</p>
        </div>
        <div className="guide-dots" aria-hidden="true">
          {slides.map((item, dot) => <span key={item.title} className={dot === index ? "on" : ""} />)}
        </div>
        <div className="guide-actions">
          <button className="ghost" onClick={() => setIndex(index - 1)} disabled={index === 0}>{s.guide.prev}</button>
          <button className="primary" ref={primary} onClick={() => (last ? close() : setIndex(index + 1))}>{last ? s.guide.start : s.guide.next}</button>
        </div>
      </div>
    </div>
  );
}
