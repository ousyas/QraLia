import { KeyboardEvent, ReactNode, useEffect, useRef, useState } from "react";

const STORAGE_KEY = "qralia.guide.seen";

export function guideAlreadySeen(): boolean {
  try { return window.localStorage.getItem(STORAGE_KEY) === "1"; } catch { return false; }
}

function rememberGuideSeen() {
  try { window.localStorage.setItem(STORAGE_KEY, "1"); } catch { /* navigation privée : on ignore */ }
}

type Slide = { title: string; text: ReactNode; art: ReactNode };

const SLIDES: Slide[] = [
  {
    title: "Bienvenue sur QraLia",
    text: <>QraLia lit votre facture à votre place et vous dit l'essentiel : <strong>combien payer</strong> et <strong>avant quelle date</strong>. Électricité ou abonnement téléphone.</>,
    art: <div className="art art-welcome"><span className="art-paper"><i /><i /><i /></span><span className="art-badge">✓</span></div>,
  },
  {
    title: "1. Ajoutez votre facture",
    text: <>Prenez une <strong>photo bien droite</strong>, à la lumière, ou choisissez un PDF. Pour ce prototype, utilisez seulement une facture d'exemple, jamais une vraie.</>,
    art: <div className="art art-upload"><span className="art-phone"><span className="art-paper small"><i /><i /></span></span><span className="art-arrow">↑</span></div>,
  },
  {
    title: "2. Lisez le résumé",
    text: <>Le montant et la date s'affichent en grand. <span className="tag ok">Confiance élevée</span> : lecture sûre. <span className="tag warn">À vérifier</span> : demandez à un proche de regarder le papier.</>,
    art: <div className="art art-summary"><span className="art-tile dark">889,20 DH</span><span className="art-tile">27 août</span></div>,
  },
  {
    title: "3. Posez vos questions",
    text: <>Touchez une question proposée ou écrivez la vôtre, en <strong>français, arabe ou darija</strong>. QraLia montre ses sources et préfère dire « je ne sais pas » plutôt que deviner.</>,
    art: <div className="art art-chat"><span className="bubble me">Chhal khassni nkhalas ?</span><span className="bubble bot">889,20 DH, avant le 27 août.</span></div>,
  },
  {
    title: "Vos informations restent à vous",
    text: <>Le fichier est supprimé juste après la lecture. Les informations lues sont effacées après <strong>30 minutes</strong>, ou tout de suite quand vous touchez « Supprimer ». Revoyez ce guide à tout moment avec le bouton <strong>Guide</strong>.</>,
    art: <div className="art art-privacy"><span className="art-lock">🔒</span></div>,
  },
];

export default function Guide({ onClose }: { onClose: () => void }) {
  const [index, setIndex] = useState(0);
  const dialog = useRef<HTMLDivElement>(null);
  const primary = useRef<HTMLButtonElement>(null);
  const last = index === SLIDES.length - 1;
  const slide = SLIDES[index];

  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = ""; previous?.focus(); };
  }, []);
  useEffect(() => { primary.current?.focus(); }, [index]);

  function close() { rememberGuideSeen(); onClose(); }

  function handleKey(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === "Escape") close();
    if (event.key === "ArrowRight" && !last) setIndex(index + 1);
    if (event.key === "ArrowLeft" && index > 0) setIndex(index - 1);
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
        <button className="guide-skip" onClick={close}>Passer</button>
        <div className="guide-art" key={`art-${index}`}>{slide.art}</div>
        <div className="guide-body" key={`body-${index}`}>
          <p className="guide-count">Étape {index + 1} sur {SLIDES.length}</p>
          <h2 id="guide-title">{slide.title}</h2>
          <p>{slide.text}</p>
        </div>
        <div className="guide-dots" aria-hidden="true">
          {SLIDES.map((item, dot) => <span key={item.title} className={dot === index ? "on" : ""} />)}
        </div>
        <div className="guide-actions">
          <button className="ghost" onClick={() => setIndex(index - 1)} disabled={index === 0}>Précédent</button>
          <button className="primary" ref={primary} onClick={() => (last ? close() : setIndex(index + 1))}>{last ? "Commencer" : "Suivant"}</button>
        </div>
      </div>
    </div>
  );
}
