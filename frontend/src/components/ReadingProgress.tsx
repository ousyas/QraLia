import { useEffect, useState } from "react";

// L'API ne renvoie pas d'avancement : ces étapes décrivent le traitement réel,
// mais leur rythme est indicatif. La dernière reste active jusqu'à la réponse.
const STEPS = [
  "Lecture du texte de la facture",
  "Reconnaissance du fournisseur",
  "Vérification des montants et des dates",
  "Préparation du résumé",
];

export default function ReadingProgress({ fileName }: { fileName: string }) {
  const [step, setStep] = useState(0);

  useEffect(() => {
    const timer = window.setInterval(() => setStep((current) => Math.min(current + 1, STEPS.length - 1)), 1700);
    return () => window.clearInterval(timer);
  }, []);

  return (
    <div className="reading" role="status" aria-live="polite">
      <div className="reading-doc" aria-hidden="true">
        <span className="line w60" /><span className="line w80" /><span className="line w40" />
        <span className="line w70" /><span className="line w50" /><span className="line w80" />
        <div className="scan-beam" />
      </div>
      <div className="reading-copy">
        <strong>QraLia lit « {fileName} »</strong>
        <ol>
          {STEPS.map((label, index) => <li key={label} className={index < step ? "done" : index === step ? "active" : ""}>{label}</li>)}
        </ol>
      </div>
    </div>
  );
}
