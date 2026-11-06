import { useEffect, useState } from "react";

// Synthèse vocale du navigateur : rien n'est envoyé à un service externe par
// QraLia. Il n'existe pas de voix darija ; l'écriture arabe est lue avec une
// voix arabe, la darija latine avec une voix française.
const VOICE_LANGS: Record<string, string[]> = {
  fr: ["fr-FR", "fr-CA", "fr"],
  ar: ["ar-MA", "ar-SA", "ar-AE", "ar-EG", "ar"],
};

export const speechSupported = typeof window !== "undefined" && "speechSynthesis" in window;

function pickVoice(lang: "fr" | "ar"): SpeechSynthesisVoice | undefined {
  const voices = window.speechSynthesis.getVoices();
  for (const wanted of VOICE_LANGS[lang]) {
    const voice = voices.find((item) => item.lang.toLowerCase().startsWith(wanted.toLowerCase()));
    if (voice) return voice;
  }
  return undefined;
}

export function speechLang(text: string): "fr" | "ar" {
  return /[؀-ۿ]/.test(text) ? "ar" : "fr";
}

export default function SpeakButton({ text, label, stopLabel, className = "" }: {
  text: string; label: string; stopLabel: string; className?: string;
}) {
  const [speaking, setSpeaking] = useState(false);

  // Une lecture en cours s'arrête si le bouton disparaît (nouvelle facture…).
  useEffect(() => () => { if (speechSupported) window.speechSynthesis.cancel(); }, []);

  if (!speechSupported || !text.trim()) return null;

  function toggle() {
    const synth = window.speechSynthesis;
    if (speaking) { synth.cancel(); setSpeaking(false); return; }
    synth.cancel();
    const lang = speechLang(text);
    const utterance = new SpeechSynthesisUtterance(text);
    const voice = pickVoice(lang);
    if (voice) utterance.voice = voice;
    utterance.lang = voice?.lang ?? VOICE_LANGS[lang][0];
    utterance.rate = 0.92;
    utterance.onend = () => setSpeaking(false);
    utterance.onerror = () => setSpeaking(false);
    setSpeaking(true);
    synth.speak(utterance);
  }

  return (
    <button type="button" className={`speak ${speaking ? "on" : ""} ${className}`} onClick={toggle} aria-pressed={speaking}>
      <span aria-hidden="true">{speaking ? "■" : "🔊"}</span> {speaking ? stopLabel : label}
    </button>
  );
}
