import { FormEvent, useEffect, useMemo, useState } from "react";
import { checkHealth, createConversation, deleteInvoice, sendConversationMessage, uploadInvoice } from "./api";
import type { ConversationMessageResponse, InvoiceResponse } from "./types";

const routeLabels = {
  invoice: "Facture",
  official_rag: "Source officielle",
  both: "Facture + source officielle",
  unknown: "Réponse non trouvée",
};

function confidenceLabel(value: number) {
  if (value >= 0.85) return "Confiance élevée";
  if (value >= 0.7) return "À vérifier";
  return "Lecture incertaine";
}

export default function App() {
  const [online, setOnline] = useState<boolean | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [confirmed, setConfirmed] = useState(false);
  const [invoice, setInvoice] = useState<InvoiceResponse | null>(null);
  const [question, setQuestion] = useState("");
  const [language, setLanguage] = useState("auto");
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Array<{ question: string; result: ConversationMessageResponse }>>([]);
  const [busy, setBusy] = useState<"upload" | "question" | "delete" | null>(null);
  const [error, setError] = useState("");

  useEffect(() => { void checkHealth().then(setOnline); }, []);

  const usefulFields = useMemo(
    () => invoice?.fields.filter((field) => field.status !== "not_present") ?? [],
    [invoice],
  );

  async function handleUpload(event: FormEvent) {
    event.preventDefault();
    if (!file || !confirmed) return;
    setBusy("upload"); setError(""); setMessages([]); setConversationId(null);
    try {
      const uploaded = await uploadInvoice(file);
      const conversation = await createConversation(uploaded.invoice_id, language);
      setInvoice(uploaded);
      setConversationId(conversation.conversation_id);
    }
    catch (cause) { setError(cause instanceof Error ? cause.message : "Impossible de lire la facture."); }
    finally { setBusy(null); }
  }

  async function handleQuestion(event: FormEvent) {
    event.preventDefault();
    if (!invoice || !question.trim()) return;
    setBusy("question"); setError("");
    try {
      const activeConversation = conversationId ?? (await createConversation(invoice.invoice_id, language)).conversation_id;
      setConversationId(activeConversation);
      const currentQuestion = question.trim();
      const result = await sendConversationMessage(activeConversation, currentQuestion, language);
      setMessages((current) => [...current, { question: currentQuestion, result }]);
      setQuestion("");
    }
    catch (cause) { setError(cause instanceof Error ? cause.message : "Impossible de répondre."); }
    finally { setBusy(null); }
  }

  async function handleDelete() {
    if (!invoice) return;
    setBusy("delete"); setError("");
    try {
      await deleteInvoice(invoice.invoice_id);
      setInvoice(null); setConversationId(null); setMessages([]); setQuestion(""); setFile(null); setConfirmed(false);
    } catch (cause) { setError(cause instanceof Error ? cause.message : "Suppression impossible."); }
    finally { setBusy(null); }
  }

  return (
    <main>
      <header className="hero">
        <nav aria-label="Navigation principale">
          <a className="brand" href="#top" aria-label="Accueil QraLia"><span>Q</span>raLia</a>
          <span className={`status ${online ? "online" : "offline"}`}>
            {online === null ? "Vérification…" : online ? "Service disponible" : "Service indisponible"}
          </span>
        </nav>
        <div className="hero-copy" id="top">
          <p className="eyebrow">Comprendre sans deviner</p>
          <h1>Votre facture,<br /><em>en mots simples.</em></h1>
          <p>Déposez une facture synthétique. QraLia lit les informations utiles, signale les incertitudes et cite les règles officielles.</p>
        </div>
      </header>

      <section className="workspace" aria-live="polite">
        <div className="step-heading"><span>01</span><div><h2>Ajouter une facture</h2><p>PDF, PNG ou JPEG — 10 Mo maximum</p></div></div>
        <form className="upload-card" onSubmit={handleUpload}>
          <label className="dropzone">
            <input type="file" accept=".pdf,.png,.jpg,.jpeg" onChange={(event) => setFile(event.target.files?.[0] ?? null)} />
            <span className="upload-icon" aria-hidden="true">↑</span>
            <strong>{file ? file.name : "Choisir une facture"}</strong>
            <small>{file ? `${(file.size / 1024 / 1024).toFixed(2)} Mo` : "Touchez ici ou glissez votre fichier"}</small>
          </label>
          <label className="consent"><input type="checkbox" checked={confirmed} onChange={(event) => setConfirmed(event.target.checked)} /> Cette facture est entièrement synthétique et ne contient aucune donnée personnelle réelle.</label>
          <button className="primary" disabled={!file || !confirmed || busy !== null}>{busy === "upload" ? "Lecture en cours…" : "Lire la facture"}</button>
        </form>

        {error && <div className="alert error" role="alert">{error}</div>}

        {invoice && <>
          <div className="step-heading"><span>02</span><div><h2>Informations détectées</h2><p>Vérifiez toujours les champs marqués comme incertains.</p></div></div>
          <div className={`alert ${invoice.validation.valid && invoice.validation.scope_supported ? "success" : "warning"}`}>
            <strong>{!invoice.validation.valid ? "Vérification nécessaire" : invoice.validation.scope_supported ? "Contrôles cohérents" : "Document hors périmètre résidentiel"}</strong>
            <span>{Math.round(invoice.document_confidence * 100)} % de confiance globale</span>
          </div>
          {invoice.validation.warnings.includes("outside_mvp_professional_tariff") && <div className="alert warning">Les champs peuvent être consultés, mais QraLia n'appliquera pas les règles résidentielles à cette facture professionnelle.</div>}
          <div className="field-grid">
            {usefulFields.map((field) => <article className={`field-card confidence-${field.confidence >= 0.85 ? "high" : "low"}`} key={field.name}>
              <div><p>{field.label}</p><strong>{field.display_value || "Non lisible"}</strong></div>
              <span>{confidenceLabel(field.confidence)}</span>
            </article>)}
          </div>

          <div className="step-heading"><span>03</span><div><h2>Discuter avec QraLia</h2><p>Vous pouvez poser une question, puis continuer avec « Et avant quelle date ? ».</p></div></div>
          {messages.length > 0 && <section className="chat-thread" aria-label="Conversation avec QraLia">
            {messages.map(({ question: userQuestion, result }) => <div className="chat-exchange" key={result.user_message_id}>
              <div className="chat-message user-message"><span>Vous</span><p>{userQuestion}</p></div>
              <div className="chat-message assistant-message">
                <span>QraLia</span>
                {result.resolution.action === "rewrite" && result.resolution.standalone_question && <small className="rewrite-note">Question comprise comme : {result.resolution.standalone_question}</small>}
                {result.clarification_question
                  ? <p>{result.clarification_question}</p>
                  : result.answer && <>
                    <div className="answer-meta"><span>{routeLabels[result.answer.route]}</span><span>{Math.round(result.answer.confidence * 100)} % de confiance</span></div>
                    <p>{result.answer.answer}</p>
                    {result.answer.citations.length > 0 && <div className="sources"><strong>Sources officielles</strong>{result.answer.citations.map((citation) => <a href={citation.url} target="_blank" rel="noreferrer" key={citation.chunk_id}>{citation.title} ↗</a>)}</div>}
                    {result.answer.warnings.map((warning) => <small className="answer-warning" key={warning}>{warning}</small>)}
                  </>}
              </div>
            </div>)}
          </section>}
          <form className="question-card" onSubmit={handleQuestion}>
            <label htmlFor="language">Langue souhaitée</label>
            <select id="language" value={language} onChange={(event) => setLanguage(event.target.value)}>
              <option value="auto">Détection automatique</option><option value="fr">Français</option>
              <option value="ar">العربية الفصحى</option><option value="darija_arab">الدارجة</option><option value="darija_latn">Darija latine</option>
            </select>
            <label htmlFor="question">Votre question</label>
            <textarea id="question" value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Exemple : Combien dois-je payer ? Puis : Et avant quelle date ?" rows={3} maxLength={500} />
            <button className="primary" disabled={!question.trim() || busy !== null}>{busy === "question" ? "Recherche des preuves…" : messages.length ? "Envoyer" : "Commencer la conversation"}</button>
          </form>

          <button className="delete" onClick={handleDelete} disabled={busy !== null}>{busy === "delete" ? "Suppression…" : "Supprimer cette facture de la session"}</button>
        </>}
      </section>

      <footer><strong>QraLia</strong><p>Prototype pédagogique — aucune donnée réelle ne doit être téléversée.</p></footer>
    </main>
  );
}
