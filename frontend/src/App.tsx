import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import { checkHealth, createConversation, deleteInvoice, sendConversationMessage, uploadInvoice } from "./api";
import { confidenceLabel, formatField, isReadable, withoutCitationIds } from "./format";
import type { ConversationMessageResponse, FieldView, InvoiceResponse } from "./types";

const routeLabels = {
  invoice: "Facture",
  official_rag: "Source officielle",
  both: "Facture + source officielle",
  unknown: "Réponse non trouvée",
};

const SUGGESTED_QUESTIONS = {
  utilities: [
    "Combien dois-je payer ?",
    "Quelle est la date limite de paiement ?",
    "Où puis-je payer ma facture ?",
    "Puis-je payer par chèque ?",
  ],
  telecom: [
    "Combien dois-je payer ?",
    "Quelle est la date limite de paiement ?",
    "Quelle est mon offre ?",
    "Comment payer ma facture ?",
  ],
};
const TELECOM_PROVIDERS = new Set(["iam", "orange_ma", "inwi"]);

function SummaryValue({ field, label, missing }: { field: FieldView | undefined; label: string; missing: string }) {
  const readable = isReadable(field) && field.confidence >= 0.7;
  return <div className={`summary-value ${readable ? "" : "uncertain"}`}>
    <p>{label}</p>
    <strong>{readable ? formatField(field) : missing}</strong>
    <span>{readable ? confidenceLabel(field.confidence) : "Demandez l'aide d'un proche pour vérifier sur le papier"}</span>
  </div>;
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
  const [pending, setPending] = useState<string | null>(null);
  const [busy, setBusy] = useState<"upload" | "question" | "delete" | null>(null);
  const [error, setError] = useState("");

  const chatEnd = useRef<HTMLDivElement>(null);

  useEffect(() => { void checkHealth().then(setOnline); }, []);
  // Sur mobile, la dernière réponse doit rester visible au-dessus du champ fixe.
  useEffect(() => {
    if (messages.length || pending) chatEnd.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length, pending]);

  const fields = useMemo(
    () => new Map((invoice?.fields ?? []).map((field) => [field.name, field])),
    [invoice],
  );
  const detailFields = useMemo(
    () => invoice?.fields.filter((field) => field.status !== "not_present" && !["total_check", "due_date"].includes(field.name)) ?? [],
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

  async function ask(text: string) {
    const currentQuestion = text.trim();
    if (!invoice || !currentQuestion || busy) return;
    setBusy("question"); setError(""); setPending(currentQuestion); setQuestion("");
    try {
      const activeConversation = conversationId ?? (await createConversation(invoice.invoice_id, language)).conversation_id;
      setConversationId(activeConversation);
      const result = await sendConversationMessage(activeConversation, currentQuestion, language);
      setMessages((current) => [...current, { question: currentQuestion, result }]);
    }
    catch (cause) {
      setQuestion(currentQuestion);
      setError(cause instanceof Error ? cause.message : "Impossible de répondre.");
    }
    finally { setBusy(null); setPending(null); }
  }

  function handleQuestion(event: FormEvent) {
    event.preventDefault();
    void ask(question);
  }

  function handleQuestionKey(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void ask(question); }
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

  const cash = fields.get("total_cash");
  const plan = fields.get("plan_name");
  const suggestions = SUGGESTED_QUESTIONS[invoice && TELECOM_PROVIDERS.has(invoice.provider_id) ? "telecom" : "utilities"];
  const professional = invoice?.validation.warnings.includes("outside_mvp_professional_tariff");
  const unknownProvider = invoice?.provider_id === "unknown";

  return (
    <main className={invoice ? "has-invoice" : undefined}>
      <header className="hero">
        <nav aria-label="Navigation principale">
          <a className="brand" href="#top" aria-label="Accueil QraLia"><span>Q</span>raLia</a>
          <span className={`status ${online ? "online" : "offline"}`}>
            {online === null ? "Vérification…" : online ? "Service disponible" : "Service indisponible"}
          </span>
        </nav>
        <div className="hero-copy" id="top">
          <p className="eyebrow">Comprendre sans deviner</p>
          <h1>Votre facture, <em>en mots simples.</em></h1>
          <p>Déposez une facture synthétique d'électricité ou d'abonnement télécom. QraLia lit les informations utiles, signale les incertitudes et cite les règles officielles.</p>
        </div>
      </header>

      <section className="workspace" aria-live="polite">
        {!invoice && <>
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
        </>}

        {error && <div className="alert error" role="alert">{error}</div>}

        {invoice && <>
          <section className="summary-card" aria-label="Résumé de la facture">
            <div className="summary-head">
              <div>
                <p className="provider">{invoice.provider_name ?? "Fournisseur non reconnu"}</p>
                <p className="eyebrow">{invoice.document_kind ?? "Facture"} {isReadable(fields.get("invoice_number")) ? `· ${fields.get("invoice_number")!.display_value}` : ""}</p>
              </div>
              <span className={`badge ${invoice.validation.valid && invoice.validation.scope_supported ? "ok" : "warn"}`}>
                {!invoice.validation.valid ? "Vérification nécessaire" : invoice.validation.scope_supported ? "Contrôles cohérents" : professional ? "Facture professionnelle" : "Hors périmètre"}
              </span>
            </div>
            <div className="summary-values">
              <SummaryValue field={fields.get("total_check")} label="À payer" missing="Montant non lisible" />
              <SummaryValue field={fields.get("due_date")} label="Avant le" missing="Date non lisible" />
            </div>
            {isReadable(plan) && <p className="summary-note">Offre : <strong>{formatField(plan)}</strong></p>}
            {isReadable(cash) && <p className="summary-note">En espèces, timbre inclus : <strong>{formatField(cash)}</strong></p>}
            {professional && <p className="summary-note warn">Les champs peuvent être consultés, mais QraLia n'appliquera pas les règles résidentielles à cette facture professionnelle.</p>}
            {unknownProvider && <p className="summary-note warn">QraLia n'a pas reconnu le fournisseur de cette facture : les champs lus sont affichés, mais aucune règle officielle ne sera appliquée.</p>}
          </section>

          <details className="details-card">
            <summary><span>Voir tous les détails lus sur la facture</span><small>{Math.round(invoice.document_confidence * 100)} % de confiance globale</small></summary>
            <ul className="field-list">
              {detailFields.map((field) => <li className={field.confidence >= 0.85 ? "" : "low"} key={field.name}>
                <span>{field.label}</span>
                <strong>{formatField(field)}</strong>
                <small>{isReadable(field) ? confidenceLabel(field.confidence) : "À vérifier sur le papier"}</small>
              </li>)}
            </ul>
          </details>

          <div className="step-heading"><span>?</span><div><h2>Poser une question</h2><p>Touchez une question ou écrivez la vôtre, en français, en arabe ou en darija.</p></div></div>
          <div className="chat-card">
            {(messages.length > 0 || pending) && <section className="chat-thread" aria-label="Conversation avec QraLia">
              {messages.map(({ question: userQuestion, result }) => <div className="chat-exchange" key={result.user_message_id}>
                <div className="chat-message user-message"><p>{userQuestion}</p></div>
                <div className="chat-message assistant-message">
                  {result.clarification_question
                    ? <p>{result.clarification_question}</p>
                    : result.answer && <>
                      <p>{withoutCitationIds(result.answer.answer)}</p>
                      <div className="answer-meta"><span>{routeLabels[result.answer.route]}</span><span>{confidenceLabel(result.answer.confidence)}</span></div>
                      {result.answer.citations.length > 0 && <div className="sources">{result.answer.citations.map((citation) => <a href={citation.url} target="_blank" rel="noreferrer" key={citation.chunk_id}>Source : {citation.title} ↗</a>)}</div>}
                    </>}
                  {(result.answer?.warnings.length || (result.resolution.action === "rewrite" && result.resolution.standalone_question)) ? <details className="tech-details">
                    <summary>Détails techniques</summary>
                    {result.resolution.action === "rewrite" && result.resolution.standalone_question && <small>Question comprise comme : {result.resolution.standalone_question}</small>}
                    {result.answer?.warnings.map((warning) => <small key={warning}>{warning}</small>)}
                  </details> : null}
                </div>
              </div>)}
              {pending && <div className="chat-exchange">
                <div className="chat-message user-message"><p>{pending}</p></div>
                <div className="chat-message assistant-message thinking" role="status"><p>QraLia cherche dans la facture et les sources officielles</p></div>
              </div>}
              <div className="chat-end" ref={chatEnd} />
            </section>}

            <div className="suggestions" aria-label="Questions suggérées">
              {suggestions.map((suggestion) => <button type="button" key={suggestion} onClick={() => void ask(suggestion)} disabled={busy !== null}>{suggestion}</button>)}
            </div>

            <form className="question-form" onSubmit={handleQuestion}>
              <label className="visually-hidden" htmlFor="question">Votre question</label>
              <textarea id="question" value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={handleQuestionKey} placeholder="Écrivez votre question…" rows={2} maxLength={500} />
              <button className="primary" disabled={!question.trim() || busy !== null}>Envoyer</button>
            </form>
            <div className="language-row">
              <label htmlFor="language">Langue de réponse</label>
              <select id="language" value={language} onChange={(event) => setLanguage(event.target.value)}>
                <option value="auto">Automatique</option><option value="fr">Français</option>
                <option value="ar">العربية الفصحى</option><option value="darija_arab">الدارجة</option><option value="darija_latn">Darija latine</option>
              </select>
            </div>
          </div>

          <button className="delete" onClick={handleDelete} disabled={busy !== null}>{busy === "delete" ? "Suppression…" : "Supprimer cette facture et lire une autre"}</button>
        </>}
      </section>

      <footer><strong>QraLia</strong><p>Prototype pédagogique — aucune donnée réelle ne doit être téléversée.</p></footer>
    </main>
  );
}
