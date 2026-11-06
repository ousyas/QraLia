import { DragEvent, FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import Guide, { guideAlreadySeen } from "./components/Guide";
import Logo from "./components/Logo";
import Countdown from "./components/Countdown";
import ReadingProgress from "./components/ReadingProgress";
import ScanDemo, { EXAMPLE_SCAN, type ScanView } from "./components/ScanDemo";
import ShowOnInvoice, { type Spot } from "./components/ShowOnInvoice";
import SpeakButton from "./components/SpeakButton";
import { MAX_UPLOAD_BYTES, MAX_UPLOAD_MB, prepareUpload } from "./photo";
import { checkHealth, createConversation, deleteInvoice, sendConversationMessage, uploadInvoice } from "./api";
import { confidenceLabel, fieldLabel, formatDate, formatField, isReadable, withoutCitationIds } from "./format";
import { DEFAULT_ANSWER_LANGUAGE, isRtl, saveUiLang, savedUiLang, spokenAmount, STRINGS, UI_LANGS, UiLang, type Strings } from "./i18n";
import type { ConversationMessageResponse, FieldView, InvoiceResponse } from "./types";

const TELECOM_PROVIDERS = new Set(["iam", "orange_ma", "inwi"]);
const ACCEPTED = /\.(pdf|png|jpe?g)$/i;
// Icônes au trait (même épaisseur que la typographie) plutôt que des emojis,
// qui changent d'aspect d'un téléphone à l'autre.
const icon = (path: string) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d={path} />
  </svg>
);
const HOW_ICONS = [
  icon("M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8zM14 3v5h5M12 17v-6M9.5 13.5 12 11l2.5 2.5"),
  icon("M10.5 4a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13zM20 20l-4.8-4.8M8 10.5h5M8 13h3"),
  icon("M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12zM8.5 12h.01M12 12h.01M15.5 12h.01"),
];
const PROVIDER_SHORT: Record<string, string> = { onee: "ONEE", iam: "Maroc Telecom", orange_ma: "Orange", inwi: "inwi" };

const sure = (field: FieldView | undefined): field is FieldView => isReadable(field) && field.confidence >= 0.7;
const shortDate = (iso: string, s: Strings) => formatDate(iso, s).replace(/\s*\d{4}$/, "");

/** Facture affichée dans l'en-tête : l'exemple, ou les valeurs lues sur la facture de l'utilisateur. */
function scanView(invoice: InvoiceResponse | null, fields: Map<string, FieldView>, s: Strings): ScanView {
  if (!invoice) return { ...EXAMPLE_SCAN, dateChip: shortDate("2026-08-27", s) };
  // Comme dans le résumé, une valeur incertaine n'est jamais montrée.
  const raw = (name: string) => {
    const field = fields.get(name);
    return sure(field) ? field.display_value || String(field.value) : "?";
  };
  const total = fields.get("total_check");
  const due = fields.get("due_date");
  const telecom = TELECOM_PROVIDERS.has(invoice.provider_id);
  return {
    kind: telecom ? "Facture d'abonnement" : "Facture d'électricité",
    provider: PROVIDER_SHORT[invoice.provider_id] ?? "?",
    icon: telecom ? "📶" : "⚡",
    rows: [["Total HT", raw("amount_before_tax")], [telecom ? "TVA" : "Taxes", raw("taxes")]],
    total: raw("total_check"),
    due: raw("due_date"),
    amountChip: sure(total) ? formatField(total, s) : s.summary.amountMissing,
    dateChip: sure(due) && typeof due.value === "string" ? shortDate(due.value, s) : s.summary.dateMissing,
    amountSure: sure(total),
    dateSure: sure(due),
  };
}

function SummaryValue({ field, label, missing, s }: { field: FieldView | undefined; label: string; missing: string; s: Strings }) {
  const readable = isReadable(field) && field.confidence >= 0.7;
  return <div className={`summary-value ${readable ? "" : "uncertain"}`}>
    <p>{label}</p>
    <strong>{readable ? formatField(field, s) : missing}</strong>
    <span>{readable ? confidenceLabel(field.confidence, s) : s.summary.askRelative}</span>
  </div>;
}

/** Phrase lue à voix haute : seules les valeurs sûres (≥ 70 %) sont prononcées. */
function spokenSummary(fields: Map<string, FieldView>, lang: UiLang, s: Strings): string {
  const total = fields.get("total_check");
  const due = fields.get("due_date");
  const amount = isReadable(total) && total.confidence >= 0.7 && typeof total.value === "number" ? spokenAmount(total.value, lang) : null;
  const date = isReadable(due) && due.confidence >= 0.7 && typeof due.value === "string" ? formatDate(due.value, s) : null;
  return s.summary.spoken(amount, date);
}

export default function App() {
  const [uiLang, setUiLang] = useState<UiLang>(savedUiLang);
  const s = STRINGS[uiLang];
  const [online, setOnline] = useState<boolean | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [confirmed, setConfirmed] = useState(false);
  const [invoice, setInvoice] = useState<InvoiceResponse | null>(null);
  const [question, setQuestion] = useState("");
  const [language, setLanguage] = useState(() => DEFAULT_ANSWER_LANGUAGE[savedUiLang()]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Array<{ question: string; result: ConversationMessageResponse }>>([]);
  const [pending, setPending] = useState<string | null>(null);
  const [busy, setBusy] = useState<"upload" | "question" | "delete" | null>(null);
  const [error, setError] = useState("");
  const [guideOpen, setGuideOpen] = useState(() => !guideAlreadySeen());
  const [dragging, setDragging] = useState(false);

  const chatEnd = useRef<HTMLDivElement>(null);
  const uploadSection = useRef<HTMLDivElement>(null);

  useEffect(() => { void checkHealth().then(setOnline); }, []);
  useEffect(() => {
    document.documentElement.lang = uiLang === "fr" ? "fr" : uiLang === "ar" ? "ar" : "ary";
    document.documentElement.dir = isRtl(uiLang) ? "rtl" : "ltr";
  }, [uiLang]);
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

  function changeUiLang(next: UiLang) {
    setUiLang(next); saveUiLang(next);
    setLanguage(DEFAULT_ANSWER_LANGUAGE[next]);
  }

  function pickFile(candidate: File | null | undefined) {
    if (!candidate) return;
    if (!ACCEPTED.test(candidate.name)) { setError(s.upload.badFormat); return; }
    // Une photo trop lourde sera compressée ; un PDF, lui, ne peut pas l'être.
    if (candidate.type === "application/pdf" && candidate.size > MAX_UPLOAD_BYTES) { setError(s.upload.tooBig(MAX_UPLOAD_MB)); return; }
    setError(""); setFile(candidate);
  }

  function handleDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault(); setDragging(false);
    pickFile(event.dataTransfer.files?.[0]);
  }

  function goToUpload() {
    uploadSection.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function handleUpload(event: FormEvent) {
    event.preventDefault();
    if (!file || !confirmed) return;
    setBusy("upload"); setError(""); setMessages([]); setConversationId(null);
    try {
      const prepared = await prepareUpload(file);
      if (prepared.size > MAX_UPLOAD_BYTES) { setError(s.upload.tooBig(MAX_UPLOAD_MB)); return; }
      const uploaded = await uploadInvoice(prepared);
      const conversation = await createConversation(uploaded.invoice_id, language);
      setInvoice(uploaded);
      setConversationId(conversation.conversation_id);
    }
    catch (cause) { setError(cause instanceof Error ? cause.message : s.errors.read); }
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
      setError(cause instanceof Error ? cause.message : s.errors.answer);
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
    } catch (cause) { setError(cause instanceof Error ? cause.message : s.errors.remove); }
    finally { setBusy(null); }
  }

  const total = fields.get("total_check");
  const due = fields.get("due_date");
  const preview = invoice?.previews?.find((item) => item.page === 1);
  // Seules les valeurs sûres, situées sur la page affichée, sont montrées sur la facture.
  const spots: Spot[] = [
    [total, s.summary.toPay, "amount"] as const,
    [due, s.summary.before, "date"] as const,
  ].flatMap(([field, label, tone]) => sure(field) && field.bbox && field.page === 1
    ? [{ key: field.name, label, tone, value: formatField(field, s), bbox: field.bbox }]
    : []);
  const cash = fields.get("total_cash");
  const plan = fields.get("plan_name");
  const suggestions = s.chat.suggestions[invoice && TELECOM_PROVIDERS.has(invoice.provider_id) ? "telecom" : "utilities"];
  const professional = invoice?.validation.warnings.includes("outside_mvp_professional_tariff");
  const unknownProvider = invoice?.provider_id === "unknown";

  return (
    <main className={invoice ? "has-invoice" : undefined}>
      <header className="topbar">
        <nav aria-label="QraLia">
          <Logo />
          <div className="nav-actions">
            <div className="lang-switch" role="group" aria-label="Langue / اللغة">
              {UI_LANGS.map((item) => <button key={item.id} className={item.id === uiLang ? "on" : ""} aria-pressed={item.id === uiLang} aria-label={item.label} onClick={() => changeUiLang(item.id)}>
                <span className="lang-long">{item.label}</span><span className="lang-short" aria-hidden="true">{item.short}</span>
              </button>)}
            </div>
            <button className="guide-button" onClick={() => setGuideOpen(true)} aria-label={s.guideButton}>
              <span className="guide-icon" aria-hidden="true">?</span><span className="guide-label">{s.guideButton}</span>
            </button>
            <span className={`status ${online ? "online" : "offline"}`} title={online === null ? s.status.checking : online ? s.status.online : s.status.offline}>
              {online === null ? s.status.checking : online ? s.status.online : s.status.offline}
            </span>
          </div>
        </nav>
      </header>

      <section className="hero">
        <div className="hero-grid">
          <div className="hero-copy" id="top">
            <p className="hero-pill"><span aria-hidden="true" />{s.hero.pill}</p>
            <h1>{s.hero.titleA} <em>{s.hero.titleB}</em></h1>
            <p>{s.hero.lead}</p>
            {!invoice && <div className="hero-actions">
              <button className="cta" onClick={goToUpload}>{s.hero.ctaRead}</button>
              <button className="cta-ghost" onClick={() => setGuideOpen(true)}>{s.hero.ctaHow}</button>
            </div>}
          </div>
          <div className="hero-demo">
            <ScanDemo view={scanView(invoice, fields, s)} toPay={s.summary.toPay} before={s.summary.before} caption={invoice ? s.demoYours : s.demoCaption} />
          </div>
        </div>
      </section>

      <section className="workspace" aria-live="polite">
        {/* Grand écran : dépôt à gauche, explications à droite. */}
        {!invoice && <div className="start-grid">
          <div className="start-main">
          <div className="step-heading" ref={uploadSection}><span>01</span><div><h2>{s.upload.title}</h2><p>{s.upload.hint}</p></div></div>
          <form className="upload-card" onSubmit={handleUpload}>
            {busy === "upload" && file ? <ReadingProgress fileName={file.name} s={s} /> : <>
              <label
                className={`dropzone ${dragging ? "dragging" : ""} ${file ? "has-file" : ""}`}
                onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={handleDrop}
              >
                <input type="file" accept=".pdf,.png,.jpg,.jpeg" onChange={(event) => pickFile(event.target.files?.[0])} />
                <span className="upload-icon" aria-hidden="true">{file ? "✓" : "↑"}</span>
                <strong>{file ? file.name : dragging ? s.upload.dropHere : s.upload.choose}</strong>
                <small>{file ? `${(file.size / 1024 / 1024).toFixed(2)} Mo — ${s.upload.change}` : s.upload.dropHint}</small>
              </label>
              <label className="consent"><input type="checkbox" checked={confirmed} onChange={(event) => setConfirmed(event.target.checked)} /> {s.upload.consent}</label>
              <button className="primary" disabled={!file || !confirmed || busy !== null}>{s.upload.read}</button>
            </>}
          </form>
          </div>

          <section className="how" aria-labelledby="how-title">
            <h2 id="how-title">{s.how.title}</h2>
            <ol>
              {s.how.items.map((item, index) => <li key={item.title}>
                <span className="how-icon" aria-hidden="true">{HOW_ICONS[index]}</span>
                <span className="how-step">{String(index + 1).padStart(2, "0")}</span>
                <h3>{item.title}</h3>
                <p>{item.text}</p>
              </li>)}
            </ol>
          </section>
        </div>}

        {error && <div className="alert error" role="alert">{error}</div>}

        {/* Grand écran : résumé et facture à gauche, conversation à droite. */}
        {invoice && <>
          <div className="result-grid">
          <div className="result-main">
          <section className="summary-card" aria-label={s.summary.toPay}>
            <div className="summary-head">
              <div>
                <p className="provider">{invoice.provider_name ?? s.summary.unknownProvider}</p>
                <p className="eyebrow">{invoice.provider_id === "unknown" ? s.summary.kindFallback : TELECOM_PROVIDERS.has(invoice.provider_id) ? s.kinds.telecom : s.kinds.electricity} {isReadable(fields.get("invoice_number")) ? `· ${fields.get("invoice_number")!.display_value}` : ""}</p>
              </div>
              <span className={`badge ${invoice.validation.valid && invoice.validation.scope_supported ? "ok" : "warn"}`}>
                {!invoice.validation.valid ? s.summary.badgeCheck : invoice.validation.scope_supported ? s.summary.badgeOk : professional ? s.summary.badgePro : s.summary.badgeOut}
              </span>
            </div>
            <div className="summary-values">
              <SummaryValue field={fields.get("total_check")} label={s.summary.toPay} missing={s.summary.amountMissing} s={s} />
              <SummaryValue field={fields.get("due_date")} label={s.summary.before} missing={s.summary.dateMissing} s={s} />
            </div>
            {sure(due) && typeof due.value === "string" && <Countdown
              dueIso={due.value} s={s}
              provider={PROVIDER_SHORT[invoice.provider_id] ?? ""}
              amount={sure(total) ? formatField(total, s) : null}
            />}
            <SpeakButton className="speak-summary" text={spokenSummary(fields, uiLang, s)} label={s.listenSummary} stopLabel={s.stop} />
            {isReadable(plan) && <p className="summary-note">{s.summary.plan} : <strong>{formatField(plan, s)}</strong></p>}
            {isReadable(cash) && <p className="summary-note">{s.summary.cash} : <strong>{formatField(cash, s)}</strong></p>}
            {professional && <p className="summary-note warn">{s.summary.proNote}</p>}
            {unknownProvider && <p className="summary-note warn">{s.summary.unknownNote}</p>}
          </section>

          {preview && <ShowOnInvoice preview={preview} spots={spots} title={s.show.title} hint={s.show.hint} fullPage={s.show.fullPage} />}

          <details className="details-card">
            <summary><span>{s.summary.details}</span><small>{s.summary.globalConfidence(Math.round(invoice.document_confidence * 100))}</small></summary>
            <ul className="field-list">
              {detailFields.map((field) => <li className={field.confidence >= 0.85 ? "" : "low"} key={field.name}>
                <span>{fieldLabel(field, s)}</span>
                <strong>{formatField(field, s)}</strong>
                <small>{isReadable(field) ? confidenceLabel(field.confidence, s) : s.summary.checkPaper}</small>
              </li>)}
            </ul>
          </details>

          </div>
          <div className="result-chat">
          <div className="step-heading"><span>?</span><div><h2>{s.chat.title}</h2><p>{s.chat.hint}</p></div></div>
          <div className="chat-card">
            {(messages.length > 0 || pending) && <section className="chat-thread" aria-label={s.chat.title}>
              {messages.map(({ question: userQuestion, result }) => {
                const reply = result.clarification_question ?? (result.answer ? withoutCitationIds(result.answer.answer) : "");
                return <div className="chat-exchange" key={result.user_message_id}>
                  <div className="chat-message user-message"><p>{userQuestion}</p></div>
                  <div className="chat-message assistant-message">
                    <p>{reply}</p>
                    {!result.clarification_question && result.answer && <>
                      <div className="answer-meta"><span>{s.chat.routes[result.answer.route]}</span><span>{confidenceLabel(result.answer.confidence, s)}</span></div>
                      {result.answer.citations.length > 0 && <div className="sources">{result.answer.citations.map((citation) => <a href={citation.url} target="_blank" rel="noreferrer" key={citation.chunk_id}>{s.chat.source}{citation.title} ↗</a>)}</div>}
                    </>}
                    <SpeakButton text={reply} label={s.listen} stopLabel={s.stop} />
                    {(result.answer?.warnings.length || (result.resolution.action === "rewrite" && result.resolution.standalone_question)) ? <details className="tech-details">
                      <summary>{s.chat.tech}</summary>
                      {result.resolution.action === "rewrite" && result.resolution.standalone_question && <small>{s.chat.understood}{result.resolution.standalone_question}</small>}
                      {result.answer?.warnings.map((warning) => <small key={warning}>{warning}</small>)}
                    </details> : null}
                  </div>
                </div>;
              })}
              {pending && <div className="chat-exchange">
                <div className="chat-message user-message"><p>{pending}</p></div>
                <div className="chat-message assistant-message thinking" role="status"><p>{s.chat.thinking}</p></div>
              </div>}
              <div className="chat-end" ref={chatEnd} />
            </section>}

            <div className="suggestions" aria-label={s.chat.title}>
              {suggestions.map((suggestion) => <button type="button" key={suggestion} onClick={() => void ask(suggestion)} disabled={busy !== null}>{suggestion}</button>)}
            </div>

            <form className="question-form" onSubmit={handleQuestion}>
              <label className="visually-hidden" htmlFor="question">{s.chat.yourQuestion}</label>
              <textarea id="question" value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={handleQuestionKey} placeholder={s.chat.placeholder} rows={2} maxLength={500}
                // Vide, le champ suit la langue de l'interface ; rempli, celle du texte tapé.
                dir={question ? "auto" : isRtl(uiLang) ? "rtl" : "ltr"} />
              <button className="primary" disabled={!question.trim() || busy !== null}>{s.chat.send}</button>
            </form>
            <div className="language-row">
              <label htmlFor="language">{s.chat.answerLang}</label>
              <select id="language" value={language} onChange={(event) => setLanguage(event.target.value)}>
                <option value="auto">{s.chat.auto}</option><option value="fr">Français</option>
                <option value="ar">العربية الفصحى</option><option value="darija_arab">الدارجة</option><option value="darija_latn">Darija latine</option>
              </select>
            </div>
          </div>
          </div>
          </div>

          <button className="delete" onClick={handleDelete} disabled={busy !== null}>{busy === "delete" ? s.removing : s.remove}</button>
        </>}
      </section>

      {guideOpen && <Guide s={s} onClose={() => setGuideOpen(false)} />}
      <footer><strong>QraLia</strong><p>{s.footer}</p></footer>
    </main>
  );
}
