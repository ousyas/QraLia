import type { ReactNode } from "react";

export type UiLang = "fr" | "ar" | "darija";

export const UI_LANGS: Array<{ id: UiLang; label: string; short: string }> = [
  { id: "fr", label: "FR", short: "FR" },
  { id: "darija", label: "الدارجة", short: "دارجة" },
  { id: "ar", label: "العربية", short: "عربي" },
];

// Langue de réponse proposée par défaut pour chaque langue d'interface.
export const DEFAULT_ANSWER_LANGUAGE: Record<UiLang, string> = { fr: "auto", ar: "ar", darija: "darija_arab" };

const STORAGE_KEY = "qralia.ui.lang";

export function savedUiLang(): UiLang {
  try {
    const value = window.localStorage.getItem(STORAGE_KEY);
    if (value === "fr" || value === "ar" || value === "darija") return value;
  } catch { /* stockage indisponible */ }
  return "fr";
}

export function saveUiLang(lang: UiLang) {
  try { window.localStorage.setItem(STORAGE_KEY, lang); } catch { /* ignoré */ }
}

export const isRtl = (lang: UiLang) => lang !== "fr";

type GuideSlide = { title: string; text: ReactNode };

export type Strings = {
  locale: string;
  status: { checking: string; online: string; offline: string };
  guideButton: string;
  hero: { titleA: string; titleB: string; lead: string; ctaRead: string; ctaHow: string };
  upload: { title: string; hint: string; choose: string; dropHere: string; dropHint: string; change: string; consent: string; read: string; badFormat: string; tooBig: (mb: number) => string };
  reading: { title: (file: string) => string; steps: string[] };
  how: { title: string; items: Array<{ title: string; text: string }> };
  summary: {
    unknownProvider: string; kindFallback: string; toPay: string; before: string;
    amountMissing: string; dateMissing: string; askRelative: string; plan: string; cash: string;
    badgeCheck: string; badgeOk: string; badgePro: string; badgeOut: string;
    proNote: string; unknownNote: string; details: string; globalConfidence: (n: number) => string; checkPaper: string;
    spoken: (amount: string | null, date: string | null) => string;
  };
  confidence: { high: string; check: string; uncertain: string; notReadable: string };
  chat: {
    title: string; hint: string; thinking: string; send: string; placeholder: string; yourQuestion: string;
    answerLang: string; auto: string; tech: string; understood: string; source: string;
    routes: Record<"invoice" | "official_rag" | "both" | "unknown", string>;
    suggestions: { utilities: string[]; telecom: string[] };
  };
  listen: string; stop: string; listenSummary: string;
  remove: string; removing: string; footer: string;
  errors: { read: string; answer: string; remove: string };
  guide: { skip: string; prev: string; next: string; start: string; stepOf: (i: number, n: number) => string; slides: GuideSlide[] };
  fields: Record<string, string>;
  tariffs: Record<string, string>;
  currency: string;
  demoCaption: string;
  show: { title: string; hint: string; fullPage: string };
  countdown: {
    left: (n: number) => string; tomorrow: string; today: string; late: (n: number) => string;
    remind: string; reminderTitle: (provider: string, amount: string | null) => string; reminderNote: string;
  };
  demoYours: string;
  kinds: { electricity: string; telecom: string };
};

const OK = ({ children }: { children: ReactNode }) => <span className="tag ok">{children}</span>;
const WARN = ({ children }: { children: ReactNode }) => <span className="tag warn">{children}</span>;

/** Isole un texte latin (nom de fichier) dans une phrase de droite à gauche. */
const isolate = (text: string) => `\u2068${text}\u2069`;

function splitAmount(amount: number) {
  const units = Math.floor(amount);
  return { units, cents: Math.round((amount - units) * 100) };
}

/** Montant écrit pour être bien prononcé par la synthèse vocale. */
export function spokenAmount(amount: number, lang: UiLang): string {
  const { units, cents } = splitAmount(amount);
  if (lang === "fr") return cents ? `${units} dirhams et ${cents} centimes` : `${units} dirhams`;
  if (lang === "ar") return cents ? `${units} درهما و${cents} سنتيما` : `${units} درهما`;
  return cents ? `${units} درهم و${cents} سنتيم` : `${units} درهم`;
}

const fr: Strings = {
  locale: "fr-FR",
  status: { checking: "Vérification…", online: "Service disponible", offline: "Service indisponible" },
  guideButton: "Guide",
  hero: {
    titleA: "Votre facture,", titleB: "lue pour vous.",
    lead: "QraLia lit votre facture d'électricité ou de téléphone, vous dit combien payer et avant quand, puis répond à vos questions en darija, en arabe ou en français.",
    ctaRead: "Lire ma facture", ctaHow: "Comment ça marche ?",
  },
  upload: {
    title: "Ajouter une facture", hint: "PDF, PNG ou JPEG — 25 Mo maximum", choose: "Choisir une facture",
    dropHere: "Déposez la facture ici", dropHint: "Touchez ici, prenez une photo ou glissez votre fichier",
    change: "touchez pour changer", consent: "Cette facture est entièrement synthétique et ne contient aucune donnée personnelle réelle.",
    read: "Lire la facture", badFormat: "Format non accepté : choisissez un PDF, PNG ou JPEG.",
    tooBig: (mb) => `Ce fichier est trop lourd (plus de ${mb} Mo). Pour un PDF, essayez d'exporter seulement la première page.`,
  },
  reading: {
    title: (file) => `QraLia lit « ${isolate(file)} »`,
    steps: ["Lecture du texte de la facture", "Reconnaissance du fournisseur", "Vérification des montants et des dates", "Préparation du résumé"],
  },
  how: {
    title: "Comment ça marche",
    items: [
      { title: "Ajoutez la facture", text: "Une photo bien droite ou un PDF : électricité ONEE, Maroc Telecom, Orange ou inwi." },
      { title: "QraLia la lit pour vous", text: "Montant, date limite, fournisseur : chaque chiffre est vérifié, jamais inventé." },
      { title: "Posez vos questions", text: "En français, en arabe ou en darija. Les réponses citent leurs sources officielles." },
    ],
  },
  summary: {
    unknownProvider: "Fournisseur non reconnu", kindFallback: "Facture", toPay: "À payer", before: "Avant le",
    amountMissing: "Montant non lisible", dateMissing: "Date non lisible",
    askRelative: "Demandez l'aide d'un proche pour vérifier sur le papier", plan: "Offre", cash: "En espèces, timbre inclus",
    badgeCheck: "Vérification nécessaire", badgeOk: "Contrôles cohérents", badgePro: "Facture professionnelle", badgeOut: "Hors périmètre",
    proNote: "Les champs peuvent être consultés, mais QraLia n'appliquera pas les règles résidentielles à cette facture professionnelle.",
    unknownNote: "QraLia n'a pas reconnu le fournisseur de cette facture : les champs lus sont affichés, mais aucune règle officielle ne sera appliquée.",
    details: "Voir tous les détails lus sur la facture", globalConfidence: (n) => `${n} % de confiance globale`, checkPaper: "À vérifier sur le papier",
    spoken: (amount, date) => [
      amount ? `Vous devez payer ${amount}.` : "Le montant n'a pas pu être lu avec certitude.",
      date ? `À payer avant le ${date}.` : "La date limite n'a pas pu être lue avec certitude.",
      !amount || !date ? "Demandez l'aide d'un proche pour vérifier sur le papier." : "",
    ].filter(Boolean).join(" "),
  },
  confidence: { high: "Confiance élevée", check: "À vérifier", uncertain: "Lecture incertaine", notReadable: "Non lisible" },
  chat: {
    title: "Poser une question", hint: "Touchez une question ou écrivez la vôtre, en français, en arabe ou en darija.",
    thinking: "QraLia cherche dans la facture et les sources officielles", send: "Envoyer", placeholder: "Écrivez votre question…",
    yourQuestion: "Votre question", answerLang: "Langue de réponse", auto: "Automatique", tech: "Détails techniques",
    understood: "Question comprise comme : ", source: "Source : ",
    routes: { invoice: "Facture", official_rag: "Source officielle", both: "Facture + source officielle", unknown: "Réponse non trouvée" },
    suggestions: {
      utilities: ["Combien dois-je payer ?", "Quelle est la date limite de paiement ?", "Où puis-je payer ma facture ?", "Puis-je payer par chèque ?"],
      telecom: ["Combien dois-je payer ?", "Quelle est la date limite de paiement ?", "Quelle est mon offre ?", "Comment payer ma facture ?"],
    },
  },
  listen: "Écouter", stop: "Arrêter", listenSummary: "Écouter le résumé",
  remove: "Supprimer cette facture et lire une autre", removing: "Suppression…",
  footer: "Prototype pédagogique — aucune donnée réelle ne doit être téléversée.",
  errors: { read: "Impossible de lire la facture.", answer: "Impossible de répondre.", remove: "Suppression impossible." },
  guide: {
    skip: "Passer", prev: "Précédent", next: "Suivant", start: "Commencer", stepOf: (i, n) => `Étape ${i} sur ${n}`,
    slides: [
      { title: "Bienvenue sur QraLia", text: <>QraLia lit votre facture à votre place et vous dit l'essentiel : <strong>combien payer</strong> et <strong>avant quelle date</strong>. Électricité ou abonnement téléphone.</> },
      { title: "1. Ajoutez votre facture", text: <>Prenez une <strong>photo bien droite</strong>, à la lumière, ou choisissez un PDF. Pour ce prototype, utilisez seulement une facture d'exemple, jamais une vraie.</> },
      { title: "2. Lisez ou écoutez le résumé", text: <>Le montant et la date s'affichent en grand, et le bouton <strong>🔊 Écouter</strong> les lit à voix haute. <OK>Confiance élevée</OK> : lecture sûre. <WARN>À vérifier</WARN> : demandez à un proche de regarder le papier.</> },
      { title: "3. Posez vos questions", text: <>Touchez une question proposée ou écrivez la vôtre, en <strong>français, arabe ou darija</strong>. QraLia montre ses sources et préfère dire « je ne sais pas » plutôt que deviner.</> },
      { title: "Vos informations restent à vous", text: <>Le fichier est supprimé juste après la lecture. Les informations lues sont effacées après <strong>30 minutes</strong>, ou tout de suite quand vous touchez « Supprimer ». Revoyez ce guide à tout moment avec le bouton <strong>Guide</strong>.</> },
    ],
  },
  fields: {},
  tariffs: { BT: "Basse tension", BT_DOMESTIQUE: "Basse tension — usage domestique", MT_GENERAL: "Moyenne tension — usage général (professionnel)" },
  currency: "DH",
  demoCaption: "Exemple fictif",
  show: { title: "Où c'est sur ma facture ?", hint: "QraLia a trouvé ces informations à cet endroit précis de votre facture.", fullPage: "Voir la page entière" },
  countdown: {
    left: (n) => `Il vous reste ${n} jours pour payer`, tomorrow: "À payer demain", today: "À payer aujourd'hui",
    late: (n) => `Date limite dépassée depuis ${n} jour${n > 1 ? "s" : ""}`, remind: "Me le rappeler",
    reminderTitle: (provider, amount) => `Payer la facture ${provider}${amount ? ` (${amount})` : ""}`,
    reminderNote: "Rappel créé par QraLia. Vérifiez le montant sur votre facture.",
  },
  demoYours: "Votre facture, lue par QraLia",
  kinds: { electricity: "Facture d'électricité", telecom: "Facture d'abonnement" },
};

const ar: Strings = {
  locale: "ar-MA",
  status: { checking: "جارٍ التحقق…", online: "الخدمة متاحة", offline: "الخدمة غير متاحة" },
  guideButton: "الدليل",
  hero: {
    titleA: "فاتورتك،", titleB: "نقرؤها من أجلك.",
    lead: "يقرأ QraLia فاتورة الكهرباء أو الهاتف، ويخبرك بالمبلغ الواجب أداؤه وآخر أجل للأداء، ثم يجيب عن أسئلتك بالدارجة أو العربية أو الفرنسية.",
    ctaRead: "اقرأ فاتورتي", ctaHow: "كيف يعمل؟",
  },
  upload: {
    title: "أضف فاتورة", hint: "PDF أو PNG أو JPEG — 25 ميغابايت كحد أقصى", choose: "اختر فاتورة",
    dropHere: "ضع الفاتورة هنا", dropHint: "انقر هنا، أو التقط صورة، أو اسحب ملفك",
    change: "انقر للتغيير", consent: "هذه الفاتورة نموذجية بالكامل ولا تحتوي على أي معطيات شخصية حقيقية.",
    read: "اقرأ الفاتورة", badFormat: "صيغة غير مقبولة: اختر PDF أو PNG أو JPEG.",
    tooBig: (mb) => `هذا الملف كبير جدا (أكثر من ${mb} ميغابايت). بالنسبة لملف PDF، جرّب تصدير الصفحة الأولى فقط.`,
  },
  reading: {
    title: (file) => `يقرأ QraLia «${isolate(file)}»`,
    steps: ["قراءة نص الفاتورة", "التعرف على الشركة", "التحقق من المبالغ والتواريخ", "إعداد الملخص"],
  },
  how: {
    title: "كيف يعمل",
    items: [
      { title: "أضف الفاتورة", text: "صورة مستقيمة أو ملف PDF: كهرباء ONEE، اتصالات المغرب، أورنج أو إنوي." },
      { title: "يقرؤها QraLia من أجلك", text: "المبلغ، آخر أجل، الشركة: يُتحقق من كل رقم ولا يُخترع أبدا." },
      { title: "اطرح أسئلتك", text: "بالفرنسية أو العربية أو الدارجة. تذكر الأجوبة مصادرها الرسمية." },
    ],
  },
  summary: {
    unknownProvider: "شركة غير معروفة", kindFallback: "فاتورة", toPay: "المبلغ الواجب أداؤه", before: "قبل",
    amountMissing: "المبلغ غير مقروء", dateMissing: "التاريخ غير مقروء",
    askRelative: "اطلب من أحد أقاربك التحقق من الورقة", plan: "العرض", cash: "نقدا، مع رسم التنبر",
    badgeCheck: "يلزم التحقق", badgeOk: "المراقبة سليمة", badgePro: "فاتورة مهنية", badgeOut: "خارج النطاق",
    proNote: "يمكنك الاطلاع على المعلومات، لكن QraLia لن يطبق قواعد الاستعمال المنزلي على هذه الفاتورة المهنية.",
    unknownNote: "لم يتعرف QraLia على الشركة المصدرة لهذه الفاتورة: تُعرض المعلومات المقروءة دون تطبيق أي قاعدة رسمية.",
    details: "عرض كل التفاصيل المقروءة", globalConfidence: (n) => `ثقة إجمالية ${n}٪`, checkPaper: "تحقق منها في الورقة",
    spoken: (amount, date) => [
      amount ? `يجب عليك أداء ${amount}.` : "تعذرت قراءة المبلغ بشكل مؤكد.",
      date ? `آخر أجل للأداء هو ${date}.` : "تعذرت قراءة آخر أجل بشكل مؤكد.",
      !amount || !date ? "اطلب من أحد أقاربك التحقق من الورقة." : "",
    ].filter(Boolean).join(" "),
  },
  confidence: { high: "ثقة عالية", check: "يلزم التحقق", uncertain: "قراءة غير مؤكدة", notReadable: "غير مقروء" },
  chat: {
    title: "اطرح سؤالا", hint: "انقر على سؤال أو اكتب سؤالك بالعربية أو الدارجة أو الفرنسية.",
    thinking: "يبحث QraLia في الفاتورة وفي المصادر الرسمية", send: "إرسال", placeholder: "اكتب سؤالك…",
    yourQuestion: "سؤالك", answerLang: "لغة الجواب", auto: "تلقائي", tech: "تفاصيل تقنية",
    understood: "فُهم السؤال كالتالي: ", source: "المصدر: ",
    routes: { invoice: "الفاتورة", official_rag: "مصدر رسمي", both: "الفاتورة + مصدر رسمي", unknown: "لم يُعثر على جواب" },
    suggestions: {
      utilities: ["ما المبلغ الواجب أداؤه؟", "ما هو آخر أجل للأداء؟", "أين يمكنني أداء الفاتورة؟", "هل يمكنني الأداء بالشيك؟"],
      telecom: ["ما المبلغ الواجب أداؤه؟", "ما هو آخر أجل للأداء؟", "ما هو العرض الذي اشتركت فيه؟", "كيف يمكنني أداء الفاتورة؟"],
    },
  },
  listen: "استمع", stop: "إيقاف", listenSummary: "استمع إلى الملخص",
  remove: "احذف هذه الفاتورة واقرأ أخرى", removing: "جارٍ الحذف…",
  footer: "نموذج تعليمي — لا يجب رفع أي وثيقة حقيقية.",
  errors: { read: "تعذرت قراءة الفاتورة.", answer: "تعذر الجواب.", remove: "تعذر الحذف." },
  guide: {
    skip: "تخطي", prev: "السابق", next: "التالي", start: "ابدأ", stepOf: (i, n) => `الخطوة ${i} من ${n}`,
    slides: [
      { title: "مرحبا بك في QraLia", text: <>يقرأ QraLia فاتورتك نيابة عنك ويخبرك بالأهم: <strong>كم تؤدي</strong> و<strong>قبل أي تاريخ</strong>. الكهرباء أو اشتراك الهاتف.</> },
      { title: "1. أضف فاتورتك", text: <>التقط <strong>صورة مستقيمة</strong> في مكان مضيء، أو اختر ملف PDF. في هذا النموذج، استعمل فقط فاتورة مثال، وليس فاتورة حقيقية.</> },
      { title: "2. اقرأ الملخص أو استمع إليه", text: <>يظهر المبلغ والتاريخ بخط كبير، والزر <strong>🔊 استمع</strong> يقرؤهما بصوت مسموع. <OK>ثقة عالية</OK>: قراءة مؤكدة. <WARN>يلزم التحقق</WARN>: اطلب من أحد أقاربك النظر في الورقة.</> },
      { title: "3. اطرح أسئلتك", text: <>انقر على سؤال مقترح أو اكتب سؤالك <strong>بالعربية أو الدارجة أو الفرنسية</strong>. يعرض QraLia مصادره ويفضل أن يقول «لا أعرف» على أن يخمن.</> },
      { title: "معلوماتك ملك لك", text: <>يُحذف الملف مباشرة بعد القراءة. وتُمحى المعلومات المقروءة بعد <strong>30 دقيقة</strong>، أو فورا عند النقر على «احذف». يمكنك الرجوع إلى هذا الدليل في أي وقت عبر زر <strong>الدليل</strong>.</> },
    ],
  },
  fields: {
    invoice_number: "رقم الفاتورة", contract_number: "رقم العقد", total_cash: "المبلغ نقدا", total_check: "المبلغ الإجمالي الواجب أداؤه",
    due_date: "آخر أجل للأداء", invoice_date: "تاريخ الفاتورة", tariff: "التعريفة", amount_before_tax: "المبلغ دون احتساب الرسوم",
    taxes: "الرسوم", previous_unpaid: "المتأخرات", plan_name: "العرض", line_number: "رقم الخط", out_of_plan_amount: "الاستهلاك خارج الباقة",
  },
  tariffs: { BT: "الجهد المنخفض", BT_DOMESTIQUE: "الجهد المنخفض — استعمال منزلي", MT_GENERAL: "الجهد المتوسط — استعمال عام (مهني)" },
  currency: "درهم",
  demoCaption: "مثال وهمي",
  show: { title: "أين توجد في فاتورتي؟", hint: "وجد QraLia هذه المعلومات في هذا المكان بالضبط من فاتورتك.", fullPage: "عرض الصفحة كاملة" },
  countdown: {
    left: (n) => `بقي لك ${n} يوما للأداء`, tomorrow: "آخر أجل غدا", today: "آخر أجل اليوم",
    late: (n) => `تجاوزت آخر أجل منذ ${n} يوما`, remind: "ذكّرني",
    reminderTitle: (provider, amount) => `أداء فاتورة ${provider}${amount ? ` (${amount})` : ""}`,
    reminderNote: "تذكير من QraLia. تحقق من المبلغ في فاتورتك.",
  },
  demoYours: "فاتورتك كما قرأها QraLia",
  kinds: { electricity: "فاتورة الكهرباء", telecom: "فاتورة الاشتراك" },
};

const darija: Strings = {
  locale: "ar-MA",
  status: { checking: "كنتأكدو…", online: "الخدمة خدامة", offline: "الخدمة ما خدامةش" },
  guideButton: "الدليل",
  hero: {
    titleA: "الفاتورة ديالك،", titleB: "كنقراوها ليك.",
    lead: "QraLia كتقرا ليك فاتورة الضو ولا التيليفون، كتقول ليك شحال خاصك تخلص وفوقاش، ومن بعد كتجاوبك على الأسئلة ديالك بالدارجة، بالعربية ولا بالفرنسية.",
    ctaRead: "قرا ليا الفاتورة", ctaHow: "كيفاش خدامة؟",
  },
  upload: {
    title: "زيد الفاتورة", hint: "PDF، PNG ولا JPEG — حتى لـ 25 ميغا", choose: "ختار الفاتورة",
    dropHere: "حط الفاتورة هنا", dropHint: "كليكي هنا، صوّر ولا جر الملف ديالك",
    change: "كليكي باش تبدلها", consent: "هاد الفاتورة مثال مصنوع، ما فيها حتى معلومة شخصية حقيقية.",
    read: "قرا الفاتورة", badFormat: "هاد النوع ما مقبولش: ختار PDF، PNG ولا JPEG.",
    tooBig: (mb) => `هاد الملف تقيل بزاف (كثر من ${mb} ميغا). إلا كان PDF، جرب تصيفط غير الصفحة الأولى.`,
  },
  reading: {
    title: (file) => `QraLia كتقرا «${isolate(file)}»`,
    steps: ["كنقراو النص ديال الفاتورة", "كنعرفو الشركة", "كنتأكدو من المبالغ والتواريخ", "كنوجدو الملخص"],
  },
  how: {
    title: "كيفاش خدامة",
    items: [
      { title: "زيد الفاتورة", text: "صورة مقادة ولا PDF: الضو ديال ONEE، اتصالات المغرب، أورونج ولا إنوي." },
      { title: "QraLia كتقراها ليك", text: "المبلغ، آخر أجل، الشركة: كل رقم كنتأكدو منو، عمرنا ما كنخترعوه." },
      { title: "سول الأسئلة ديالك", text: "بالدارجة، بالعربية ولا بالفرنسية. الأجوبة كتبين المصادر الرسمية ديالها." },
    ],
  },
  summary: {
    unknownProvider: "ما عرفناش الشركة", kindFallback: "فاتورة", toPay: "خاصك تخلص", before: "قبل",
    amountMissing: "المبلغ ما بانش مزيان", dateMissing: "التاريخ ما بانش مزيان",
    askRelative: "طلب من شي واحد من العائلة يشوف معاك الورقة", plan: "العرض", cash: "بالكاش، مع التمبر",
    badgeCheck: "خاص التأكد", badgeOk: "كلشي مزيان", badgePro: "فاتورة ديال شركة", badgeOut: "خارج المجال",
    proNote: "تقدر تشوف المعلومات، ولكن QraLia ما غاديش تطبق القواعد ديال الدار على فاتورة ديال شركة.",
    unknownNote: "QraLia ما عرفاتش الشركة ديال هاد الفاتورة: المعلومات اللي تقراو باينين، ولكن حتى قاعدة رسمية ما غادي تطبق.",
    details: "شوف كاع التفاصيل اللي تقراو", globalConfidence: (n) => `ثقة عامة ${n}٪`, checkPaper: "تأكد منها فالورقة",
    spoken: (amount, date) => [
      amount ? `خاصك تخلص ${amount}.` : "المبلغ ما تقراش مزيان.",
      date ? `قبل ${date}.` : "آخر أجل ما تقراش مزيان.",
      !amount || !date ? "طلب من شي واحد من العائلة يشوف معاك الورقة." : "",
    ].filter(Boolean).join(" "),
  },
  confidence: { high: "ثقة عالية", check: "خاص التأكد", uncertain: "القراية ماشي متأكدة", notReadable: "ما تقراتش" },
  chat: {
    title: "سول سؤال", hint: "كليكي على سؤال ولا كتب السؤال ديالك، بالدارجة، بالعربية ولا بالفرنسية.",
    thinking: "QraLia كتقلب فالفاتورة وفالمصادر الرسمية", send: "صيفط", placeholder: "كتب السؤال ديالك…",
    yourQuestion: "السؤال ديالك", answerLang: "لغة الجواب", auto: "أوتوماتيك", tech: "تفاصيل تقنية",
    understood: "فهمنا السؤال هكا: ", source: "المصدر: ",
    routes: { invoice: "الفاتورة", official_rag: "مصدر رسمي", both: "الفاتورة + مصدر رسمي", unknown: "ما لقيناش الجواب" },
    suggestions: {
      utilities: ["شحال خاصني نخلص؟", "شنو هو آخر أجل باش نخلص؟", "فين نقدر نخلص الفاتورة؟", "واش نقدر نخلص بالشيك؟"],
      telecom: ["شحال خاصني نخلص؟", "شنو هو آخر أجل باش نخلص؟", "شنو هو العرض ديالي؟", "كيفاش نخلص الفاتورة؟"],
    },
  },
  listen: "سمع", stop: "حبس", listenSummary: "سمع الملخص",
  remove: "مسح هاد الفاتورة وقرا وحدة أخرى", removing: "كنمسحو…",
  footer: "نموذج تعليمي — ما تحطش حتى وثيقة حقيقية.",
  errors: { read: "ما قدرناش نقراو الفاتورة.", answer: "ما قدرناش نجاوبو.", remove: "ما قدرناش نمسحو." },
  guide: {
    skip: "دوز", prev: "اللور", next: "التالي", start: "يلاه نبداو", stepOf: (i, n) => `المرحلة ${i} من ${n}`,
    slides: [
      { title: "مرحبا بيك فـ QraLia", text: <>QraLia كتقرا ليك الفاتورة وكتقول ليك المهم: <strong>شحال تخلص</strong> و<strong>قبل من إمتى</strong>. الضو ولا الاشتراك ديال التيليفون.</> },
      { title: "1. زيد الفاتورة", text: <>صوّر الفاتورة <strong>مقادة</strong> وفالضو، ولا ختار PDF. فهاد النموذج، خدم غير بفاتورة مثال، عمرك ما تحط فاتورة حقيقية.</> },
      { title: "2. قرا ولا سمع الملخص", text: <>المبلغ والتاريخ كيبانو كبار، والزر <strong>🔊 سمع</strong> كيقراهم ليك بالصوت. <OK>ثقة عالية</OK>: القراية مضمونة. <WARN>خاص التأكد</WARN>: طلب من شي واحد من العائلة يشوف الورقة.</> },
      { title: "3. سول الأسئلة ديالك", text: <>كليكي على سؤال ولا كتب السؤال ديالك <strong>بالدارجة، بالعربية ولا بالفرنسية</strong>. QraLia كتبين المصادر ديالها وكتفضل تقول «ما عرفتش» على أنها تخمن.</> },
      { title: "المعلومات ديالك كتبقى ليك", text: <>الملف كيتمسح غير كتسالي القراية. المعلومات اللي تقراو كيتمسحو من بعد <strong>30 دقيقة</strong>، ولا ديريكت ملي تكليكي على «مسح». تقدر ترجع لهاد الدليل فأي وقت بالزر <strong>الدليل</strong>.</> },
    ],
  },
  fields: {
    invoice_number: "نمرة الفاتورة", contract_number: "نمرة الكونطرا", total_cash: "المبلغ بالكاش", total_check: "المبلغ اللي خاصك تخلص",
    due_date: "آخر أجل باش تخلص", invoice_date: "تاريخ الفاتورة", tariff: "التعريفة", amount_before_tax: "المبلغ بلا ضريبة",
    taxes: "الضرائب", previous_unpaid: "الفلوس اللي باقين ما تخلصوش", plan_name: "العرض", line_number: "نمرة الخط", out_of_plan_amount: "الاستهلاك برا الفورفي",
  },
  tariffs: { BT: "الضغط المنخفض", BT_DOMESTIQUE: "الضغط المنخفض — ديال الدار", MT_GENERAL: "الضغط المتوسط — ديال الشركات" },
  currency: "درهم",
  demoCaption: "مثال ماشي حقيقي",
  show: { title: "فين كاينة فالفاتورة ديالي؟", hint: "QraLia لقات هاد المعلومات هنا بالضبط فالفاتورة ديالك.", fullPage: "شوف الصفحة كاملة" },
  countdown: {
    left: (n) => `باقي ليك ${n} يام باش تخلص`, tomorrow: "آخر أجل غدا", today: "آخر أجل اليوم",
    late: (n) => `فات آخر أجل هادي ${n} يام`, remind: "فكرني",
    reminderTitle: (provider, amount) => `خلّص الفاتورة ديال ${provider}${amount ? ` (${amount})` : ""}`,
    reminderNote: "تفكير من QraLia. تأكد من المبلغ فالفاتورة ديالك.",
  },
  demoYours: "الفاتورة ديالك كيفما قراتها QraLia",
  kinds: { electricity: "فاتورة الضو", telecom: "فاتورة الاشتراك" },
};

export const STRINGS: Record<UiLang, Strings> = { fr, ar, darija };
