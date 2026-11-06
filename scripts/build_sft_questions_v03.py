"""Construit la banque de questions v0.3 avant toute cible SFT.

Le split est porté par l'intention : un intent, ses traductions et son chunk
restent dans un seul split. Aucun exemple n'est présenté comme ASR sans audio.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from qralia.rag.bm25_evaluation import load_active_chunks


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "evaluation" / "sft_questions.v0.3.jsonl"
MANIFEST = ROOT / "data" / "evaluation" / "sft_questions.v0.3.manifest.json"
SCHEMA = ROOT / "schemas" / "sft_question_v0.3.schema.json"
CHUNK_SCHEMA = ROOT / "schemas" / "retrieval_chunk.schema.json"
CHUNK_PATHS = [
    ROOT / "data/processed/onee-regler-facture-residentiel/chunks.jsonl",
    ROOT / "data/processed/redal/chunks.jsonl",
    ROOT / "data/processed/telecom/chunks.jsonl",
]
LANGUAGES = ("fr", "ar", "darija_latn", "darija_arab")


def q(fr: str, ar: str, latn: str, arab: str) -> dict[str, str]:
    return {"fr": fr, "ar": ar, "darija_latn": latn, "darija_arab": arab}


# Une seule formulation réellement distincte par langue à ce stade. Les
# paraphrases futures seront écrites ou dictées par des humains et rattachées
# au même intent_id, donc elles ne pourront jamais traverser les splits.
POSITIVE: list[dict[str, Any]] = [
    {"intent": "onee_date_holiday", "family": "billing_due_date_holiday", "provider": "onee", "section": "date-reglement", "split": "train", "questions": q("Que devient la date de paiement ONEE lorsqu'elle tombe un jour férié ?", "ماذا يحدث لموعد أداء فاتورة المكتب إذا صادف يوم عطلة؟", "date dyal khlass loffice jat f nhar 3otla chno ndir", "تاريخ خلاص فاتورة المكتب جا فنهار عطلة شنو ندير؟")},
    {"intent": "onee_cash", "family": "cash_payment", "provider": "onee", "section": "paiement-especes", "split": "train", "questions": q("Où puis-je payer ma facture ONEE en espèces ?", "أين يمكنني أداء فاتورة المكتب نقدا؟", "fin n9der nkhlles factora dyal loffice b cash", "فين نقدر نخلص فاتورة المكتب بالكاش؟")},
    {"intent": "onee_cheque", "family": "cheque_payment", "provider": "onee", "section": "paiement-cheque", "split": "train", "questions": q("À quel ordre établir mon chèque ONEE et où le remettre ?", "باسم من أحرر شيك المكتب وأين أقدمه؟", "3la smit chkon nkteb chik dyal loffice w fin n3tih", "على سميت شكون نكتب شيك المكتب وفين نعطيه؟")},
    {"intent": "onee_bank_transfer", "family": "bank_transfer_payment", "provider": "onee", "section": "paiement-virement", "split": "train", "questions": q("Comment mettre en place le paiement mensuel de ONEE par ma banque ?", "كيف أفعل الأداء الشهري لفاتورة المكتب عن طريق البنك؟", "bghit lbanka tkhlles lia factora dyal do kol chher kifach ndir", "بغيت البنك يخلص ليا فاتورة الضو كل شهر كيفاش ندير؟")},
    {"intent": "onee_advance", "family": "advance_payment", "provider": "onee", "section": "paiement-acompte", "split": "test", "questions": q("Puis-je déposer une avance pour mes prochaines factures ONEE ?", "هل يمكنني وضع تسبيق لفواتير المكتب المقبلة؟", "n9der nkhelli tesbi9 3la factorat loffice li jayine", "نقدر نخلي تسبيق على فواتير المكتب الجاية؟")},

    {"intent": "redal_online_card", "family": "online_card_payment", "provider": "srm_rsk", "section": "paiement-en-ligne-carte", "split": "train", "questions": q("Après un paiement Redal par carte en ligne, puis-je obtenir un reçu ?", "هل أحصل على وصل بعد أداء فاتورة Redal بالبطاقة عبر الإنترنت؟", "mn b3d ma nkhlles redal b carte online fin nl9a reçu", "من بعد ما نخلص Redal بالكارط فالإنترنت فين نلقى الوصل؟")},
    {"intent": "redal_ebanking", "family": "ebanking_payment", "provider": "srm_rsk", "section": "paiement-ebanking", "split": "validation", "questions": q("Puis-je régler Redal depuis le service en ligne de ma banque ?", "هل يمكنني أداء فاتورة Redal من الخدمة الإلكترونية لبنكي؟", "n9der nkhlles redal men site wla appli dyal lbanka", "نقدر نخلص Redal من الموقع ولا تطبيق البنك؟")},
    {"intent": "redal_mobile_wallet", "family": "mobile_wallet_payment", "provider": "srm_rsk", "section": "paiement-mobile-jiwar", "split": "train", "questions": q("Quels portefeuilles mobiles permettent de payer Redal depuis un téléphone ?", "ما المحافظ الهاتفية التي تسمح بأداء فاتورة Redal؟", "achmen appli f telephone n9der nkhlles biha redal", "بأشمن تطبيق فالتلفون نقدر نخلص Redal؟")},
    {"intent": "redal_direct_debit", "family": "automatic_direct_debit", "provider": "srm_rsk", "section": "paiement-prelevement", "split": "train", "questions": q("Comment éviter d'oublier Redal grâce au prélèvement automatique ?", "كيف أتجنب نسيان فاتورة Redal باستعمال الاقتطاع التلقائي؟", "kanensa nkhlles redal kayn chi tari9a tkhlles bo7dha", "كننسى نخلص Redal واش كاينة طريقة تخلص بوحدها؟")},
    {"intent": "redal_atm", "family": "atm_payment", "provider": "srm_rsk", "section": "paiement-gab", "split": "validation", "questions": q("De quoi ai-je besoin pour payer Redal à un guichet automatique ?", "ماذا أحتاج لأداء فاتورة Redal في الشباك البنكي الآلي؟", "bghit nkhlles redal f machine dyal lbanka ach khasni", "بغيت نخلص Redal فماكينة البنك شنو خاصني؟")},

    {"intent": "iam_monthly_billing", "family": "monthly_billing", "provider": "iam", "section": "facturation-mensuelle", "split": "train", "questions": q("La facture mobile Maroc Telecom est-elle émise chaque mois ?", "هل تصدر فاتورة اتصالات المغرب للهاتف كل شهر؟", "factora dyal maroc telecom katji kol chher", "فاتورة اتصالات المغرب كتجي كل شهر؟")},
    {"intent": "iam_invoice_not_received", "family": "invoice_due_without_receipt", "provider": "iam", "section": "exigibilite-non-reception", "split": "train", "questions": q("Dois-je payer IAM à l'échéance même si je n'ai pas reçu la facture ?", "هل يجب أداء فاتورة اتصالات المغرب في موعدها حتى إن لم أتوصل بها؟", "ma wslatnich factora dyal iam wach khasni nkhlles f date", "ما وصلاتنيش فاتورة IAM واش خاصني نخلص فالموعد؟")},
    {"intent": "iam_billing_error", "family": "confirmed_billing_error", "provider": "iam", "section": "erreur-facturation", "split": "train", "questions": q("Que fait IAM lorsqu'une surfacturation est confirmée ?", "ماذا تفعل اتصالات المغرب إذا تأكد خطأ في الفاتورة؟", "ila t2ekdo belli iam 7sbat lia montant ghalet ach kaydiro", "إلا تأكدو بلي IAM حسبات ليا مبلغ غالط شنو كيديرو؟")},
    {"intent": "iam_debit_dispute", "family": "bank_debit_dispute", "provider": "iam", "section": "prelevement-contestation", "split": "test", "questions": q("Quel délai ai-je pour contester par écrit un prélèvement IAM ?", "ما الأجل المتاح للاعتراض كتابة على اقتطاع اتصالات المغرب؟", "iam 9t3at lia flous mn compte ch7al 3ndi bach n3tared ktebi", "IAM قطعات ليا الفلوس من الكونط شحال عندي باش نعترض بالكتابة؟")},
    {"intent": "iam_late_penalty", "family": "late_payment_penalty", "provider": "iam", "section": "penalites-retard", "split": "validation", "questions": q("Comment IAM calcule-t-elle les pénalités mensuelles de retard ?", "كيف تحتسب اتصالات المغرب غرامات التأخر الشهرية؟", "kifach iam kat7seb gharama chheriya ila t3ettelt f khlass", "كيفاش IAM كتحسب الغرامة الشهرية إلا تعطلت فالخلاص؟")},

    {"intent": "orange_payment_delay", "family": "minimum_payment_period", "provider": "orange_ma", "section": "delai-paiement", "split": "test", "questions": q("Quel délai minimum Orange accorde-t-elle pour payer une facture ?", "ما الحد الأدنى للمهلة التي تمنحها Orange لأداء الفاتورة؟", "ch7al mn nhar 3la l2a9al 3ndi bach nkhlles factora orange", "شحال من نهار على الأقل عندي باش نخلص فاتورة Orange؟")},
    {"intent": "orange_maxit_card", "family": "app_card_payment", "provider": "orange_ma", "section": "paiement-max-it-carte", "split": "train", "questions": q("Puis-je payer Orange par carte bancaire depuis Max it ?", "هل يمكن أداء فاتورة Orange بالبطاقة من تطبيق Max it؟", "n9der nkhlles orange b carte men max it", "نقدر نخلص Orange بالكارط من Max it؟")},
    {"intent": "orange_money", "family": "provider_mobile_wallet", "provider": "orange_ma", "section": "paiement-orange-money", "split": "train", "questions": q("Orange Money permet-il de régler une facture Orange ?", "هل تسمح Orange Money بأداء فاتورة Orange؟", "orange money katkhlles factora orange", "واش Orange Money كتخلص فاتورة Orange؟")},
    {"intent": "orange_recharge", "family": "recharge_payment", "provider": "orange_ma", "section": "paiement-recharge", "split": "train", "questions": q("Comment payer ma facture Orange par recharge ?", "كيف أؤدي فاتورة Orange بواسطة التعبئة؟", "bghit nkhlles factora orange b recharge ach ndir", "بغيت نخلص فاتورة Orange بالريشارج شنو ندير؟")},
    {"intent": "orange_invoice_download", "family": "invoice_download", "provider": "orange_ma", "section": "consulter-facture", "split": "train", "questions": q("Où consulter et télécharger mes factures Orange ?", "أين أطلع على فواتير Orange وأحملها؟", "fin nchouf w ntelecharger factorat orange", "فين نشوف ونحمّل فواتير Orange؟")},

    {"intent": "inwi_myinwi_reward", "family": "payment_app_reward", "provider": "inwi", "section": "paiement-my-inwi", "split": "validation", "questions": q("Quel avantage est annoncé lorsque je paie dans My inwi ?", "ما الامتياز المعلن عند أداء الفاتورة في My inwi؟", "ila khllest f my inwi ach kay3tiwni", "إلا خلصت فـ My inwi شنو كيعطيوني؟")},
    {"intent": "inwi_agency_hours", "family": "agency_opening_hours", "provider": "inwi", "section": "paiement-agence", "split": "train", "questions": q("Les agences inwi sont-elles ouvertes le samedi à 19 h ?", "هل تكون وكالات inwi مفتوحة السبت عند 19h؟", "n9der nemchi l agence inwi sebt m3a 19h", "نقدر نمشي لوكالة inwi السبت مع 19h؟")},
    {"intent": "inwi_payment_agents", "family": "partner_agent_payment", "provider": "inwi", "section": "paiement-agents", "split": "train", "questions": q("Chez quels réseaux partenaires puis-je payer une facture inwi ?", "لدى أي شبكات شريكة يمكنني أداء فاتورة inwi؟", "fin n9der nkhlles inwi 3nd chi partenaire", "فين نقدر نخلص inwi عند شي شريك؟")},
    {"intent": "inwi_money_steps", "family": "wallet_payment_steps", "provider": "inwi", "section": "paiement-inwi-money", "split": "test", "questions": q("Quelles étapes suivre pour payer avec inwi money ?", "ما الخطوات اللازمة لأداء الفاتورة بواسطة inwi money؟", "kifach nkhlles factora b inwi money khotwa b khotwa", "كيفاش نخلص الفاتورة بـ inwi money خطوة بخطوة؟")},

    {"intent": "anrt_first_complaint", "family": "first_level_complaint", "provider": "anrt", "section": "reclamation-operateur", "split": "train", "questions": q("Ma facture téléphonique paraît fausse : à qui réclamer d'abord ?", "تبدو فاتورة هاتفي خاطئة، فلمن أقدم الشكوى أولا؟", "factora dyal telephone bayna ghalta lmen nchki lwl", "فاتورة التلفون باينة غالطة لمن نشكي اللول؟")},
]


NEGATIVE: list[dict[str, Any]] = [
    {"intent": "onee_card_missing", "family": "unsupported_payment_method", "provider": "onee", "context_section": "paiement-cheque", "split": "train", "answerable_corpus": False, "questions": q("Puis-je payer cette facture ONEE en ligne par carte Visa ?", "هل يمكن أداء فاتورة المكتب عبر الإنترنت ببطاقة Visa؟", "n9der nkhlles factora dyal loffice online b visa", "نقدر نخلص فاتورة المكتب فالإنترنت بفيزا؟")},
    {"intent": "redal_cheque_missing", "family": "unsupported_payment_method", "provider": "srm_rsk", "context_section": "paiement-mobile-jiwar", "split": "train", "answerable_corpus": False, "questions": q("À quel ordre dois-je établir un chèque Redal ?", "باسم من أحرر شيك Redal؟", "3la smit chkon nkteb chik dyal redal", "على سميت شكون نكتب شيك Redal؟")},
    {"intent": "iam_agency_hours_missing", "family": "missing_opening_hours", "provider": "iam", "context_section": "facturation-mensuelle", "split": "train", "answerable_corpus": False, "questions": q("Les agences Maroc Telecom ouvrent-elles le dimanche ?", "هل تفتح وكالات اتصالات المغرب يوم الأحد؟", "wakalat maroc telecom kay7ello l7ed", "وكالات اتصالات المغرب كيحلو الحد؟")},
    {"intent": "orange_refund_missing", "family": "duplicate_payment_refund", "provider": "orange_ma", "context_section": "paiement-recharge", "split": "train", "answerable_corpus": False, "questions": q("J'ai payé Orange deux fois : quand serai-je remboursé ?", "أديت فاتورة Orange مرتين، فمتى أسترجع المبلغ؟", "khllest orange juj mrat imta yrj3o lia flous", "خلصت Orange جوج مرات إمتى يرجعو ليا الفلوس؟")},
    {"intent": "inwi_failed_payment_missing", "family": "failed_transaction_refund", "provider": "inwi", "context_section": "paiement-agence", "split": "validation", "answerable_corpus": False, "questions": q("Mon paiement inwi a échoué mais mon compte a été débité : quand récupérerai-je l'argent ?", "فشل أداء inwi لكن المبلغ اقتطع، فمتى أسترجعه؟", "paiement inwi fchel w flous t9t3o imta yrj3o", "أداء inwi فشل والفلوس تقطعو إمتى يرجعو؟")},
    {"intent": "redal_outage_missing", "family": "missing_service_level", "provider": "srm_rsk", "context_section": "facturation-mensuelle", "split": "test", "answerable_corpus": False, "questions": q("Après une panne Redal, dans quel délai le courant revient-il ?", "بعد عطب لدى Redal، متى يعود التيار؟", "3ndi panne redal ch7al kayakhdo bach yrj3 do", "عندي عطب Redal شحال كياخدو باش يرجع الضو؟")},
]


def provider_of(chunk: dict[str, Any]) -> str:
    return chunk["metadata"]["provider_id"]


def main() -> None:
    chunks = [chunk for path in CHUNK_PATHS for chunk in load_active_chunks(path, CHUNK_SCHEMA)]
    by_provider_section = {(provider_of(chunk), chunk["section_id"]): chunk for chunk in chunks}
    validator = Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8")))
    rows: list[dict[str, Any]] = []
    case_number = 1

    for spec in POSITIVE:
        chunk = by_provider_section[(spec["provider"], spec["section"])]
        for language in LANGUAGES:
            row = {"schema_version": "0.3.0", "case_id": f"sftq-v03-{case_number:04d}", "intent_id": spec["intent"], "semantic_family_id": spec["family"], "source_chunk_id": chunk["chunk_id"], "context_chunk_ids": [chunk["chunk_id"]], "language": language, "domain": chunk["metadata"]["domain"], "provider_id": spec["provider"], "channel": "typed", "register": "informal" if language.startswith("darija") else "standard", "question": spec["questions"][language], "split": spec["split"], "should_abstain": False, "answerable_in_given_context": True, "answerable_in_corpus": True, "review_status": "candidate_human_review"}
            validator.validate(row)
            rows.append(row)
            case_number += 1

    for spec in NEGATIVE:
        chunk = by_provider_section[(spec["provider"], spec["context_section"])]
        for language in LANGUAGES:
            row = {"schema_version": "0.3.0", "case_id": f"sftq-v03-{case_number:04d}", "intent_id": spec["intent"], "semantic_family_id": spec["family"], "source_chunk_id": None, "context_chunk_ids": [chunk["chunk_id"]], "language": language, "domain": chunk["metadata"]["domain"], "provider_id": spec["provider"], "channel": "typed", "register": "informal" if language.startswith("darija") else "standard", "question": spec["questions"][language], "split": spec["split"], "should_abstain": True, "answerable_in_given_context": False, "answerable_in_corpus": spec["answerable_corpus"], "review_status": "candidate_human_review"}
            validator.validate(row)
            rows.append(row)
            case_number += 1

    OUTPUT.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    intent_splits: dict[str, set[str]] = {}
    for row in rows:
        intent_splits.setdefault(row["intent_id"], set()).add(row["split"])
    if any(len(splits) != 1 for splits in intent_splits.values()):
        raise ValueError("Une intention traverse plusieurs splits")

    manifest = {"schema_version": "0.3.0", "status": "candidate_human_review", "counts": {"questions": len(rows), "intents": len(intent_splits), "positive_intents": len(POSITIVE), "negative_intents": len(NEGATIVE)}, "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())), "language_counts": dict(sorted(Counter(row["language"] for row in rows).items())), "abstention_count": sum(row["should_abstain"] for row in rows), "abstention_rate": sum(row["should_abstain"] for row in rows) / len(rows), "positive_chunk_coverage": len({row["source_chunk_id"] for row in rows if row["source_chunk_id"]}), "split_policy": "Group split by intent_id; all languages and future paraphrases remain with their intent.", "asr_policy": "No ASR-labelled case without real audio and an observed recognizer transcript.", "file": {"path": OUTPUT.name, "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}}
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
