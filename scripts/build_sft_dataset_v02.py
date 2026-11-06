"""Construit le dataset SFT QraLia v0.2 centré sur la robustesse linguistique.

Contrairement à v0.1, ce builder ne fabrique pas du volume avec des préfixes
polis. Chaque famille part de formulations sémantiquement différentes, puis
ajoute seulement deux variantes ASR/WhatsApp traçables. Le test est choisi par
type de formulation avant toute exécution du modèle et n'est jamais entraîné.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

import build_sft_dataset as v01
from qralia.rag.bm25_evaluation import load_active_chunks
from qralia.rag.generation_evaluation import build_generation_prompt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "training" / "qralia_sft_v0.2"
SCHEMA = ROOT / "schemas" / "sft_example_v0.2.schema.json"
CHUNK_SCHEMA = ROOT / "schemas" / "retrieval_chunk.schema.json"
LANGUAGES = ("darija_latn", "darija_arab", "fr", "ar")

# Deux formulations ajoutées à chaque famille positive : une situation vécue
# et une formulation avec alias ou sans nom explicite du fournisseur.
POSITIVE_EXTRAS: dict[str, dict[str, list[str]]] = {
    "onee_cheque": v01._q(
        ["3ndi factora dyal do w bghit nkhllesha b chik fin n3tih w 3la smit chkon", "bghit n3ti cheque l loffice kifach n3emro"],
        ["عندي فاكتورة ديال الضو وبغيت نخلصها بالشيك فين نعطيه وعلى سميت شكون", "بغيت نعطي شيك لوكالة الما والضو كيفاش نعمرو"],
        ["J'ai une facture d'électricité à payer par chèque : quel ordre indiquer et où le remettre ?", "Comment remplir puis remettre un chèque à l'office ?"],
        ["لدي فاتورة كهرباء أريد أداءها بشيك، باسم من أحرره وأين أقدمه؟", "كيف أملأ شيكا وأقدمه إلى مكتب الماء والكهرباء؟"],
    ),
    "onee_holiday": v01._q(
        ["l date li f factora jat nhar ma kaykhdmoch wach nmchi mn b3d", "loffice 3tani nhar lkhlass f 3otla chno ndir"],
        ["التاريخ لي فالفاكتورة جا فنهار ما كيخدموش واش نمشي من بعد", "المكتب عطاني نهار الخلاص فعطلة شنو ندير"],
        ["L'échéance affichée tombe un jour où les agences ferment : quand dois-je payer ?", "L'office a fixé le paiement un jour férié, que dois-je faire ?"],
        ["يصادف الأجل المكتوب في الفاتورة يوما لا تعمل فيه الوكالات، فمتى أؤدي؟", "حدد المكتب يوم الأداء خلال عطلة، ماذا أفعل؟"],
    ),
    "redal_orange_money": v01._q(
        ["ana f rabat wach n9der nkhlles lma w do b orange money daba", "bghit nkhlles had lfactora mn portable wach orange money kayn"],
        ["أنا فالرباط واش نقدر نخلص الما والضو بأورنج موني دابا", "بغيت نخلص هاد الفاكتورة من البورطابل واش أورنج موني كاين"],
        ["À Rabat, puis-je régler l'eau et l'électricité depuis Orange Money ?", "Je veux payer cette facture depuis mon téléphone : Orange Money est-il proposé ?"],
        ["هل يمكنني في الرباط أداء فاتورة الماء والكهرباء بواسطة Orange Money؟", "أريد أداء هذه الفاتورة من هاتفي، فهل تتوفر خدمة Orange Money؟"],
    ),
    "redal_monthly": v01._q(
        ["jatni wer9a fiha lma w do wach hadi hiya facture chheriya", "had charika katjme3 lma lkahraba w ttathir f wer9a wa7da"],
        ["جاتني ورقة فيها الما والضو واش هادي هي الفاتورة الشهرية", "هاد الشركة كتجمع الما والكهربا والتطهير فورقة وحدة"],
        ["J'ai reçu une seule facture pour l'eau et l'électricité : est-ce la facture mensuelle ?", "Cette société regroupe-t-elle eau, électricité et assainissement sur le même document ?"],
        ["توصلت بفاتورة واحدة للماء والكهرباء، فهل هي الفاتورة الشهرية؟", "هل تجمع هذه الشركة الماء والكهرباء والتطهير في وثيقة واحدة؟"],
    ),
    "iam_contestation": v01._q(
        ["ittisalat 9t3at lia flous mn compte w ma mwafe9ch ch7al ba9i lia bach n3tared", "maroc telecom dazet prélèvement ghalet fin nchki w imta"],
        ["اتصالات المغرب قطعات ليا الفلوس من الكونط وما موافقش شحال باقي ليا باش نعترض", "ماروك تيليكوم دارت اقتطاع غالط فين نشكي وإمتى"],
        ["Maroc Telecom a débité mon compte et je conteste : combien de temps me reste-t-il ?", "Un prélèvement d'Ittisalat Al-Maghrib me semble erroné : où et quand le contester ?"],
        ["اقتطعت اتصالات المغرب مبلغا من حسابي وأعترض عليه، فما الأجل المتاح؟", "يبدو اقتطاع اتصالات المغرب خاطئا، فأين ومتى أعترض؟"],
    ),
    "orange_recharge": v01._q(
        ["factora 180dh wach n3emmer 180 w n3ayet l chi numero", "ma3ndich carte bancaire n9der nsawb paiement b recharge"],
        ["الفاكتورة 180 درهم واش نعمر 180 ونعيط لشي نمرة", "ما عنديش كارت بنكية نقدر نصاوب الخلاص بالتعبئة"],
        ["Ma facture est de 180 DH : puis-je recharger 180 DH puis appeler un numéro ?", "Je n'ai pas de carte bancaire : puis-je payer cette facture par recharge ?"],
        ["قيمة فاتورتي 180 درهما، فهل أعبئ 180 درهما ثم أتصل برقم؟", "لا أملك بطاقة بنكية، فهل يمكنني أداء الفاتورة بالتعبئة؟"],
    ),
    "inwi_hours": v01._q(
        ["khdam 7ta l3chiya wach nl7e9 nkhlles factora m3a 19h", "bghit nmchi l wakala sebt m3a 10 wach nl9aha ma7loula"],
        ["خدام حتى للعشية واش نلحق نخلص الفاكتورة مع 19h", "بغيت نمشي للوكالة السبت مع 10 واش نلقاها محلولة"],
        ["Je termine tard : puis-je encore payer en agence à 19 h ?", "Si je vais à la boutique samedi à 10 h, sera-t-elle ouverte ?"],
        ["أنهي عملي متأخرا، فهل يمكنني الأداء في الوكالة عند 19h؟", "إذا ذهبت إلى الوكالة السبت عند 10h، فهل تكون مفتوحة؟"],
    ),
    "anrt_complaint": v01._q(
        ["factora dyal telephone ghalia bzaf fin nebda nchki", "ma tfahemtch m3a montant dyal telecom lmen nemchi lwl"],
        ["فاكتورة ديال التلفون غالية بزاف فين نبدا نشكي", "ما تفاهمتش مع المبلغ ديال التيليكوم لمن نمشي اللول"],
        ["Ma facture de téléphone paraît trop élevée : par qui commencer pour réclamer ?", "Je conteste un montant télécom sans connaître l'ANRT : à qui écrire d'abord ?"],
        ["تبدو فاتورة هاتفي مرتفعة جدا، فلمن أقدم الشكوى أولا؟", "أعترض على مبلغ فاتورة الاتصالات، فإلى من أكتب أولا؟"],
    ),
}


def _n(latn: list[str], arab: list[str], fr: list[str], ar: list[str]) -> dict[str, list[str]]:
    return v01._q(latn, arab, fr, ar)


# Chaque négatif contient trois formulations ordinaires. Aucun mot comme
# "exact", "bddabt" ou "بالضبط" n'est nécessaire pour déclencher l'abstention.
NEGATIVE_INTENTS: dict[str, dict[str, Any]] = {
    "onee_penalty": {"provider_id": "onee", "chunk": "onee-regler-facture-residentiel-20260929t192825z--date-reglement", "questions": _n(
        ["t3ettelt 3la factora dyal loffice ch7al ghadi nzid", "ma khllestch f lwe9t wach kayna gharama", "factora fatat l date chno ghadi y7sbo 3lia"],
        ["تعطلت على فاكتورة المكتب شحال غادي نزيد", "ما خلصتش فالوقت واش كاينة غرامة", "الفاكتورة فاتت التاريخ شنو غادي يحسبو عليا"],
        ["J'ai payé l'office en retard : quelle pénalité vais-je supporter ?", "Une pénalité est-elle facturée si je dépasse l'échéance ?", "Ma facture est échue : quel supplément sera calculé ?"],
        ["تأخرت في أداء فاتورة المكتب، فما الغرامة؟", "هل تفرض غرامة عند تجاوز تاريخ الأداء؟", "انقضى أجل فاتورتي، فما المبلغ الإضافي المحتسب؟"]), "contrast": "amount_requested"},
    "onee_card": {"provider_id": "onee", "chunk": "onee-regler-facture-residentiel-20260929t192825z--paiement-cheque", "questions": _n(
        ["n9der nkhlles had factora b carte bancaire online", "loffice kay9bel paiement b visa mn telephone", "ma bghitch chik wach carte katmchi"],
        ["نقدر نخلص هاد الفاكتورة بالكارت البنكية فالإنترنت", "المكتب كيقبل الأداء بفيزا من التلفون", "ما بغيتش الشيك واش الكارت كتمشي"],
        ["Puis-je régler cette facture ONEE en ligne par carte ?", "L'office accepte-t-il Visa depuis un téléphone ?", "Je ne veux pas payer par chèque : la carte est-elle acceptée ?"],
        ["هل يمكن أداء فاتورة المكتب عبر الإنترنت بالبطاقة؟", "هل يقبل المكتب بطاقة Visa من الهاتف؟", "لا أريد الأداء بشيك، فهل تقبل البطاقة؟"]), "contrast": "ordinary_missing_method"},
    "redal_outage": {"provider_id": "srm_rsk", "chunk": "redal-questions-frequentes-20260930t201159z--facturation-mensuelle", "questions": _n(
        ["t9ta3 do f dar ch7al kayakhdo bach yrj3oh", "3ndi panne redal imta yrj3 courant", "do m9to3 wach kayn délai dyal réparation"],
        ["تقطع الضو فالدار شحال كياخدو باش يرجعوه", "عندي عطب ريدال إمتى يرجع التيار", "الضو مقطوع واش كاين أجل ديال الإصلاح"],
        ["Après une coupure, sous quel délai Redal rétablit-elle le courant ?", "J'ai une panne : quand le courant sera-t-il rétabli ?", "Existe-t-il un délai de réparation après une coupure Redal ?"],
        ["بعد انقطاع الكهرباء، متى تعيد Redal التيار؟", "لدي عطب، فمتى يعود التيار؟", "هل يوجد أجل لإصلاح انقطاع الكهرباء لدى Redal؟"]), "contrast": "ordinary_missing_service_level"},
    "redal_cheque": {"provider_id": "srm_rsk", "chunk": "redal-e-paiement-20260930t201158z--paiement-mobile-jiwar", "questions": _n(
        ["redal kat9bel chik bach nkhlles", "3la smit chkon nkteb cheque dyal lma w do", "fin n3ti chik dyal had factora"],
        ["ريضال كتقبل الشيك باش نخلص", "على سميت شكون نكتب شيك ديال الما والضو", "فين نعطي الشيك ديال هاد الفاكتورة"],
        ["Redal accepte-t-elle le paiement par chèque ?", "À quel ordre établir un chèque pour l'eau et l'électricité Redal ?", "Où remettre le chèque de cette facture ?"],
        ["هل تقبل Redal الأداء بالشيك؟", "باسم من أحرر شيك فاتورة الماء والكهرباء؟", "أين أقدم شيك هذه الفاتورة؟"]), "contrast": "ordinary_provider_confusion"},
    "iam_old_invoice": {"provider_id": "iam", "chunk": "telecom-iam-mobile-postpaid-20261001t082840z--exigibilite-non-reception", "questions": _n(
        ["bghit facture 9dima bzaf kifach nrj3ha", "wach espace client fih factorat dyal snin li fato", "bdelt compte w khasni ancienne facture maroc telecom"],
        ["بغيت فاتورة قديمة بزاف كيفاش نرجعها", "واش فضاء الزبون فيه فواتير ديال السنين لي فاتو", "بدلت الكونط وخاصني فاتورة قديمة ديال اتصالات المغرب"],
        ["Comment récupérer une très ancienne facture Maroc Telecom ?", "L'espace client conserve-t-il les factures des années passées ?", "J'ai changé de compte et il me faut une ancienne facture IAM."],
        ["كيف أسترجع فاتورة قديمة جدا لاتصالات المغرب؟", "هل يحتفظ فضاء الزبون بفواتير السنوات الماضية؟", "غيرت حسابي وأحتاج إلى فاتورة قديمة لاتصالات المغرب."]), "contrast": "ordinary_missing_retention"},
    "iam_agency_hours": {"provider_id": "iam", "chunk": "telecom-iam-mobile-postpaid-20261001t082840z--facturation-mensuelle", "questions": _n(
        ["wakala maroc telecom kat7ell sebt", "f achmen sa3a katsed agence ittisalat", "n9der nmchi l IAM m3a 19h"],
        ["وكالة اتصالات المغرب كتحل السبت", "فاش من ساعة كتسد وكالة اتصالات المغرب", "نقدر نمشي لوكالة IAM مع 19h"],
        ["Les agences Maroc Telecom ouvrent-elles le samedi ?", "À quelle heure ferme une agence IAM ?", "Puis-je aller en agence Ittisalat à 19 h ?"],
        ["هل تفتح وكالات اتصالات المغرب يوم السبت؟", "في أي ساعة تغلق وكالة اتصالات المغرب؟", "هل يمكن الذهاب إلى وكالة اتصالات المغرب عند 19h؟"]), "contrast": "numeric_positive_balance"},
    "orange_refund": {"provider_id": "orange_ma", "chunk": "telecom-orange-payer-facture-20261001t082112z--paiement-recharge", "questions": _n(
        ["khllest orange juj mrat imta yrj3o lia flous", "t3awed lia paiement dyal factora kifach nt3ewed", "double paiement orange ch7al kayakhod remboursement"],
        ["خلصت أورنج جوج مرات إمتى يرجعو ليا الفلوس", "تعاود ليا أداء الفاتورة كيفاش نتعوض", "خلصت أورنج مرتين شحال كياخد التعويض"],
        ["J'ai payé Orange deux fois : quand serai-je remboursé ?", "Le paiement de ma facture est passé deux fois : comment obtenir le remboursement ?", "Combien de temps prend le remboursement d'un double paiement Orange ?"],
        ["أديت فاتورة Orange مرتين، فمتى أسترجع مالي؟", "تكرر أداء فاتورتي، فكيف أحصل على التعويض؟", "كم يستغرق تعويض الأداء المزدوج لدى Orange؟"]), "contrast": "ordinary_missing_refund"},
    "orange_complaint": {"provider_id": "orange_ma", "chunk": "telecom-orange-payer-facture-20261001t082112z--paiement-agents", "questions": _n(
        ["factora orange ghalta fin ndir chikaya", "orange 7sbat lia montant ma fhemtouch lmen ntwjeh", "bghit n3tared 3la facture dyal telephone"],
        ["فاكتورة أورنج غالطة فين ندير شكاية", "أورنج حسبات ليا مبلغ ما فهمتوش لمن نتوجه", "بغيت نعترض على فاتورة ديال التلفون"],
        ["Ma facture Orange semble erronée : où déposer une réclamation ?", "Orange m'a facturé un montant incompris : qui contacter ?", "Je souhaite contester ma facture de téléphone Orange."],
        ["تبدو فاتورة Orange خاطئة، فأين أقدم شكوى؟", "احتسبت Orange مبلغا لم أفهمه، فمن أتصل به؟", "أريد الاعتراض على فاتورة هاتف Orange."]), "contrast": "ordinary_missing_procedure"},
    "inwi_failed_payment": {"provider_id": "inwi", "chunk": "telecom-inwi-payer-facture-20261001t082113z--paiement-agence", "questions": _n(
        ["paiement inwi fchel w flous t9t3o imta yrj3o", "khllest online w ma dazetsh l3amaliya chno ndir", "transaction dyal factora ma nj7atsh ch7al remboursement"],
        ["أداء inwi فشل والفلوس تقطعو إمتى يرجعو", "خلصت فالإنترنت وما دازتش العملية شنو ندير", "عملية الفاتورة ما نجحاتش شحال التعويض"],
        ["Mon paiement inwi a échoué mais le montant a été débité : quand revient-il ?", "Le règlement en ligne n'a pas abouti : que faire pour être remboursé ?", "Quel délai de remboursement après une transaction inwi échouée ?"],
        ["فشل أداء inwi لكن المبلغ اقتطع، فمتى يرجع؟", "لم تنجح عملية الأداء عبر الإنترنت، فكيف أسترجع المبلغ؟", "ما أجل التعويض بعد فشل عملية أداء inwi؟"]), "contrast": "ordinary_missing_refund"},
    "inwi_double_payment": {"provider_id": "inwi", "chunk": "telecom-inwi-payer-facture-20261001t082113z--paiement-my-inwi", "questions": _n(
        ["application khllsat lia nafs factora juj mrat kifach nrj3 flous", "t7seb 3lia paiement mrrtin f my inwi fin nchki", "double débit dyal inwi wach kaytrj3 bo7do"],
        ["الأبليكاسيون خلصات ليا نفس الفاكتورة جوج مرات كيفاش نرجع الفلوس", "تحسب عليا الأداء مرتين فـ My inwi فين نشكي", "اقتطاع مزدوج ديال inwi واش كيرجع بوحدو"],
        ["My inwi a payé la même facture deux fois : comment récupérer l'argent ?", "J'ai été débité deux fois dans My inwi : où réclamer ?", "Un double débit inwi est-il remboursé automatiquement ?"],
        ["أدى تطبيق My inwi الفاتورة نفسها مرتين، فكيف أسترجع المال؟", "اقتطع مني المبلغ مرتين في My inwi، فأين أشتكي؟", "هل يعاد الاقتطاع المزدوج لدى inwi تلقائيا؟"]), "contrast": "ordinary_missing_refund"},
}


ASR_REPLACEMENTS = {
    "darija_latn": [("kifach", "kifsh"), ("chno", "shno"), ("wach", "wash"), ("n9der", "nqdr"), ("facture", "factora")],
    "darija_arab": [("كيفاش", "كفاش"), ("شنو", "اش"), ("فاتورة", "فاكتورة"), ("الهاتف", "التلفون")],
    "fr": [("facture", "factur"), ("paiement", "paiment"), ("Comment", "comment")],
    "ar": [("فاتورة", "الفاتورة"), ("ماذا", "ما الذي")],
}


def asr_variant(question: str, language: str, alternate: bool = False) -> str:
    """Produit un bruit léger et déterministe, sans toucher aux valeurs."""
    result = question.strip().rstrip("?.،")
    replacements = ASR_REPLACEMENTS[language]
    selected = replacements[1::2] if alternate else replacements[::2]
    original = result
    for source, target in selected:
        result = result.replace(source, target)
    if language in {"darija_latn", "fr"}:
        result = result.lower()
    result = re.sub(r"\s+", " ", result)
    if result.casefold() == original.casefold():
        if language == "darija_latn":
            result = result.replace("a", "", 1)
        elif language == "darija_arab":
            result = result.replace("ا", "", 1)
        elif language == "fr":
            result = result.replace("e", "", 1)
        else:
            result = result.replace("أ", "ا", 1)
    if result.casefold() == original.casefold():
        # Dernier recours typographique : espace manquant fréquent en saisie
        # mobile. Il garde strictement les nombres et le sens.
        result = result.replace(" ", "", 1)
    return result


def oral_variant(question: str, language: str) -> str:
    """Reformulation orale utilisée pour équilibrer les graines FR/AR."""
    result = question.strip().rstrip("?.،")
    if language == "ar":
        if result.startswith("هل "):
            return result[3:]
        if result.startswith("كيف "):
            return "بأي طريقة " + result[4:]
        return "أريد أن أعرف " + result
    if language == "fr":
        if result.startswith("Puis-je "):
            return "Est-ce que je peux " + result[8:].lower()
        if result.startswith("Comment "):
            return "Je voudrais savoir comment " + result[8:].lower()
        return "Je voudrais savoir si " + result[0].lower() + result[1:]
    return asr_variant(result, language)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def mention_provider(question: str, provider_id: str) -> bool:
    aliases = {
        "onee": ("onee", "office", "loffice", "المكتب", "الماء والكهرباء"),
        "srm_rsk": ("redal", "ridal", "ريضال"),
        "iam": ("iam", "maroc telecom", "ittisalat", "اتصالات المغرب"),
        "orange_ma": ("orange", "orange", "أورنج"),
        "inwi": ("inwi",),
        "anrt": ("anrt", "الوكالة الوطنية"),
    }
    folded = question.casefold()
    return any(alias in folded for alias in aliases.get(provider_id, (provider_id,)))


def question_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    qid = 1
    positive_styles = ("natural", "indirect", "lexical_gap", "edge_or_numeric", "contextual", "alias_or_implicit", "asr", "whatsapp")
    negative_styles = ("natural", "contextual", "alias_or_implicit", "asr", "whatsapp", "indirect")
    positive_splits = ("train", "train", "validation", "test", "train", "train", "train", "test")
    negative_splits = ("train", "train", "validation", "train", "test", "test")

    for intent_id, spec in v01.POSITIVE_INTENTS.items():
        for language in LANGUAGES:
            base = list(spec["questions"][language]) + POSITIVE_EXTRAS[intent_id][language]
            # Les familles historiques FR/AR avaient trois graines contre
            # quatre en darija. On complète avec une transcription orale
            # contrôlée plutôt qu'avec un préfixe poli générique.
            if len(base) == 5:
                base.append(oral_variant(base[1], language))
            questions = base + [asr_variant(base[0], language), asr_variant(base[2], language, True)]
            if len(questions) != 8 or len(set(questions)) != 8:
                raise ValueError(f"Variantes positives invalides: {intent_id}/{language}")
            provider_id = v01.load_generation_cases(v01.BASE_CASES, v01.GENERATION_SCHEMA)
            provider = next(c["provider_id"] for c in provider_id if c["case_id"] == spec["source_case_id"])
            for question, style, split in zip(questions, positive_styles, positive_splits):
                rows.append({"schema_version": "0.2.0", "question_id": f"sftq-{qid:04d}", "intent_id": intent_id, "source_case_id": spec["source_case_id"], "language": language, "provider_id": provider, "style": style, "split": split, "should_abstain": False, "provider_mentioned": mention_provider(question, provider), "contrast_group": "answerable", "question": question, "distractor_chunk_id": None})
                qid += 1

    for intent_id, spec in NEGATIVE_INTENTS.items():
        for language in LANGUAGES:
            base = spec["questions"][language]
            questions = base + [asr_variant(base[0], language), asr_variant(base[1], language, True), asr_variant(base[2], language)]
            if len(questions) != 6 or len(set(questions)) != 6:
                raise ValueError(f"Variantes négatives invalides: {intent_id}/{language}")
            for question, style, split in zip(questions, negative_styles, negative_splits):
                rows.append({"schema_version": "0.2.0", "question_id": f"sftq-{qid:04d}", "intent_id": intent_id, "source_case_id": None, "language": language, "provider_id": spec["provider_id"], "style": style, "split": split, "should_abstain": True, "provider_mentioned": mention_provider(question, spec["provider_id"]), "contrast_group": spec["contrast"], "question": question, "distractor_chunk_id": spec["chunk"]})
                qid += 1
    return rows


def main() -> None:
    base_cases = v01.load_generation_cases(v01.BASE_CASES, v01.GENERATION_SCHEMA)
    cases_by_id = {case["case_id"]: case for case in base_cases}
    chunks = [chunk for path in v01.CHUNK_PATHS for chunk in load_active_chunks(path, CHUNK_SCHEMA)]
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    validator = Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8")))

    questions = question_rows()
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []
    for index, item in enumerate(questions, start=1):
        if item["should_abstain"]:
            contexts = [chunks_by_id[item["distractor_chunk_id"]]]
            answer = v01.make_abstention(item["language"])
            critical_values: list[str] = []
        else:
            source_case = cases_by_id[item["source_case_id"]]
            contexts = [chunks_by_id[cid] for cid in source_case["gold_chunk_ids"]]
            answer = v01.make_answer(item["language"], v01.POSITIVE_INTENTS[item["intent_id"]]["claims"][item["language"]], contexts[0]["chunk_id"])
            critical_values = list(source_case["critical_values"])
        case = {"language": item["language"], "provider_id": item["provider_id"], "question": item["question"]}
        common = {key: item[key] for key in ("intent_id", "source_case_id", "language", "provider_id", "style", "should_abstain", "provider_mentioned", "contrast_group", "question")}
        if item["split"] == "test":
            tests.append({"schema_version": "0.2.0", "case_id": item["question_id"], **common, "split": "test", "critical_values": critical_values, "source_chunk_ids": [c["chunk_id"] for c in contexts]})
            continue
        record = {"schema_version": "0.2.0", "sample_id": f"sft-v02-{index:04d}", **common, "split": item["split"], "critical_values": critical_values, "source_chunk_ids": [c["chunk_id"] for c in contexts], "review_status": "candidate_human_review", "messages": [{"role": "user", "content": build_generation_prompt(case, contexts, "darija_few_shot")}, {"role": "assistant", "content": answer.model_dump_json()}]}
        validator.validate(record)
        (train if item["split"] == "train" else validation).append(record)

    normalized = lambda text: re.sub(r"\s+", " ", text).strip().casefold()
    split_sets = {name: {normalized(row["question"]) for row in data} for name, data in (("train", train), ("validation", validation), ("test", tests))}
    if split_sets["train"] & split_sets["validation"] or (split_sets["train"] | split_sets["validation"]) & split_sets["test"]:
        raise ValueError("Fuite textuelle entre les splits")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {name: OUTPUT_DIR / f"{name}.jsonl" for name in ("questions", "train", "validation", "test_cases", "test_prompts", "review_queue")}
    write_jsonl(paths["questions"], questions)
    write_jsonl(paths["train"], train)
    write_jsonl(paths["validation"], validation)
    write_jsonl(paths["test_cases"], tests)
    test_prompts = []
    for item in tests:
        contexts = [chunks_by_id[cid] for cid in item["source_chunk_ids"]]
        case = {"language": item["language"], "provider_id": item["provider_id"], "question": item["question"]}
        test_prompts.append({"case_id": item["case_id"], "messages": [{"role": "user", "content": build_generation_prompt(case, contexts, "darija_few_shot")} ]})
    write_jsonl(paths["test_prompts"], test_prompts)

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for item in questions:
        key = (item["intent_id"], item["language"])
        group = grouped.setdefault(key, {"review_group_id": f"{item['intent_id']}--{item['language']}", "intent_id": item["intent_id"], "language": item["language"], "provider_id": item["provider_id"], "should_abstain": item["should_abstain"], "questions": [], "human_decision": None, "human_comment": ""})
        group["questions"].append({"question_id": item["question_id"], "split": item["split"], "style": item["style"], "provider_mentioned": item["provider_mentioned"], "text": item["question"]})
    write_jsonl(paths["review_queue"], list(grouped.values()))

    manifest = {"schema_version": "0.2.0", "status": "rejected_split_leakage", "rejection_reason": "Intentions, chunks et quasi-paraphrases traversent les splits; ne pas entraîner.", "base_model": "Qwen/Qwen3.5-4B", "prompt_profile": "darija_few_shot", "seed": 42, "counts": {"questions": len(questions), "train": len(train), "validation": len(validation), "test": len(tests), "review_groups": len(grouped)}, "language_counts": dict(sorted(Counter(q["language"] for q in questions).items())), "style_counts": dict(sorted(Counter(q["style"] for q in questions).items())), "split_counts": dict(sorted(Counter(q["split"] for q in questions).items())), "abstention_counts_by_split": {split: sum(q["should_abstain"] and q["split"] == split for q in questions) for split in ("train", "validation", "test")}, "provider_implicit_count": sum(not q["provider_mentioned"] for q in questions), "test_policy": "Dataset rejeté après audit; remplacé par la banque v0.3 groupée par intent.", "files": {name: {"path": path.name, "sha256": sha256(path)} for name, path in paths.items()}}
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
