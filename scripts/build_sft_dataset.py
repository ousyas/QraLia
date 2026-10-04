"""Construit le dataset SFT QraLia sans ouvrir ni copier le split de test.

Les réponses sont des cibles pédagogiques déterministes relues par schéma.
Elles restent marquées ``candidate_human_review`` tant qu'Oussama et Meryeme
n'ont pas validé la naturalité de chaque famille linguistique.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from qralia.rag.bm25_evaluation import load_active_chunks
from qralia.rag.generation_evaluation import (
    AnswerClaim,
    GroundedAnswer,
    build_generation_prompt,
    load_generation_cases,
    render_answer_from_claims,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "training" / "qralia_sft_v0.1"
SFT_SCHEMA = ROOT / "schemas" / "sft_example.schema.json"
GENERATION_SCHEMA = ROOT / "schemas" / "generation_evaluation_case.schema.json"
BASE_CASES = ROOT / "data" / "evaluation" / "generation_questions.v0.1.jsonl"
DARIIJA_BENCHMARK = ROOT / "data" / "evaluation" / "generation_darija_questions.v0.2.jsonl"
CHUNK_PATHS = [
    ROOT / "data" / "processed" / "onee-regler-facture-residentiel" / "chunks.jsonl",
    ROOT / "data" / "processed" / "redal" / "chunks.jsonl",
    ROOT / "data" / "processed" / "telecom" / "chunks.jsonl",
]
CHUNK_SCHEMA = ROOT / "schemas" / "retrieval_chunk.schema.json"

LANGUAGES = ("darija_latn", "darija_arab", "fr", "ar")
TRAIN_COUNTS = {"darija_latn": 16, "darija_arab": 16, "fr": 7, "ar": 7}
VALIDATION_COUNTS = {"darija_latn": 3, "darija_arab": 3, "fr": 2, "ar": 2}
NEGATIVE_TRAIN_COUNTS = {"darija_latn": 8, "darija_arab": 8, "fr": 3, "ar": 3}
NEGATIVE_VALIDATION_COUNTS = {"darija_latn": 1, "darija_arab": 1, "fr": 1, "ar": 1}

PREFIXES = {
    "darija_latn": ["", "3afak, ", "Bghit n3ref: ", "Momkin tgoul lia: "],
    "darija_arab": ["", "عافاك، ", "بغيت نعرف: ", "ممكن تقول ليا: "],
    "fr": ["", "S'il vous plaît, ", "J'aimerais savoir : "],
    "ar": ["", "من فضلك، ", "أريد أن أعرف: "],
}
SUFFIXES = {
    "darija_latn": ["", " 3afak.", " Ila mumkin.", " Bghit jawab wad7."],
    "darija_arab": ["", " عافاك.", " إلا ممكن.", " بغيت جواب واضح."],
    "fr": ["", " Merci.", " Pouvez-vous me répondre ?", " J'ai besoin d'une réponse précise."],
    "ar": ["", " من فضلك.", " أرجو جوابا واضحا.", " هل توجد معلومة مؤكدة؟"],
}


def _q(latn: list[str], arab: list[str], fr: list[str], ar: list[str]) -> dict[str, list[str]]:
    return {"darija_latn": latn, "darija_arab": arab, "fr": fr, "ar": ar}


POSITIVE_INTENTS: dict[str, dict[str, Any]] = {
    "onee_cheque": {
        "source_case_id": "generation-001",
        "questions": _q(
            [
                "Chkon khasni nkteb f chik dyal ONEE w fin ndfe3o?",
                "Chik bancaire wla CCP dyal ONEE, b smiya dyal chkon?",
                "Fin n9dem chik dyal facture ONEE?",
                "Bghit nkhlles ONEE b chik, ach khasni ndir?",
            ],
            [
                "فسمية شكون نكتب الشيك ديال ONEE وفين نقدمو؟",
                "واش الشيك ديال ONEE يقدر يكون بنكي ولا CCP؟",
                "فين خاصني ندي الشيك باش نخلّص فاتورة ONEE؟",
                "بغيت نخلّص ONEE بالشيك، شنو ندير؟",
            ],
            [
                "À quel ordre établir le chèque ONEE et où le déposer ?",
                "Puis-je payer ONEE par chèque bancaire ou CCP ?",
                "Comment payer une facture ONEE par chèque ?",
            ],
            [
                "لأي جهة يحرر شيك فاتورة المكتب الوطني وأين يقدم؟",
                "هل يمكن الأداء بشيك بنكي أو شيك CCP؟",
                "كيف تؤدى فاتورة المكتب الوطني بالشيك؟",
            ],
        ),
        "claims": {
            "darija_latn": [
                "T9der tkhlles b chik bancaire wla CCP.",
                "Khass chik ykoun b smiya dyal l'Office National de l'Électricité et de l'Eau Potable, w t9edmo f agence ONEE Branche Électricité.",
            ],
            "darija_arab": [
                "تقدر تخلّص بشيك بنكي ولا CCP.",
                "خاص الشيك يكون بسمية Office National de l'Électricité et de l'Eau Potable، وتقدمو فوكالة ONEE Branche Électricité.",
            ],
            "fr": [
                "Le chèque peut être bancaire ou CCP.",
                "Il doit être établi à l'ordre de l'Office National de l'Électricité et de l'Eau Potable et remis à une agence ONEE Branche Électricité.",
            ],
            "ar": [
                "يمكن الأداء بشيك بنكي أو شيك CCP.",
                "يحرر الشيك باسم Office National de l'Électricité et de l'Eau Potable ويقدم إلى وكالة ONEE Branche Électricité.",
            ],
        },
    },
    "onee_holiday": {
        "source_case_id": "generation-003",
        "questions": _q(
            [
                "Ila date dyal lkhlass jat f nhar 3otla, ach kaytra?",
                "Imta nkhlles ila délai ONEE tsadef m3a jour ferie?",
                "Nhar lkhlass dyal ONEE ja f 3otla, wach kayet2ejjel?",
                "Facture ONEE khas tkhlles f nhar ferie, chno ndir?",
            ],
            [
                "إلا تاريخ الخلاص جا فنهار عطلة، آش كيوقع؟",
                "إمتى نخلّص إلا أجل ONEE صادف يوم عطلة؟",
                "نهار خلاص ONEE جا فعطلة، واش كيتأجل؟",
                "الفاتورة خاصها تخلص فنهار عطلة، شنو ندير؟",
            ],
            [
                "Que se passe-t-il si la date de paiement tombe un jour férié ?",
                "Quand payer ONEE lorsque l'échéance coïncide avec un jour férié ?",
                "La date de règlement est-elle décalée pendant un jour férié ?",
            ],
            [
                "ماذا يحدث إذا وافق تاريخ الأداء يوم عطلة؟",
                "متى تؤدى فاتورة المكتب الوطني إذا صادف الأجل يوم عطلة؟",
                "هل يؤجل تاريخ الأداء عندما يصادف يوم عطلة؟",
            ],
        ),
        "claims": {
            "darija_latn": ["Ila date dyal lkhlass tsadfat m3a nhar 3otla, katet2ejjel l nhar lkhedma li men b3d."],
            "darija_arab": ["إلا تاريخ الخلاص صادف نهار عطلة، كيتأجل لنهار الخدمة اللي من بعد."],
            "fr": ["Si la date de règlement coïncide avec un jour férié, elle est décalée au jour ouvrable suivant."],
            "ar": ["إذا صادف تاريخ الأداء يوم عطلة، يؤجل إلى يوم العمل الموالي."],
        },
    },
    "redal_orange_money": {
        "source_case_id": "generation-006",
        "questions": _q(
            [
                "N9der nkhlles Redal b Orange Money men téléphone?",
                "Orange Money kaykhdem bach nkhlles facture Redal?",
                "Kifach nkhlles Redal men téléphone b Orange Money?",
                "Service Orange Money dyal Redal kaykhdem dima?",
            ],
            [
                "نقدر نخلّص Redal بـ Orange Money من التليفون؟",
                "واش Orange Money كيخدم لخلاص فاتورة Redal؟",
                "كيفاش نخلّص Redal من التليفون بـ Orange Money؟",
                "واش خدمة Orange Money ديال Redal خدامة ديما؟",
            ],
            [
                "Puis-je payer Redal par téléphone avec Orange Money ?",
                "Le paiement Redal avec Orange Money est-il toujours disponible ?",
                "Comment utiliser Orange Money pour régler Redal ?",
            ],
            [
                "هل يمكن أداء فاتورة Redal عبر الهاتف باستعمال Orange Money؟",
                "هل خدمة Orange Money لأداء Redal متاحة باستمرار؟",
                "كيف أؤدي فاتورة Redal بواسطة Orange Money؟",
            ],
        ),
        "claims": {
            "darija_latn": ["T9der tkhlles Redal men téléphone b Orange Money.", "Had service disponible 24h/24 w 7j/7."],
            "darija_arab": ["تقدر تخلّص Redal من التليفون بـ Orange Money.", "هاد الخدمة متوفرة 24h/24 و7j/7."],
            "fr": ["Le règlement Redal par téléphone est disponible via Orange Money.", "Le service est disponible 24h/24 et 7j/7."],
            "ar": ["يمكن أداء فاتورة Redal عبر الهاتف بواسطة Orange Money.", "الخدمة متاحة 24h/24 و7j/7."],
        },
    },
    "redal_monthly": {
        "source_case_id": "generation-008",
        "questions": _q(
            [
                "Facture Redal katji kol chher w ach katjme3?",
                "Redal katseft facture chheriya dyal chno?",
                "Wach lma w do kayjiw f nafs facture Redal?",
                "Chno kayn f facture chheriya dyal Redal?",
            ],
            [
                "فاتورة Redal كتجي كل شهر وشنو كتجمع؟",
                "Redal كتصيفط فاتورة شهرية ديال شنو؟",
                "واش الما والضو كيجيو فنفس فاتورة Redal؟",
                "شنو كاين فالفاتورة الشهرية ديال Redal؟",
            ],
            [
                "La facture Redal est-elle mensuelle et quels services regroupe-t-elle ?",
                "L'eau et l'électricité figurent-elles sur la même facture Redal ?",
                "Que contient la facture mensuelle Redal ?",
            ],
            [
                "هل فاتورة Redal شهرية وما الخدمات التي تجمعها؟",
                "هل يجتمع الماء والكهرباء في فاتورة Redal نفسها؟",
                "ماذا تتضمن فاتورة Redal الشهرية؟",
            ],
        ),
        "claims": {
            "darija_latn": ["Facture Redal katji kol chher, w katjme3 lma, lkahraba w ttathir ssayel."],
            "darija_arab": ["فاتورة Redal كتجي كل شهر، وكتجمع الما والكهربا والتطهير السائل."],
            "fr": ["La facture Redal est mensuelle et regroupe l'eau, l'électricité et l'assainissement liquide."],
            "ar": ["فاتورة Redal شهرية وتجمع الماء والكهرباء والتطهير السائل."],
        },
    },
    "iam_contestation": {
        "source_case_id": "generation-010",
        "questions": _q(
            [
                "Ch7al 3ndi bach n3tared 3la prélèvement IAM 3nd banque?",
                "Banque 9t3at lia montant IAM, imta n9der ncontestih?",
                "Kifach w f achmen délai n3tared 3la débit IAM?",
                "Daz délai dyal opposition IAM, wach ba9i n9der nchteki?",
            ],
            [
                "شحال عندي باش نعترض على اقتطاع IAM عند البنك؟",
                "البنك اقتاطع ليا مبلغ IAM، إمتى نقدر نحتج؟",
                "كيفاش وفاش من أجل نعترض على اقتطاع IAM؟",
                "داز أجل الاعتراض، واش باقي نقدر نشتكي لـ IAM؟",
            ],
            [
                "Quel est le délai pour contester un prélèvement IAM auprès de la banque ?",
                "Comment contester par écrit un débit IAM ?",
                "Puis-je encore adresser une contestation à IAM après le délai bancaire ?",
            ],
            [
                "ما أجل الاعتراض على اقتطاع IAM لدى البنك؟",
                "كيف أعترض كتابيا على مبلغ اقتطعته IAM؟",
                "هل يمكن تقديم اعتراض إلى IAM بعد انقضاء الأجل البنكي؟",
            ],
        ),
        "claims": {
            "darija_latn": ["Khass l'i3tirad 3nd banque ykoun ktebi f 7 iyam men b3d prélèvement.", "Mora had délai, had procédure ma katrjje3ch lmontant, walakin t9der tsift contestation m3ella l IAM."],
            "darija_arab": ["خاص الاعتراض عند البنك يكون كتابي فـ 7 أيام من بعد الاقتطاع.", "من بعد هاد الأجل، هاد المسطرة ما كترجعش المبلغ، ولكن تقدر تصيفط اعتراض معلل لـ IAM."],
            "fr": ["La contestation auprès de la banque doit être faite par écrit dans les 7 jours suivant le débit.", "Après ce délai, cette procédure ne permet plus la restitution, mais une contestation motivée peut être adressée à IAM."],
            "ar": ["يجب تقديم الاعتراض إلى البنك كتابة خلال 7 أيام من تاريخ الاقتطاع.", "بعد هذا الأجل لا تعاد الأموال بهذه المسطرة، لكن يمكن إرسال اعتراض معلل إلى IAM."],
        },
    },
    "orange_recharge": {
        "source_case_id": "generation-012",
        "questions": _q(
            [
                "Kifach nkhlles facture Orange b recharge?",
                "Ch7al n3emmer w fin n3ayet bach nkhlles Orange?",
                "N9der nkhlles Orange b chharj wla la?",
                "Ach ndir men b3d ma n3emmer recharge dyal facture Orange?",
            ],
            [
                "كيفاش نخلّص فاتورة Orange بالشارج؟",
                "شحال نعمّر وفين نعيط باش نخلّص Orange؟",
                "نقدر نخلّص Orange بالتعبئة ولا لا؟",
                "شنو ندير من بعد ما نعمّر تعبئة فاتورة Orange؟",
            ],
            [
                "Comment payer une facture Orange par recharge ?",
                "Quel montant recharger et quel numéro appeler pour Orange ?",
                "Que faire après avoir rechargé le montant de la facture Orange ?",
            ],
            [
                "كيف أؤدي فاتورة Orange بواسطة التعبئة؟",
                "ما المبلغ الواجب تعبئته وما الرقم الذي يجب الاتصال به؟",
                "ماذا أفعل بعد تعبئة قيمة فاتورة Orange؟",
            ],
        ),
        "claims": {
            "darija_latn": ["Khassk t3emmer recharge b montant kay9ed wla kayfout montant dyal facture.", "Men b3d 3ayet l 5555 bach t2ekked l'opération."],
            "darija_arab": ["خاصك تعمّر بتعبئة كتساوي ولا كتفوت مبلغ الفاتورة.", "من بعد عيط لـ 5555 باش تأكد العملية."],
            "fr": ["Il faut effectuer une recharge d'un montant égal ou supérieur au montant de la facture.", "Il faut ensuite appeler le 5555 pour confirmer l'opération."],
            "ar": ["يجب تعبئة رصيد بقيمة تساوي أو تفوق قيمة الفاتورة.", "بعد ذلك يجب الاتصال بالرقم 5555 لتأكيد العملية."],
        },
    },
    "inwi_hours": {
        "source_case_id": "generation-013",
        "questions": _q(
            [
                "Fach kat7ell w katsed wakalat inwi?",
                "Chno homa swaye3 lkhedma dyal agences inwi?",
                "N9der nmchi l agence inwi nhar sebt w f achmen sa3a?",
                "Men imta l imta n9der nkhlles facture f wakala inwi?",
            ],
            [
                "فاش كتحل وكتسد وكالات inwi؟",
                "شنو هما سوايع الخدمة ديال وكالات inwi؟",
                "نقدر نمشي لوكالة inwi نهار السبت وفاش من ساعة؟",
                "من إمتى لإمتى نقدر نخلّص الفاتورة فوكالة inwi؟",
            ],
            [
                "Quels sont les horaires des agences inwi ?",
                "Les agences inwi ouvrent-elles le samedi ?",
                "De quelle heure à quelle heure peut-on payer en agence inwi ?",
            ],
            [
                "ما أوقات عمل وكالات inwi؟",
                "هل تفتح وكالات inwi يوم السبت؟",
                "بين أي ساعتين يمكن أداء الفاتورة في وكالة inwi؟",
            ],
        ),
        "claims": {
            "darija_latn": ["Wakalat inwi kaykhdmo men ltnin 7ta sebt, men 9h d sba7 7ta 20h."],
            "darija_arab": ["وكالات inwi كيخدمو من الاثنين حتى السبت، من 9h دالصباح حتى 20h."],
            "fr": ["Les agences inwi sont ouvertes du lundi au samedi de 9h à 20h."],
            "ar": ["وكالات inwi مفتوحة من الاثنين إلى السبت، من 9h إلى 20h."],
        },
    },
    "anrt_complaint": {
        "source_case_id": "generation-015",
        "questions": _q(
            [
                "9bel ANRT, fin nsift chikaya dyal facture telecom?",
                "Lmen ntwjeh lwl ila bghit nchteki 3la facture telecom?",
                "N9der nbda directement b ANRT wla khasni nchki l opérateur?",
                "Kifach nsift reclamation l opérateur telecom?",
            ],
            [
                "قبل ANRT، فين نصيفط شكاية ديال فاتورة الاتصالات؟",
                "لمن نتوجه اللول إلا بغيت نشتكي على فاتورة الاتصالات؟",
                "نقدر نبدا مباشرة بـ ANRT ولا خاصني نشكي للمتعهد؟",
                "كيفاش نصيفط شكاية لمتعهد الاتصالات؟",
            ],
            [
                "À qui adresser d'abord une réclamation sur une facture télécom ?",
                "Faut-il contacter l'opérateur avant de saisir l'ANRT ?",
                "Comment envoyer une réclamation à l'opérateur télécom ?",
            ],
            [
                "إلى من توجه الشكوى أولا قبل اللجوء إلى ANRT؟",
                "هل يجب الاتصال بالمتعهد قبل تقديم شكوى إلى ANRT؟",
                "كيف ترسل الشكوى إلى متعهد الاتصالات؟",
            ],
        ),
        "claims": {
            "darija_latn": ["Khassk lwl twjjeh chikaya directement l'agence commerciale dyal opérateur telecom, wla tsiftha b risala madmouna.", "Sowel 3la délai w chorout dyal 9oboul chikaya."],
            "darija_arab": ["خاصك اللول توجه الشكاية مباشرة للوكالة التجارية ديال متعهد الاتصالات، ولا تصيفطها برسالة مضمونة.", "سول على الأجل وشروط قبول الشكاية."],
            "fr": ["Il faut d'abord adresser la réclamation directement à l'agence commerciale de l'opérateur télécom ou l'envoyer par lettre recommandée.", "Il faut se renseigner sur le délai et les conditions de recevabilité."],
            "ar": ["يجب أولا توجيه الشكوى مباشرة إلى الوكالة التجارية لمتعهد الاتصالات أو إرسالها برسالة مضمونة.", "ينبغي الاستعلام عن الأجل وشروط قبول الشكوى."],
        },
    },
}


NEGATIVE_INTENTS: dict[str, dict[str, Any]] = {
    "onee_penalty": {"provider_id": "onee", "distractor_chunk_id": "onee-regler-facture-residentiel-20260929t192825z--date-reglement", "topic": {"darija_latn": "Ch7al pénalité exacte dyal retard ONEE?", "darija_arab": "شحال الغرامة المضبوطة ديال التأخر عند ONEE؟", "fr": "Quel est le montant exact de la pénalité de retard ONEE ?", "ar": "ما المبلغ الدقيق لغرامة التأخر لدى ONEE؟"}},
    "redal_outage": {"provider_id": "srm_rsk", "distractor_chunk_id": "redal-questions-frequentes-20260930t201159z--facturation-mensuelle", "topic": {"darija_latn": "Redal ch7al katakhod bddabt bach trje3 do mor panne?", "darija_arab": "Redal شحال كتاخد بالضبط باش ترجع الضو من بعد العطب؟", "fr": "Quel délai exact Redal garantit-elle pour rétablir le courant ?", "ar": "ما الأجل الدقيق الذي تضمنه Redal لإعادة الكهرباء؟"}},
    "iam_old_invoice": {"provider_id": "iam", "distractor_chunk_id": "telecom-iam-mobile-postpaid-20261001t082840z--exigibilite-non-reception", "topic": {"darija_latn": "Fin n9der nl9a facture IAM dyal 6 snin hadi?", "darija_arab": "فين نقدر نلقى فاتورة IAM ديال ست سنين هادي؟", "fr": "Comment retrouver une facture IAM vieille de six ans ?", "ar": "كيف أسترجع فاتورة IAM مضى عليها ست أعوام؟"}},
    "orange_refund": {"provider_id": "orange_ma", "distractor_chunk_id": "telecom-orange-payer-facture-20261001t082112z--paiement-recharge", "topic": {"darija_latn": "Ch7al délai dyal remboursement Orange ila khllest marrtin?", "darija_arab": "شحال أجل التعويض عند Orange إلا خلّصت جوج مرات؟", "fr": "Quel est le délai de remboursement Orange après un double paiement ?", "ar": "ما أجل التعويض لدى Orange بعد أداء الفاتورة مرتين؟"}},
    "inwi_failed_payment": {"provider_id": "inwi", "distractor_chunk_id": "telecom-inwi-payer-facture-20261001t082113z--paiement-agence", "topic": {"darija_latn": "Ch7al kayt3ettel remboursement ila paiement inwi online fchel?", "darija_arab": "شحال كيتأخر التعويض إلا فشل أداء inwi فالإنترنت؟", "fr": "Sous quel délai inwi rembourse-t-elle un paiement en ligne échoué ?", "ar": "ما أجل استرجاع مبلغ أداء inwi إلكتروني فاشل؟"}},
}

ABSTENTIONS = {
    "darija_latn": "Ma 3ndich ma3loumat kafya f passages li t3taw lia bach n2ekked lik had lma3louma.",
    "darija_arab": "ما عنديش معلومات كافية فالمقاطع اللي تعطاو ليا باش نأكد ليك هاد المعلومة.",
    "fr": "Les passages fournis ne contiennent pas assez d'informations pour confirmer ce point.",
    "ar": "لا تحتوي المقاطع المقدمة على معلومات كافية لتأكيد ذلك.",
}


def expand_questions(seeds: list[str], language: str, count: int) -> list[str]:
    variants: list[str] = []
    for suffix in SUFFIXES[language]:
        for prefix in PREFIXES[language]:
            for seed in seeds:
                candidate = prefix + seed[0].lower() + seed[1:] if prefix else seed
                candidate += suffix
                if candidate not in variants:
                    variants.append(candidate)
    if len(variants) < count:
        raise ValueError(f"Pas assez de variantes {language}: {len(variants)} < {count}")
    return variants[:count]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl_write(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


def make_answer(language: str, claims: list[str], chunk_id: str) -> GroundedAnswer:
    answer = GroundedAnswer(
        answer="",
        claims=[AnswerClaim(text=claim, citations=[chunk_id]) for claim in claims],
        abstain=False,
        abstention_reason=None,
        language=language,
    )
    return render_answer_from_claims(answer)


def make_abstention(language: str) -> GroundedAnswer:
    return GroundedAnswer(
        answer=ABSTENTIONS[language],
        claims=[],
        abstain=True,
        abstention_reason=ABSTENTIONS[language],
        language=language,
    )


def record_for(
    *,
    sample_number: int,
    intent_id: str,
    source_case_id: str | None,
    language: str,
    provider_id: str,
    split: str,
    question: str,
    critical_values: list[str],
    contexts: list[dict[str, Any]],
    answer: GroundedAnswer,
    should_abstain: bool,
) -> dict[str, Any]:
    case = {
        "language": language,
        "provider_id": provider_id,
        "question": question,
    }
    return {
        "schema_version": "0.1.0",
        "sample_id": f"sft-{intent_id.replace('_', '-')}-{sample_number:03d}",
        "intent_id": intent_id,
        "source_case_id": source_case_id,
        "language": language,
        "provider_id": provider_id,
        "question": question,
        "split": split,
        "should_abstain": should_abstain,
        "critical_values": critical_values,
        "source_chunk_ids": [chunk["chunk_id"] for chunk in contexts],
        "review_status": "candidate_human_review",
        "messages": [
            {
                "role": "user",
                "content": build_generation_prompt(case, contexts, "darija_few_shot"),
            },
            {
                "role": "assistant",
                "content": answer.model_dump_json(),
            },
        ],
    }


def main() -> None:
    base_cases = load_generation_cases(BASE_CASES, GENERATION_SCHEMA)
    cases_by_id = {case["case_id"]: case for case in base_cases}
    chunks = [chunk for path in CHUNK_PATHS for chunk in load_active_chunks(path, CHUNK_SCHEMA)]
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    validator = Draft202012Validator(json.loads(SFT_SCHEMA.read_text(encoding="utf-8")))

    records: list[dict[str, Any]] = []
    sample_number = 1
    for intent_id, spec in POSITIVE_INTENTS.items():
        source_case = cases_by_id[spec["source_case_id"]]
        if source_case["split"] != "calibration":
            raise ValueError(f"Fuite de split pour {intent_id}")
        contexts = [chunks_by_id[chunk_id] for chunk_id in source_case["gold_chunk_ids"]]
        chunk_id = contexts[0]["chunk_id"]
        for language in LANGUAGES:
            train_count = TRAIN_COUNTS[language]
            validation_count = VALIDATION_COUNTS[language]
            questions = expand_questions(
                spec["questions"][language], language, train_count + validation_count
            )
            answer = make_answer(language, spec["claims"][language], chunk_id)
            for index, question in enumerate(questions):
                split = "train" if index < train_count else "validation"
                records.append(record_for(
                    sample_number=sample_number,
                    intent_id=intent_id,
                    source_case_id=source_case["case_id"],
                    language=language,
                    provider_id=source_case["provider_id"],
                    split=split,
                    question=question,
                    critical_values=list(source_case["critical_values"]),
                    contexts=contexts,
                    answer=answer,
                    should_abstain=False,
                ))
                sample_number += 1

    for intent_id, spec in NEGATIVE_INTENTS.items():
        contexts = [chunks_by_id[spec["distractor_chunk_id"]]]
        for language in LANGUAGES:
            train_count = NEGATIVE_TRAIN_COUNTS[language]
            validation_count = NEGATIVE_VALIDATION_COUNTS[language]
            questions = expand_questions(
                [spec["topic"][language]], language, train_count + validation_count
            )
            answer = make_abstention(language)
            for index, question in enumerate(questions):
                split = "train" if index < train_count else "validation"
                records.append(record_for(
                    sample_number=sample_number,
                    intent_id=intent_id,
                    source_case_id=None,
                    language=language,
                    provider_id=spec["provider_id"],
                    split=split,
                    question=question,
                    critical_values=[],
                    contexts=contexts,
                    answer=answer,
                    should_abstain=True,
                ))
                sample_number += 1

    for record in records:
        validator.validate(record)

    train = [record for record in records if record["split"] == "train"]
    validation = [record for record in records if record["split"] == "validation"]
    test_cases = [
        case
        for case in load_generation_cases(DARIIJA_BENCHMARK, GENERATION_SCHEMA)
        if case["split"] == "test"
    ]
    train_questions = {
        re.sub(r"\s+", " ", record["question"]).casefold()
        for record in train + validation
    }
    for case in test_cases:
        if re.sub(r"\s+", " ", case["question"]).casefold() in train_questions:
            raise ValueError(f"Question de test copiée dans le SFT : {case['case_id']}")

    test_prompts = []
    for case in test_cases:
        contexts = [chunks_by_id[chunk_id] for chunk_id in case["gold_chunk_ids"]]
        test_prompts.append({
            "case_id": case["case_id"],
            "messages": [{
                "role": "user",
                "content": build_generation_prompt(case, contexts, "darija_few_shot"),
            }],
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": OUTPUT_DIR / "train.jsonl",
        "validation": OUTPUT_DIR / "validation.jsonl",
        "test_prompts": OUTPUT_DIR / "test_prompts.jsonl",
        "test_cases": OUTPUT_DIR / "test_cases.jsonl",
    }
    jsonl_write(paths["train"], train)
    jsonl_write(paths["validation"], validation)
    jsonl_write(paths["test_prompts"], test_prompts)
    jsonl_write(paths["test_cases"], test_cases)
    review_seen: set[tuple[str, str]] = set()
    review_queue: list[dict[str, Any]] = []
    for record in records:
        key = (record["intent_id"], record["language"])
        if key in review_seen:
            continue
        review_seen.add(key)
        review_queue.append({
            "sample_id": record["sample_id"],
            "intent_id": record["intent_id"],
            "language": record["language"],
            "provider_id": record["provider_id"],
            "should_abstain": record["should_abstain"],
            "question": record["question"],
            "assistant_answer": json.loads(record["messages"][1]["content"])["answer"],
            "human_decision": None,
            "human_comment": "",
        })
    paths["review_queue"] = OUTPUT_DIR / "review_queue.jsonl"
    jsonl_write(paths["review_queue"], review_queue)

    manifest = {
        "schema_version": "0.1.0",
        "status": "candidate_human_review",
        "base_model": "Qwen/Qwen3.5-4B",
        "prompt_profile": "darija_few_shot",
        "seed": 42,
        "counts": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test_cases),
            "review_queue": len(review_queue),
        },
        "train_language_counts": dict(sorted(Counter(r["language"] for r in train).items())),
        "validation_language_counts": dict(sorted(Counter(r["language"] for r in validation).items())),
        "train_abstention_count": sum(r["should_abstain"] for r in train),
        "validation_abstention_count": sum(r["should_abstain"] for r in validation),
        "test_source": str(DARIIJA_BENCHMARK.relative_to(ROOT)),
        "test_policy": "Le test est exporté pour l'inférence, jamais passé à swift sft.",
        "source_case_ids": sorted({r["source_case_id"] for r in records if r["source_case_id"]}),
        "files": {
            name: {"path": path.name, "sha256": sha256(path)} for name, path in paths.items()
        },
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
