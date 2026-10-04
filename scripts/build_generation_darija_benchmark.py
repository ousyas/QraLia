"""Construit le benchmark darija 0.2 à partir des cas factuels déjà validés."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "evaluation" / "generation_questions.v0.1.jsonl"
OUTPUT = ROOT / "data" / "evaluation" / "generation_darija_questions.v0.2.jsonl"

# Deux paraphrases latines et deux paraphrases arabes par intention. Le split
# est hérité du cas source afin qu'une même intention ne soit jamais présente
# à la fois dans calibration et test.
QUESTION_VARIANTS: dict[str, list[tuple[str, str]]] = {
    "generation-001": [
        ("darija_latn", "Chkon khasni nkteb smiyto f chik dyal facture ONEE, w fin ndfe3o?"),
        ("darija_latn", "Bghit nkhlles ONEE b chik: chik bancaire wla CCP, w lfin nmchi bih?"),
        ("darija_arab", "فسمية شكون خاصني نكتب الشيك ديال فاتورة ONEE، وفين نقدمو؟"),
        ("darija_arab", "بغيت نخلّص ONEE بالشيك، واش يقدر يكون بنكي ولا CCP وفين نديه؟"),
    ],
    "generation-002": [
        ("darija_latn", "Fin n9der nkhlles facture ONEE b cash?"),
        ("darija_latn", "Ila bghit nkhelles ONEE b lflous, nmchi l wakala wla kayn chi point akhor?"),
        ("darija_arab", "فين نقدر نخلّص فاتورة ONEE بالكاش؟"),
        ("darija_arab", "إلا بغيت نخلّص ONEE بالفلوس، نمشي للوكالة ولا كاينة شي نقطة أخرى؟"),
    ],
    "generation-003": [
        ("darija_latn", "Ila nhar lkhlass dyal ONEE ja f nhar 3otla, ach kaytra?"),
        ("darija_latn", "Date dyal paiement ONEE tsadfat m3a jour ferie, imta nkhlles?"),
        ("darija_arab", "إلا نهار الخلاص ديال ONEE جا فنهار عطلة، آش كيوقع؟"),
        ("darija_arab", "تاريخ أداء فاتورة ONEE صادف يوم عطلة، إمتى نخلّص؟"),
    ],
    "generation-004": [
        ("darija_latn", "N9der nkhlli avance 3nd ONEE bach y9t3o menha lfactorat?"),
        ("darija_latn", "Kifach kaykhdem l'acompte 3nd ONEE, w imta kay3lmouni belli sold na9s?"),
        ("darija_arab", "نقدر نخلي تسبيق عند ONEE باش يقتاطعو منو الفواتير؟"),
        ("darija_arab", "كيفاش كيخدم التسبيق عند ONEE، وإمتى كيعلموني باللي الرصيد نقص؟"),
    ],
    "generation-005": [
        ("darija_latn", "Ach khasni bach nkhlles Redal online b carte, w ach kanakhod mn b3d?"),
        ("darija_latn", "Site Redal.ma kaykhdem dima? W wach kay3ti reçu w confirmation b mail?"),
        ("darija_arab", "آش خاصني باش نخلّص Redal فالإنترنت بالكارت، وشنو كناخد من بعد؟"),
        ("darija_arab", "واش Redal.ma خدام ديما، وكيعطيني وصل وتأكيد فالإيميل؟"),
    ],
    "generation-006": [
        ("darija_latn", "N9der nkhlles Redal b Orange Money men téléphone?"),
        ("darija_latn", "Service Orange Money dyal Redal kaykhdem 24h/24 w 7j/7?"),
        ("darija_arab", "نقدر نخلّص Redal بـ Orange Money من التليفون؟"),
        ("darija_arab", "واش خدمة Orange Money ديال Redal خدامة 24h/24 و7j/7؟"),
    ],
    "generation-007": [
        ("darija_latn", "Ila agent Redal mal9ach compteur, kifach nsift lih relevé?"),
        ("darija_latn", "Ach nkteb men compteur w fin nsift auto-relevé dyal Redal?"),
        ("darija_arab", "إلا موظف Redal ما لقاوش العداد، كيفاش نصيفط ليهم القراءة؟"),
        ("darija_arab", "شنو نكتب من العداد وفين نصيفط القراءة الذاتية ديال Redal؟"),
    ],
    "generation-008": [
        ("darija_latn", "Facture Redal katji kol chher? W ach katjme3 fiha?"),
        ("darija_latn", "Redal katseft facture chheriya dyal chno: lma, do wla tathir?"),
        ("darija_arab", "واش فاتورة Redal كتجي كل شهر، وشنو كتجمع فيها؟"),
        ("darija_arab", "فاتورة Redal الشهرية فيها الما والضو والتطهير كاملين؟"),
    ],
    "generation-009": [
        ("darija_latn", "Ma wslatnich facture IAM, wach ntsenna 7ta twsel bach nkhlles?"),
        ("darija_latn", "Ila ma jatnich facture papier dyal IAM, wach mazal khasni nkhlles f date?"),
        ("darija_arab", "ما وصلاتنيش فاتورة IAM، واش نتسنى حتى توصل باش نخلّص؟"),
        ("darija_arab", "إلا ما جاتنيش الفاتورة الورقية ديال IAM، واش خاصني نخلّص فالوقت؟"),
    ],
    "generation-010": [
        ("darija_latn", "Ch7al 3ndi men nhar bach n3tared 3la prélèvement IAM 3nd banque?"),
        ("darija_latn", "Banque 9t3at lia montant IAM, f achmen délai n9der ncontestih?"),
        ("darija_arab", "شحال عندي من نهار باش نعترض على اقتطاع IAM عند البنك؟"),
        ("darija_arab", "البنك اقتاطع ليا مبلغ IAM، فاش من أجل نقدر نحتج عليه؟"),
    ],
    "generation-011": [
        ("darija_latn", "IAM wa9fat lia service 7it ma khllestch, ch7al 3ndi bach nsawi lwad3iya?"),
        ("darija_latn", "Mn b3d suspension dyal IAM, ch7al howa délai bach nrj3 service?"),
        ("darija_arab", "IAM وقفات ليا الخدمة حيث ما خلّصتش، شحال عندي باش نسوي الوضعية؟"),
        ("darija_arab", "من بعد ما توقفات خدمة IAM، شحال هو الأجل باش نرجعها؟"),
    ],
    "generation-012": [
        ("darija_latn", "Kifach nkhlles facture Orange b recharge, w fin n3ayet bach n2ekked?"),
        ("darija_latn", "Bghit nkhlles Orange b recharge, ch7al n3emmer w achmen numéro n3ayet lih?"),
        ("darija_arab", "كيفاش نخلّص فاتورة Orange بالشارج، وفين نعيط باش نأكد؟"),
        ("darija_arab", "بغيت نخلّص Orange بالتعبئة، شحال نعمّر وشنو هو الرقم اللي نعيط ليه؟"),
    ],
    "generation-013": [
        ("darija_latn", "Fach kat7ell w katsed wakalat inwi bach nkhlles facture?"),
        ("darija_latn", "Mn ltnin lsebt, chno homa swaye3 lkhedma dyal agences inwi?"),
        ("darija_arab", "فاش كتحل وكتسد وكالات inwi باش نخلّص الفاتورة؟"),
        ("darija_arab", "من الاثنين للسبت، شنو هما سوايع الخدمة ديال وكالات inwi؟"),
    ],
    "generation-014": [
        ("darija_latn", "Kifach nkhlles facture b inwi money étape b étape?"),
        ("darija_latn", "F inwi money, ach nkhtar w ach ndkhel bach nkhlles numéro inwi?"),
        ("darija_arab", "كيفاش نخلّص الفاتورة بـ inwi money مرحلة بمرحلة؟"),
        ("darija_arab", "فـ inwi money، شنو نختار وشنو ندخل باش نخلّص رقم inwi؟"),
    ],
    "generation-015": [
        ("darija_latn", "9bel ma nemchi l ANRT, fin khasni nsift chikaya dyal facture telecom?"),
        ("darija_latn", "Bghit nechteki 3la facture telecom: lmen ntwjeh lwl w kifach nsift reclamation?"),
        ("darija_arab", "قبل ما نمشي لـ ANRT، فين خاصني نصيفط شكاية ديال فاتورة الاتصالات؟"),
        ("darija_arab", "بغيت نشتكي على فاتورة الاتصالات، لمن نتوجه اللول وكيفاش نصيفط الشكاية؟"),
    ],
    "generation-016": [
        ("darija_latn", "N9der nkhlles facture SRM-FM f wakala Redal b nafs chorout?"),
        ("darija_latn", "Wach tari9a dyal Redal katssel7 bach nkhlles SRM-FM?"),
        ("darija_arab", "نقدر نخلّص فاتورة SRM-FM فوكالة Redal بنفس الشروط؟"),
        ("darija_arab", "واش طريقة Redal كتصلح باش نخلّص SRM-FM؟"),
    ],
}

# Quatre intentions hors corpus, chacune déclinée dans les deux écritures.
# Deux intentions vont en calibration et deux restent totalement invisibles
# jusqu'au test. Avec les quatre abstentions issues de generation-016, le jeu
# final contient 20 abstentions sur 80 cas (25 %).
HARD_NEGATIVE_GROUPS = [
    {
        "provider_id": "onee",
        "split": "calibration",
        "forbidden_claims": [
            "Le corpus précise le montant exact d'une pénalité de retard ONEE."
        ],
        "variants": [
            ("darija_latn", "Ch7al bddabt pénalité ila t2ekhert 5 iyam 3la facture ONEE?"),
            ("darija_latn", "ONEE ch7al katzid 3la facture ila khllest mn b3d délai?"),
            ("darija_arab", "شحال بالضبط الغرامة إلا تأخرت 5 أيام على فاتورة ONEE؟"),
            ("darija_arab", "ONEE شحال كتزيد على الفاتورة إلا خلّصت من بعد الأجل؟"),
        ],
    },
    {
        "provider_id": "redal",
        "split": "calibration",
        "forbidden_claims": [
            "Le corpus donne un délai garanti de rétablissement après une panne Redal."
        ],
        "variants": [
            ("darija_latn", "Redal ch7al katakhod bach trje3 do mn b3d panne?"),
            ("darija_latn", "Kayna chi garantie belli Redal trje3 lkahraba f sa3tin?"),
            ("darija_arab", "Redal شحال كتاخد باش ترجع الضو من بعد العطب؟"),
            ("darija_arab", "واش كاينة شي ضمانة باللي Redal ترجع الكهربا فساعتين؟"),
        ],
    },
    {
        "provider_id": "orange",
        "split": "test",
        "forbidden_claims": [
            "Le corpus précise les frais d'une carte bancaire internationale chez Orange."
        ],
        "variants": [
            ("darija_latn", "Ch7al frais ila khllest Orange b carte bancaire internationale?"),
            ("darija_latn", "Orange katakhod ch7al 3la paiement b carte men barra?"),
            ("darija_arab", "شحال المصاريف إلا خلّصت Orange بكارت بنكية دولية؟"),
            ("darija_arab", "Orange كتاخد شحال على الأداء بكارت من برا المغرب؟"),
        ],
    },
    {
        "provider_id": "inwi",
        "split": "test",
        "forbidden_claims": [
            "Le corpus précise les frais exacts de résiliation anticipée inwi."
        ],
        "variants": [
            ("darija_latn", "Ch7al nkhlles ila fssekht contrat inwi 9bel lwa9t?"),
            ("darija_latn", "inwi 3ndha ch7al dyal frais bach n7bes abonnement daba?"),
            ("darija_arab", "شحال نخلّص إلا فسخت عقد inwi قبل الوقت؟"),
            ("darija_arab", "inwi عندها شحال ديال المصاريف باش نحبس الاشتراك دابا؟"),
        ],
    },
]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    source_cases = load_jsonl(SOURCE)
    by_id = {case["case_id"]: case for case in source_cases}
    if set(by_id) != set(QUESTION_VARIANTS):
        raise ValueError("Les intentions sources et les variantes ne correspondent pas.")

    generated: list[dict] = []
    next_id = 201
    for source_case in source_cases:
        variants = QUESTION_VARIANTS[source_case["case_id"]]
        if len(variants) != 4:
            raise ValueError(f"Quatre variantes attendues pour {source_case['case_id']}")
        for language, question in variants:
            case = deepcopy(source_case)
            case["case_id"] = f"generation-{next_id:03d}"
            case["question"] = question
            case["language"] = language
            case["notes"] = (
                f"Variante de {source_case['case_id']}; question candidate à relire "
                "humainement avant de considérer le benchmark comme gelé."
            )
            generated.append(case)
            next_id += 1

    for group in HARD_NEGATIVE_GROUPS:
        for language, question in group["variants"]:
            generated.append({
                "schema_version": "0.1.0",
                "case_id": f"generation-{next_id:03d}",
                "question": question,
                "language": language,
                "provider_id": group["provider_id"],
                "gold_chunk_ids": [],
                "should_abstain": True,
                "required_facts": [],
                "forbidden_claims": group["forbidden_claims"],
                "critical_values": [],
                "split": group["split"],
                "notes": "Hard negative darija candidat à relire humainement.",
            })
            next_id += 1

    if len(generated) != 80:
        raise ValueError(f"80 cas attendus, reçu {len(generated)}")
    OUTPUT.write_text(
        "".join(json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n" for case in generated),
        encoding="utf-8",
    )
    print(f"{len(generated)} cas écrits dans {OUTPUT}")


if __name__ == "__main__":
    main()
