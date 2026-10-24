"""Construit le holdout conversationnel candidat v0.2 de QraLia.

Les formulations sont écrites explicitement ci-dessous afin que la provenance
reste auditable. Elles sont synthétiques et doivent être relues par des
locuteurs avant de recevoir le statut ``human_reviewed``.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/evaluation/conversation_followups.v0.2.jsonl"
MANIFEST = ROOT / "data/evaluation/conversation_followups.v0.2.manifest.json"
LANGUAGES = ("fr", "ar", "darija_arab", "darija_latn")


def variants(fr, ar, darija_arab, darija_latn):
    return {"fr": fr, "ar": ar, "darija_arab": darija_arab, "darija_latn": darija_latn}


# Chaque variante vaut (historique utilisateur, question courante).
SCENARIOS: list[dict[str, Any]] = [
    {"id":"standalone_cheque","action":"standalone","concepts":["cheque"],"dimensions":["how"],"v":variants(
        ([],"Comment je peux régler cette facture par chèque ?"),([],"كيف يمكنني أداء هذه الفاتورة بالشيك؟"),([],"كيفاش نقدر نخلص هاد الفاتورة بالشيك؟"),([],"kifach n9der nkhlles had factora b chik"))},
    {"id":"standalone_amount","action":"standalone","concepts":["amount"],"dimensions":["how_much"],"v":variants(
        ([],"Je dois payer combien au total ?"),([],"كم هو المبلغ الإجمالي الذي يجب أداؤه؟"),([],"شحال خاصني نخلص فالمجموع؟"),([],"ch7al khasni nkhlles total"))},
    {"id":"standalone_due_date","action":"standalone","concepts":["due_date"],"dimensions":["when"],"v":variants(
        ([],"C'est quand la date limite de cette facture ?"),([],"متى يحل آخر أجل لأداء هذه الفاتورة؟"),([],"فوقاش آخر نهار نخلص فيه هاد الفاتورة؟"),([],"fo9ach akher nhar bach nkhlles had factura"))},
    {"id":"standalone_complaint","action":"standalone","concepts":["complaint"],"dimensions":["where"],"v":variants(
        ([],"Où faire une réclamation si la facture est incorrecte ?"),([],"أين أقدم شكوى إذا كانت الفاتورة غير صحيحة؟"),([],"فين نمشي نشكي إلا كانت الفاتورة غالطة؟"),([],"fin nmchi ndir chikaya ila factora ghalta"))},
    {"id":"pronoun_cheque_location","action":"rewrite","concepts":["cheque"],"dimensions":["where"],"v":variants(
        (["Je veux payer cette facture par chèque."],"Et je le dépose où ?"),(["أريد أداء هذه الفاتورة بالشيك."],"وأين أقدمه؟"),(["بغيت نخلص هاد الفاتورة بالشيك."],"وفين نقدمو؟"),(["bghit nkhlles had factora b chik"],"fin n3tih"))},
    {"id":"option_virement","action":"rewrite","concepts":["virement"],"dimensions":["how"],"v":variants(
        (["Quels moyens de paiement sont possibles ?"],"Et par virement, comment ça marche ?"),(["ما وسائل الأداء المتاحة؟"],"وكيف يتم الأداء بالتحويل؟"),(["شنو طرق الخلاص لي كاينة؟"],"وكيفاش ندير بالتحويل؟"),(["chno toro9 lkhlass li kaynin"],"w b virman kifach ndir"))},
    {"id":"option_cash","action":"rewrite","concepts":["cash"],"dimensions":["how"],"v":variants(
        (["Je peux régler cette facture de plusieurs façons."],"Et en espèces, comment je fais ?"),(["يمكنني أداء هذه الفاتورة بعدة طرق."],"وكيف أدفع نقدا؟"),(["نقدر نخلص هاد الفاتورة بكذا طريقة."],"وكيفاش نخلص بالكاش؟"),(["n9der nkhlles had factura b ktr mn tari9a"],"w cash kifach ndir"))},
    {"id":"ellipsis_due_date","action":"rewrite","concepts":["due_date"],"dimensions":["when"],"v":variants(
        (["Le montant total est affiché sur la facture."],"Et la date limite, c'est quand ?"),(["المبلغ الإجمالي موجود في الفاتورة."],"ومتى يحل آخر أجل؟"),(["المبلغ الكامل باين فالفاتورة."],"وآخر نهار فوقاش؟"),(["total kayn f factora"],"w akher nhar fo9ach"))},
    {"id":"ellipsis_amount","action":"rewrite","concepts":["amount"],"dimensions":["how_much"],"v":variants(
        (["Je vois bien la date d'échéance."],"Et le montant, je paie combien ?"),(["أرى تاريخ الاستحقاق."],"وكم هو المبلغ الذي سأدفعه؟"),(["شفت تاريخ الخلاص."],"والمبلغ شحال خاصني نخلص؟"),(["cheft date dyal lkhlass"],"w montant ch7al nkhlles"))},
    {"id":"ellipsis_invoice_number","action":"rewrite","concepts":["invoice_number"],"dimensions":["which"],"v":variants(
        (["J'ai retrouvé le numéro du contrat."],"Et le numéro de facture, c'est lequel ?"),(["وجدت رقم العقد."],"وما رقم الفاتورة؟"),(["لقيت نمرة الكونطرا."],"ونمرة الفاتورة شنو هي؟"),(["l9it numero dyal contrat"],"w num facture chno howa"))},
    {"id":"ellipsis_contract_number","action":"rewrite","concepts":["contract_number"],"dimensions":["which"],"v":variants(
        (["J'ai le numéro de facture."],"Et celui du contrat ?"),(["لدي رقم الفاتورة."],"وماذا عن رقم العقد؟"),(["عندي نمرة الفاتورة."],"ونمرة الكونطرا؟"),(["3ndi num facture"],"w numero contrat"))},
    {"id":"complaint_followup","action":"rewrite","concepts":["complaint"],"dimensions":["where"],"v":variants(
        (["Le montant me semble incorrect."],"D'accord, je réclame où ?"),(["يبدو لي المبلغ غير صحيح."],"حسنا، أين أقدم الشكوى؟"),(["بان ليا المبلغ غالط."],"واخا فين نمشي نشكي؟"),(["ban lia montant ghalet"],"wakha fin ndir chikaya"))},
    {"id":"ambiguous_two_methods","action":"clarify","concepts":[],"dimensions":[],"v":variants(
        (["On a parlé du chèque et du prélèvement."],"Et pour ça, je fais quoi ?"),(["تحدثنا عن الشيك والاقتطاع."],"وماذا أفعل بشأنه؟"),(["هضرنا على الشيك والاقتطاع."],"وداكشي كيفاش ندير ليه؟"),(["hdrna 3la chik w prelevement"],"w hadak kifach ndir lih"))},
    {"id":"no_antecedent","action":"clarify","concepts":[],"dimensions":[],"v":variants(
        (["J'ai une autre question."],"Et lui ?"),(["لدي سؤال آخر."],"وماذا عنه؟"),(["عندي سؤال آخر."],"وداك؟"),(["3ndi so2al akhor"],"w howa"))},
    {"id":"alternative_cash_after_cheque","action":"rewrite","concepts":["cash"],"dimensions":["how"],"v":variants(
        (["Comment payer par chèque ?"],"Sinon en espèces je fais comment ?"),(["كيف أؤدي بالشيك؟"],"وإذا دفعت نقدا فكيف أفعل؟"),(["كيفاش نخلص بالشيك؟"],"ولا بالكاش كيفاش ندير؟"),(["kifach nkhlles b chik"],"ola cash kifach ndir"))},
    {"id":"agency_followup","action":"rewrite","concepts":["agency"],"dimensions":["where"],"v":variants(
        (["Je préfère payer en personne."],"Dans quelle agence je peux aller ?"),(["أفضل الأداء بشكل مباشر."],"إلى أي وكالة يمكنني الذهاب؟"),(["بغيت نخلص بيدي."],"لأي وكالة نقدر نمشي؟"),(["bghit nkhlles b yeddi"],"l anahi wakala n9der nmchi"))},
    {"id":"correction_cash_amount","action":"rewrite","concepts":["cash","amount"],"dimensions":["how_much"],"v":variants(
        (["Le total par chèque est-il affiché ?"],"Non, je parle du montant en espèces : combien ?"),(["هل المبلغ المؤدى بالشيك ظاهر؟"],"لا، أقصد المبلغ نقدا، كم هو؟"),(["واش total ديال الشيك باين؟"],"لا كنقصد المبلغ بالكاش شحال؟"),(["wach total b chik bayn"],"la kan9sed montant cash ch7al"))},
    {"id":"explicit_topic_switch","action":"standalone","concepts":["contract_number"],"dimensions":["which"],"v":variants(
        (["Comment payer par virement ?"],"Maintenant, quel est le numéro de contrat sur ma facture ?"),(["كيف أؤدي بالتحويل؟"],"الآن ما رقم العقد الموجود في فاتورتي؟"),(["كيفاش نخلص بالتحويل؟"],"دابا نمرة الكونطرا لي فالفاتورة شنو هي؟"),(["kifach nkhlles b virman"],"daba chno howa numero contrat f factora"))},
    {"id":"ambiguous_object_location","action":"clarify","concepts":[],"dimensions":["where"],"v":variants(
        (["J'ai le chèque et le formulaire de prélèvement."],"Où est-ce que je le dépose ?"),(["لدي الشيك وطلب الاقتطاع."],"أين أقدمه؟"),(["عندي الشيك وطلب الاقتطاع."],"فين نعطيه؟"),(["3ndi chik w demande prelevement"],"fin n3tih"))},
    {"id":"agency_saturday","action":"rewrite","concepts":["agency"],"dimensions":["when"],"v":variants(
        (["Je cherche les horaires de l'agence."],"Et le samedi, elle ferme quand ?"),(["أبحث عن أوقات عمل الوكالة."],"ومتى تغلق يوم السبت؟"),(["كنقلب على وقت الخدمة ديال الوكالة."],"والسبت فوقاش كتسد؟"),(["kan9leb 3la wa9t lkhedma dyal wakala"],"w sebt fo9ach katsed"))},
]


def build() -> list[dict[str, Any]]:
    rows = []
    for scenario in SCENARIOS:
        for language in LANGUAGES:
            history, question = scenario["v"][language]
            rows.append({
                "schema_version": "0.2.0",
                "case_id": f"holdout-{scenario['id']}-{language}",
                "split": "holdout",
                "review_status": "candidate_human_review",
                "source_type": "synthetic_conversation_candidate",
                "scenario_family": scenario["id"],
                "language": language,
                "history": history,
                "question": question,
                "expected_action": scenario["action"],
                "required_semantic_concepts": scenario["concepts"],
                "expected_dimensions": scenario["dimensions"],
            })
    return rows


def main() -> None:
    rows = build()
    assert len(rows) == 80
    assert Counter(row["language"] for row in rows) == Counter({language: 20 for language in LANGUAGES})
    content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    # Écriture binaire pour figer les fins de ligne LF et obtenir le même hash
    # sous Windows, Linux et dans GitHub Actions.
    payload = content.encode("utf-8")
    OUTPUT.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    manifest = {
        "schema_version": "0.2.0",
        "status": "candidate_human_review",
        "case_count": len(rows),
        "scenario_family_count": len(SCENARIOS),
        "language_counts": dict(Counter(row["language"] for row in rows)),
        "action_counts": dict(Counter(row["expected_action"] for row in rows)),
        "sha256": digest,
        "warning": "Formulations synthétiques candidates ; ne pas présenter comme conversations d'utilisateurs réels.",
    }
    MANIFEST.write_bytes((json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
