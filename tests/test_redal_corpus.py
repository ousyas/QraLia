from __future__ import annotations

from qralia.rag.redal_corpus import extract_faq_sections, extract_payment_sections


def _component(root: str, title_class: str, content_class: str, title: str, text: str) -> str:
    return (
        f'<div class="{root}"><button class="{title_class}"><span>{title}</span></button>'
        f'<div class="{content_class}"><p>{text}</p></div></div>'
    )


def test_payment_parser_builds_five_semantic_sections_without_bank_urls() -> None:
    html = "<html><body>navigation ignorée"
    html += _component(
        "tabs-item", "tabs-item__title", "tabs-item__content", "Via le net",
        "Redal.ma permet de payer par carte. Autres sites : Banque A https://bank.example/login Banque B.",
    )
    html += _component("tabs-item", "tabs-item__title", "tabs-item__content", "JIWAR MOBIL", "Paiement par téléphone.")
    html += _component("tabs-item", "tabs-item__title", "tabs-item__content", "Prélèvement Automatique Bancaire", "Prélèvement auprès des banques.")
    html += _component("tabs-item", "tabs-item__title", "tabs-item__content", "Guichets Automatiques Bancaires", "Paiement au guichet automatique.")
    html += "</body></html>"

    sections = extract_payment_sections(html.encode("utf-8"))

    assert [section[0] for section in sections] == [
        "paiement-en-ligne-carte", "paiement-ebanking", "paiement-mobile-jiwar",
        "paiement-prelevement", "paiement-gab",
    ]
    assert "https://" not in sections[1][2]
    assert "Banque A" in sections[1][2]
    assert "navigation" not in " ".join(section[2] for section in sections)


def test_faq_parser_groups_six_answers_into_four_autonomous_sections() -> None:
    questions = {
        "Comment se passe le relevé des compteurs ?": "Les agents relèvent les compteurs.",
        "Comment faire si mon compteur est inaccessible à la relève par Redal ?": "Le client peut transmettre son index.",
        "Comment lire ma facture ?": "Vérifier le nom, les index et la date limite.",
        "Combien de facture je reçois mensuellement ?": "La facture est mensuelle.",
        "Quel est le délai de paiement ?": "La date limite figure sur la facture.",
        "Comment s’effectue la facturation ?": "La consommation est calculée après la relève.",
    }
    html = "<html><body>" + "".join(
        _component("accordion-item", "accordion-item__title", "accordion-item__content", title, text)
        for title, text in questions.items()
    ) + "</body></html>"

    sections = extract_faq_sections(html.encode("utf-8"))

    assert len(sections) == 4
    by_id = {section_id: text for section_id, _, text in sections}
    assert "transmettre son index" in by_id["releve-et-auto-releve"]
    assert "facture est mensuelle" in by_id["facturation-mensuelle"]
    assert "consommation est calculée" in by_id["facturation-mensuelle"]
