# Mémoire de passation de la conversation

## Origine du projet

Oussama et Meryeme souhaitent apprendre concrètement les LLM, le RAG, les agents, l’IA générative, l’OCR/Document AI, Docker, les tests, CI/CD, LLMOps et le déploiement. Le projet doit être crédible pour des postes ML Engineer, Data Scientist ou AI Engineer, sans ajouter prématurément des technologies uniquement pour enrichir le CV.

Oussama souhaite prendre une part plus importante sur le RAG, les LLM et l’évaluation, tout en comprenant l’architecture complète.

## Vision produit

QraLia (« lis-moi ») vise notamment des adultes marocains ayant des difficultés de lecture. Une personne fournit la photo d’un document et reçoit une extraction fidèle, une explication simple en darija et, lorsque la confiance est insuffisante, un avertissement et la proposition de demander l’aide d’un proche.

La vision future comprend la mémoire documentaire, les rappels, la préparation d’actions et un cercle de confiance. Ces fonctions restent hors du premier MVP.

## Périmètre décidé

Le MVP se limite aux factures d’électricité ONEE : image ou PDF synthétique, OCR/extraction, validation déterministe des montants et dates, petit corpus officiel, réponse simple avec citations. La voix pourra venir ensuite.

Sont différés : multi-agent complexe, GraphRAG, Kubernetes, fine-tuning multiple, DPO et intégration WhatsApp complète.

## Séparation OCR et RAG

- La facture fournit les montants, dates, consommations et identifiants.
- L’OCR ou le VLM extrait ces valeurs avec leur provenance.
- Des règles déterministes protègent les valeurs critiques.
- Le RAG fournit seulement les règles et explications officielles.
- La réponse combine les deux sources en indiquant citations et confiance.

## Travail accompli le 29 septembre 2026

- Inspection du dossier initial, qui était vide hors dossiers techniques.
- Création d’un dépôt Git indépendant sur la branche `main`, sans commit.
- Rédaction du README, de l’architecture MVP, du backlog, de la répartition du binôme et des règles de gouvernance.
- Création de schémas JSON pour registre, document normalisé, chunk et cas d’évaluation.
- Création de 30 questions en français, arabe et darija, avec faits obligatoires, affirmations interdites et cas d’abstention.
- Validation syntaxique et JSON Schema du registre et des 30 cas.
- Vérification en ligne de six pages officielles candidates.
- Copie complète du projet vers `C:\oussama\QraLia` avec dépôt Git intact.

## Résultat de la vérification des sources

- Le guide interactif est pertinent mais son accès direct est instable.
- La FAQ Wakalaty est encore indexée, mais son URL directe renvoie une erreur 404.
- La page sur les factures estimées vise surtout les professionnels MT/BT et ne doit pas être généralisée aux particuliers.
- La page de retard initialement proposée apparaît dans une rubrique collectivités locales ; ses délais ne doivent pas être appliqués aux particuliers sans confirmation.
- La page « Régler votre facture — Résidentiel » est la meilleure source actuelle pour les moyens de paiement.
- La page « Comprendre votre facture » reste candidate sous réserve d’une revue humaine de son audience.

Aucun fichier distant n’a été téléchargé et aucune ingestion n’a eu lieu.

## Prochaine décision

Approuver une liste blanche initiale très courte, probablement la page résidentielle de paiement et la page de compréhension de la facture après revue. Les sources dégradées ou destinées à une autre audience restent en attente ou dans un corpus séparé.

## Règles d’évaluation

Mesurer au minimum Recall@k/Hit Rate, MRR, précision des citations, fidélité, abstention, conservation exacte des montants et dates, latence et coût éventuel. L’objectif d’hallucination sur les montants et dates est 0 %, mais il ne doit être annoncé comme résultat qu’après mesure reproductible.

