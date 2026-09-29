# QraLia

QraLia (« lis-moi ») est un projet pédagogique d’assistant de lecture de documents. Le premier cas d’usage est volontairement étroit : aider à comprendre une facture d’électricité ONEE à partir d’une image ou d’un PDF, puis expliquer les règles utiles à partir de pages officielles citées.

## Objectif du MVP

Le MVP doit pouvoir :

1. recevoir une facture synthétique ONEE ;
2. extraire sans les altérer le montant, la date d’échéance, les identifiants et quelques champs de consommation ;
3. valider ces champs avec des règles déterministes ;
4. retrouver un passage pertinent dans un petit corpus officiel ;
5. répondre simplement, en français, arabe ou darija, avec une citation et un niveau de confiance ;
6. refuser de deviner et recommander l’aide d’un proche lorsque la lecture est incertaine.

L’audio, WhatsApp, la mémoire personnelle, le multi-agent, le fine-tuning et le déploiement cloud sont hors du premier MVP.

## Principe architectural

```text
facture synthétique -> OCR/extraction -> validation déterministe ----+
                                                                 réponse citée
question -----------> recherche hybride -> passage officiel --------+
```

Les faits propres à la facture viennent exclusivement du document utilisateur. Les règles générales viennent exclusivement du corpus officiel. Le modèle de langage ne recalcule pas et ne corrige pas silencieusement les montants, dates ou identifiants.

## État actuel

- fondation documentaire du dépôt ;
- périmètre et architecture MVP définis ;
- registre initial de six sources officielles candidates ;
- format documentaire interne et format des chunks définis ;
- jeu initial de 30 questions d’évaluation en français, arabe et darija ;
- une page résidentielle ONEE approuvée, archivée localement, normalisée et découpée en cinq chunks actifs ;
- une baseline BM25 reproductible évaluée sur un jeu dédié de 35 questions ;
- aucune donnée personnelle utilisée.

## Structure cible

```text
qralia/
├── data/
│   ├── evaluation/       # jeux de test versionnés, sans données réelles
│   ├── sources/          # registre et futurs manifestes des sources publiques
│   ├── raw/              # téléchargements locaux ignorés par Git
│   ├── processed/        # documents normalisés ignorés par Git
│   └── indexes/          # index reconstruisibles ignorés par Git
├── docs/
│   ├── architecture/     # architecture et contrats du MVP
│   ├── decisions/        # décisions techniques courtes et datées
│   └── rag/              # acquisition, corpus et protocole d’évaluation
├── schemas/              # contrats JSON des documents, chunks et cas de test
├── src/qralia/           # code applicatif futur
│   ├── api/
│   ├── ocr/
│   ├── rag/
│   ├── validation/
│   └── domain/
└── tests/                # tests unitaires, intégration et évaluation
```

Les dossiers de code seront ajoutés au moment du premier incrément exécutable, afin de ne pas donner l’illusion d’une implémentation déjà existante.

## Documents de départ

- [Architecture MVP](docs/architecture/mvp.md)
- [Backlog par phases](docs/backlog.md)
- [Répartition du binôme](docs/team.md)
- [Règles de sécurité des données](docs/data-governance.md)
- [Mémoire de passation](docs/conversation-handoff.md)
- [Plan du Sprint RAG ONEE](docs/rag/sprint-onee.md)
- [Baseline BM25](docs/rag/baseline-bm25.md)
- [Registre des sources](data/sources/onee_sources.v0.1.json)
- [Jeu d’évaluation](data/evaluation/onee_questions.v0.1.jsonl)
- [Jeu d’évaluation de la récupération](data/evaluation/onee_retrieval_questions.v0.1.jsonl)

## Prochaine étape contrôlée

Analyser les erreurs de la baseline BM25, puis mesurer une baseline d'embeddings multilingues sur exactement le même corpus et le même jeu de test. Le futur collecteur devra aussi capturer automatiquement le statut HTTP, l’URL finale et le type MIME. Une source dégradée ou destinée à une autre catégorie de clients ne doit pas entrer dans l’index principal.

## Éthique et confidentialité

QraLia ne doit jamais parcourir un dossier personnel. Le développement utilise uniquement des sources publiques officielles et des factures entièrement synthétiques. Les documents réels sont hors périmètre tant que le consentement, le chiffrement, la suppression et la durée de conservation ne sont pas définis et testés.
