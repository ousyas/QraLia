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

Chaque règle est routée par domaine, fournisseur, type de service, zone et
audience. Ces filtres sont appliqués avant la recherche afin de ne jamais
mélanger des modalités locales ou des secteurs différents, même lorsque les
documents utilisent les mêmes mots.

## État actuel

- fondation documentaire du dépôt ;
- périmètre et architecture MVP définis ;
- registre initial de six sources officielles candidates ;
- format documentaire interne et format des chunks définis ;
- jeu initial de 30 questions d’évaluation en français, arabe et darija ;
- une page résidentielle ONEE approuvée, archivée localement, normalisée et découpée en cinq chunks actifs ;
- deux pages officielles Redal archivées et transformées en neuf chunks actifs,
  routés vers `srm_rsk` / `rabat_sale_ex_redal` ;
- six sources officielles télécom transformées en 25 chunks actifs : sept IAM,
  dix Orange, six inwi et deux ANRT ;
- une baseline BM25 reproductible évaluée sur un jeu dédié de 35 questions ;
- un jeu Redal de 42 cas de récupération, dont 14 demandes d'abstention et
  plusieurs hard negatives appartenant à d'autres distributeurs ;
- un benchmark apparié de 48 questions comparant 24 intentions en darija
  latine et en darija arabe ;
- un jeu de robustesse darija de 46 cas et une normalisation lexicale BM25
  optionnelle, évaluée séparément de la baseline ;
- une recherche sémantique locale Qwen3-Embedding-0.6B orchestrée avec
  LangChain, évaluée sur les mêmes 35 cas (Recall@5 `1.000`, MRR `0.886`) ;
- une recherche hybride BM25 normalisé + Qwen3 par Weighted RRF, évaluée sur
  39 chunks et 227 cas multilingues ; la baseline atteint Recall@5 `0.973`,
  Binary Recall@5 `0.977` et MRR `0.745` ;
- un reranking local Qwen3-Reranker-0.6B du Top-20 hybride ; sur le même test,
  il atteint Recall@5 `0.981`, Binary Recall@5 `0.988` et MRR `0.799`, avec un
  coût de latence nettement supérieur et une abstention encore inférieure à
  BM25 ;
- une baseline de génération sur 16 gold contexts avec Qwen3-4B local, rendu
  déterministe des citations, contrôles de valeurs critiques et juge Gemini
  structuré ; elle obtient 62,5 % au validateur déterministe et 31,25 % au
  critère final, résultat insuffisant mais reproductible qui localise surtout
  des problèmes de complétude, d'abstention et de darija latine ;
- un holdout gelé de 12 cas comparant Qwen3-4B et Qwen3.5-4B avec contrôle
  indépendant de l'écriture : Qwen3.5 atteint 58,3 % contre 41,7 % pour
  Qwen3 ; l'échantillon reste trop petit pour conclure à une qualité de
  production ;
- une intégration expérimentale d'Al-Atlas-0.5B comme reformulateur darija
  protégé par verrouillage des valeurs et retour automatique à la réponse
  originale ; sur trois cas en darija arabe, deux sorties sont rejetées et le
  score reste à 1/3, donc Atlas est désactivé par défaut ;
- une variante de prompt few-shot qui injecte un lexique de style darija sans
  templates métier ; elle atteint 8/12 contre 7/12 sur le holdout déjà observé,
  mais reste exploratoire à cause d'erreurs de langue, d'abstention et de
  conservation des littéraux ;
- un backend Ollama pour Atlas-Chat-2B avec les mêmes valeurs verrouillées et
  fallbacks ; un smoke test Q2_K donne 2/3 automatiquement mais contient une
  dérive sémantique évidente, donc le modèle n'est pas retenu à ce stade ;
- un benchmark darija candidat de 80 cas, équilibré entre écritures et splits,
  avec 25 % d'abstentions ; sur ses 40 cas de calibration, Qwen3.5 seul atteint
  82,5 %, contre 72,5 % après Atlas-2B avec prompt long et 70 % avec prompt
  compact en mode d'observation. Atlas reste désactivé ; le test est intact ;
- la v0.2 du dataset SFT est explicitement rejetée après détection de fuites
  d'intentions et de quasi-paraphrases entre les splits ;
- une banque v0.3 de 124 questions, 31 intentions et 25 chunks positifs est
  séparée par intention : 84 questions train, 20 validation et 20 test, avec
  19,35 % d'abstentions, aucun faux ASR et deux labels distincts pour la
  réponse possible dans le contexte ou dans le corpus ; elle doit être relue
  avant de construire les cibles SFT et d'autoriser Kaggle ;
- une banque enrichie v0.4 contient 620 questions candidates : 360 train, 100
  validation et 160 test, groupées en 18/5/8 intentions. Chaque contexte
  contient quatre chunks ; 5,16 % des cas simulent un retrieval raté, 10,32 %
  nécessitent deux chunks, 20 % sont contextuels longs et 20 % utilisent un
  fournisseur implicite issu d'une facture synthétique. Les paraphrases
  générées restent candidates à revue humaine ;
- les cibles SFT v0.4 sont maintenant assemblées pour les 360 exemples de
  train et 100 de validation à partir de 28 chunks. Le test gelé contient 160
  prompts sans réponse assistant. Oussama et Meryeme ont accepté les 124
  groupes de questions et les 168 groupes de réponses ; le manifeste est
  désormais `human_reviewed` et le notebook Kaggle peut démarrer ;
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
- [Recherche sémantique LangChain + Qwen3](docs/rag/semantic-search-langchain.md)
- [Benchmark darija latine / darija arabe](docs/rag/darija-script-benchmark.md)
- [Construction du corpus Redal](docs/rag/construction-corpus-redal.md)
- [Construction du corpus télécom](docs/rag/construction-corpus-telecom.md)
- [Recherche hybride BM25 + Qwen3](docs/rag/hybrid-search.md)
- [Reranking Qwen3 du Top-N hybride](docs/rag/reranker-search.md)
- [Génération Qwen3 et juge Gemini](docs/rag/generation-and-judge.md)
- [SFT QLoRA Qwen3.5 sur Kaggle](docs/rag/sft-qwen35-kaggle.md)
- [Registre des sources](data/sources/onee_sources.v0.1.json)
- [Registre des sources Redal](data/sources/redal_sources.v0.1.json)
- [Jeu d’évaluation](data/evaluation/onee_questions.v0.1.jsonl)
- [Jeu d’évaluation de la récupération](data/evaluation/onee_retrieval_questions.v0.1.jsonl)
- [Jeu d’évaluation Redal](data/evaluation/redal_retrieval_questions.v0.1.jsonl)

## Prochaine étape contrôlée

Faire relire manuellement les sorties arabes et darija du holdout, puis évaluer
Qwen3.5 avec le juge structuré sans ajuster le prompt sur ces douze cas. Ensuite,
comparer la génération sur gold chunks à la génération alimentée par le Top-k
réellement récupéré. En parallèle, analyser les erreurs du reranker sur la
calibration et conserver BM25 comme signal d'abstention. Une source dégradée ou
destinée à une autre catégorie de clients ne doit pas entrer dans l'index
principal.

## Éthique et confidentialité

QraLia ne doit jamais parcourir un dossier personnel. Le développement utilise uniquement des sources publiques officielles et des factures entièrement synthétiques. Les documents réels sont hors périmètre tant que le consentement, le chiffrement, la suppression et la durée de conservation ne sont pas définis et testés.
