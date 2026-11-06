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

## Mise à jour du 1er octobre 2026

- Cinq chunks ONEE et neuf chunks Redal sont maintenant construits avec un
  routage explicite par distributeur, zone et audience.
- BM25 manuel et Qwen3-Embedding-0.6B via LangChain sont évalués séparément.
- Le GPU local GTX 1650 Ti exécute Qwen3 avec PyTorch CUDA.
- Le jeu Redal contient 42 cas et le benchmark d'écriture darija contient 24
  intentions appariées, soit 48 questions.
- Sur les paires de test, Qwen3 obtient un MRR de 0,521 en darija latine et
  0,573 en darija arabe. Les résultats sont mixtes et ne prouvent pas une
  supériorité générale de l'alphabet arabe.
- La prochaine expérience RAG est la fusion hybride de BM25 et Qwen3, suivie
  d'un reranking évalué sur les mêmes jeux gelés.

## Mise à jour télécom du 1er octobre 2026

- Cinq pages ont été acquises directement avec manifeste et SHA-256 ; la page
  IAM protégée par Cloudflare a été archivée comme passages rendus et revus.
- Six documents normalisés produisent exactement 25 chunks actifs : 7 IAM,
  10 Orange, 6 inwi et 2 ANRT.
- Le contrat des documents et chunks passe en 0.3.0 avec un routage générique
  `domain`, `provider_id`, `service_type`, `service_zone_id`, `audiences`.
- Les jeux d'évaluation 0.2.0 existants restent gelés. La prochaine étape est
  de créer les questions télécom avant d'observer les scores des moteurs.

## Évaluation hybride du 1er octobre 2026

- Un jeu télécom 0.3.0 de 56 cas équilibrés entre français, arabe, darija
  latine et darija arabe couvre les 25 chunks et 16 abstentions.
- Le pipeline hybride compare BM25 normalisé, Qwen3 et Weighted RRF en une seule
  exécution, avec routage strict et calibration séparée du test.
- Sur 227 cas globaux et 39 chunks, l'hybride obtient Recall@5 0,973 et Binary
  Recall@5 0,977. BM25 conserve le meilleur MRR (0,785) et la meilleure
  balanced accuracy d'abstention (0,745).
- Les poids 0,5/0,5 sont une baseline gelée ; ne pas les ajuster après lecture
  du test. Toute optimisation future doit sélectionner les poids uniquement
  sur la calibration.

## Évaluation du reranker du 1er octobre 2026

- `Qwen/Qwen3-Reranker-0.6B` est appliqué localement au Top-20 hybride après le
  routage fournisseur/zone/service/audience. Sa révision est figée dans le code.
- Les modèles d'embeddings et de reranking sont chargés successivement afin de
  tenir sur le GPU GTX 1650 Ti de 4 Go. Le reranker utilise FP16 et des lots de
  16 pour l'expérience de référence.
- Le Top-20 contient 100 % des sections pertinentes du test. Sur les 123 cas de
  test, le reranker atteint Recall@5 0,981, Binary Recall@5 0,988 et MRR 0,799.
- Il dépasse les autres systèmes sur ces trois mesures, mais l'hybride garde le
  meilleur Recall@3 (0,900 contre 0,817) et BM25 la meilleure balanced accuracy
  d'abstention (0,745 contre 0,627).
- La latence moyenne amortie du pipeline reranké est d'environ 3,78 s par cas,
  contre 84 ms pour l'hybride. Le rapport complet est
  `outputs/reranker-all-corpora-qwen3.json`.
- Prochaine étape : analyse d'erreurs sur la calibration et conception d'une
  politique d'abstention combinant les signaux sans retoucher le test observé.

## Génération et juge du 1er octobre 2026

- `qwen3:4b-instruct-2507-q4_K_M` génère localement une réponse Pydantic à
  partir des gold chunks ; les citations visibles sont assemblées ensuite par
  un renderer déterministe.
- Les garde-fous vérifient langue déclarée, abstention, IDs de chunks, présence
  des citations, valeurs critiques exactes et affirmations interdites.
- Gemini 2.5 Flash évalue fidélité, pertinence, complétude, citations, langue et
  abstention. Gemini Pro a un quota gratuit nul pour le projet API testé ;
  Flash est configurable et a permis la baseline complète.
- Sur 16 cas multilingues, Qwen passe 62,5 % des contrôles déterministes et
  31,25 % du critère final avec juge. La correction des citations est forte
  (4,75/5), mais la complétude moyenne est 3,13/5 et quatre violations critiques
  sont signalées.
- Une modification post-hoc du prompt a réduit le taux déterministe à 56,25 %
  et n'est pas retenue. Le test actuel ayant été observé, la prochaine
  comparaison de modèle doit utiliser un nouveau holdout gelé.
- Rapports : `outputs/generation-qwen3-4b-gemini-2.5-flash.json` et
  `outputs/generation-qwen3-4b-prompt-v0.2-deterministic-exploratory.json`.

## Comparaison Qwen3.5 et Atlas du 2 octobre 2026

- Un holdout de 12 cas, avec trois cas par langue/écriture, a été gelé avant
  l'exécution de Qwen3.5.
- Le validateur contrôle désormais l'alphabet réellement utilisé après avoir
  retiré les IDs de citation. Une réponse française déclarée `darija_arab`
  n'est donc plus acceptée.
- Avec le même prompt, les mêmes gold chunks et le validateur corrigé,
  Qwen3-4B atteint 41,7 % et Qwen3.5-4B atteint 58,3 %. Qwen3.5 devient le
  candidat local retenu pour la prochaine expérience, sans prétention de
  qualité de production.
- Al-Atlas-0.5B est intégré comme reformulateur optionnel. Les valeurs critiques
  sont verrouillées par marqueurs et les sorties dégradées déclenchent un
  fallback. Sur trois cas darija arabe, deux sorties sont rejetées et le score
  reste à 1/3 ; Atlas demeure donc désactivé par défaut.
- Prochaine étape : revue humaine marocaine, jugement structuré du holdout,
  puis comparaison gold-context contre Top-k récupéré.

## Prompt few-shot darija du 2 octobre 2026

- Le profil `darija_few_shot` injecte un petit lexique de style et un exemple
  non factuel pour chaque écriture darija, sans imposer de template métier.
- Il insiste sur la langue cible même lorsque les chunks sont en français. La
  baseline 0.1.0 reste inchangée via `--prompt-profile baseline`.
- Sur le holdout déjà observé, le score déterministe passe de 7/12 à 8/12 et la
  darija de 3/6 à 4/6. Cette mesure est post-hoc et ne doit pas être présentée
  comme un nouveau test vierge.
- La revue qualitative montre encore une réponse française acceptée comme
  darija latine, une abstention incorrecte et deux littéraux transformés. La
  variante doit être gelée puis testée sur un nouveau holdout humain.

## Smoke test Atlas-Chat-2B du 2 octobre 2026

- Un backend Ollama permet maintenant d'utiliser Atlas-Chat avec le même
  verrouillage des valeurs, contrôle de sortie et fallback qu'Al-Atlas-0.5B.
- Le modèle GGUF Q2_K de 1,2 Go fonctionne sur la GTX 1650 Ti de 4 Go. Son
  score déterministe est 2/3 sur les trois cas darija arabe déjà observés.
- Ce score est trompeur : une réponse mélange français et arabe et déforme la
  règle Redal. Une autre perd la valeur protégée `10` et est rejetée. Atlas-Chat
  Q2 n'est donc pas retenu sur cette expérience.
- Le prochain benchmark de génération doit compter environ 60 à 100 cas, dont
  au moins 40 à 60 cas darija équilibrés entre écritures arabe et latine, avec
  20 à 25 % d'abstentions et une revue humaine indépendante.

## Benchmark darija 0.2 et mode d'observation du 2 octobre 2026

- `generation_darija_questions.v0.2.jsonl` contient maintenant 80 cas
  candidats : 40 par écriture, 40 par split et 20 abstentions (25 %).
- Seule la calibration a été exécutée. Le test de 40 cas reste intact et le jeu
  doit encore être relu humainement avant gel officiel.
- Le mode Atlas d'observation conserve les sorties qui perdent un marqueur ou
  échouent au contrôle de qualité, tout en enregistrant qu'un fallback aurait
  normalement eu lieu. Il est strictement diagnostique.
- Sur 40 cas de calibration, Qwen3.5 seul obtient 33/40. Atlas-Chat-2B Q2 tombe
  à 29/40 avec le prompt long et 28/40 avec le prompt compact. Le compact passe
  de 6,00 s à 1,40 s par reformulation, mais produit davantage de sorties que
  le garde-fou aurait rejetées (10 contre 8).
- Conclusion actuelle : ne pas mettre Atlas dans le chemin de réponse. Les
  chiffres déterministes sont eux-mêmes optimistes, car une revue humaine a
  trouvé des répétitions, fuites de prompt et dérives sémantiques non détectées.
- Les rapports conservent `answer_before_rewrite`, `raw_candidate`,
  `would_fallback` et la réponse observée pour permettre l'audit.

## Dataset SFT et notebook Kaggle du 2 octobre 2026

- Le builder `scripts/build_sft_dataset.py` produit 478 exemples de train, 100
  de validation et exporte les 40 tests gelés sans réponse assistant.
- Le train contient 168 exemples par écriture darija, 71 en français, 71 en
  arabe standard et 110 abstentions. Seules huit intentions de calibration
  fournissent des cibles positives ; aucune intention positive du test n'est
  utilisée pour calculer les gradients.
- Les 52 couples intention-langue sont exposés dans `review_queue.jsonl`. Le
  statut reste `candidate_human_review` tant qu'Oussama et Meryeme n'ont pas
  accepté ces représentants. Toute correction doit être faite dans le builder
  puis reconstruite, afin de conserver la provenance.
- `kaggle/qralia_qwen35_qlora.py` entraîne Qwen3.5-4B avec QLoRA NF4, FP16,
  rang 8, batch 1 et accumulation 16. Il vérifie les SHA-256 et bloque si la
  revue humaine n'est pas finalisée.
- Le test n'est jamais transmis à `swift sft`. Il sera ouvert une seule fois
  après choix des hyperparamètres sur la validation et récupération de
  l'adaptateur.

## Révision SFT v0.2 du 2 octobre 2026

- Le retour humain externe a identifié des questions trop propres, des alias
  absents et un raccourci lexical dans les abstentions de la v0.1.
- `scripts/build_sft_dataset_v02.py` construit désormais 496 questions
  équilibrées entre les quatre langues/écritures : 280 train, 72 validation et
  144 test gelé. La v0.1 est conservée comme trace, mais ne doit plus être
  envoyée à Kaggle.
- Les questions couvrent huit styles auditables, 185 formulations sans nom
  explicite du fournisseur et des alias comme Maroc Telecom, Ittisalat,
  l'office et lma w do. Aucun négatif ne contient les marqueurs `exact`,
  `bddabt` ou `بالضبط`.
- Dix familles de hard negatives utilisent un passage officiel plausible du
  bon fournisseur. Des questions positives contenant des chiffres empêchent
  d'apprendre « nombre demandé = abstention ».
- Les 496 formulations sont exposées dans 72 groupes de revue humaine. Le
  notebook Kaggle cible maintenant `qralia_sft_v0.2` et reste bloqué jusqu'à
  la validation explicite de ces groupes.

## Banque de questions SFT v0.3 du 2 octobre 2026

- Un second audit a montré que la v0.2 restait invalide : mêmes intentions et
  quasi-paraphrases dans plusieurs splits, styles confondus avec les splits et
  faux bruit ASR. Son manifeste porte désormais `rejected_split_leakage`.
- `scripts/build_sft_questions_v03.py` produit une banque préalable de 124
  questions : 31 intentions, quatre langues chacune, 25 chunks positifs
  distincts et six hard negatives. Le taux d'abstention est de 19,35 %.
- Le split est fait par `intent_id`, et chaque chunk positif appartient à un
  seul split. Les tests bloquent aussi les quasi-doublons inter-splits à 0,85.
- Toutes les entrées sont `typed`. Le projet n'utilisera le label ASR qu'après
  transcription de vrais audios synthétiques/non personnels par un moteur
  réellement exécuté.
- `answerable_in_given_context` et `answerable_in_corpus` sont séparés pour ne
  plus confondre l'évaluation du générateur et celle du RAG complet.
- Étape suivante : revue humaine des 124 formulations, ajout de paraphrases
  humaines dans le split déjà attribué, puis seulement création des cibles SFT
  train/validation et autorisation du notebook Kaggle.

## Enrichissement SFT v0.4 du 2 octobre 2026

- `scripts/build_sft_questions_v04.py` enrichit la structure v0.3 en 620
  questions candidates : cinq formulations par couple intention-langue, 360
  train, 100 validation et 160 test, sur 18/5/8 intentions exclusives.
- Gemini 2.5 Flash a produit les 21 premiers groupes avant sa limite gratuite ;
  Gemini 3.5 Flash Lite a produit les dix groupes restants et le lot
  multi-chunks. Les modèles et sorties brutes sont conservés comme provenance.
- Chaque contexte contient quatre chunks. Le gold est absent des 32 retrieval
  misses. Le jeu comprend 152 abstentions (24,52 %), 64 cas multi-chunks
  (10,32 %), 124 questions longues et 124 fournisseurs implicites.
- Les tests contrôlent les splits d'intentions, familles et chunks gold, les
  quasi-doublons, les alphabets, les contextes, les taux cibles et la diversité
  de registre. Les 620 questions restent candidates à revue humaine.
- `sft_questions.v0.4.review.jsonl` expose 124 groupes de cinq formulations.
  Les réponses SFT et l'entraînement Kaggle restent interdits avant validation.

## Cibles SFT v0.4 du 2 octobre 2026

- `scripts/build_sft_dataset_v04.py` génère une fiche factuelle dans quatre
  langues pour chacun des 28 chunks gold, puis assemble déterministiquement
  les citations, les abstentions et les prompts.
- Le dataset contient 360 exemples de train, 100 de validation et 160 prompts
  de test sans réponse assistant. Les 460 réponses sont regroupées en 168
  groupes de revue.
- Les contrôles automatiques vérifient le schéma, les SHA-256, les citations,
  les valeurs critiques, les retrieval misses, l'alphabet darija latin et
  l'étanchéité du test.
- Oussama et Meryeme ont explicitement accepté les 124 groupes de questions et
  les 168 groupes de réponses. Les deux manifestes portent maintenant le
  statut `human_reviewed` et leurs empreintes de revue ont été recalculées.
- Le notebook Kaggle vise `qralia_sft_v0.4`, vérifie les empreintes SHA-256 et
  peut désormais lancer QLoRA.
