# SFT QLoRA de Qwen3.5-4B sur Kaggle

## Statut

Le dataset `qralia_sft_v0.2` est **rejeté** : ses intentions et ses
quasi-paraphrases traversaient les splits. Il ne doit pas être téléversé sur
Kaggle. La banque `data/evaluation/sft_questions.v0.3.jsonl` est le candidat
courant à la revue humaine. Elle ne contient pas encore les cibles assistant :
le notebook reste donc volontairement inutilisable jusqu'à la construction du
dataset SFT final après revue.

## Banque de questions v0.3

La v0.3 contient 124 questions, soit 31 intentions traduites dans les quatre
langues/écritures. Elle couvre 25 chunks positifs distincts et six hard
negatives, soit 19,35 % d'abstentions. Les 84 questions de train, 20 de
validation et 20 de test sont séparées par `intent_id`. Toutes les langues et
toutes les futures paraphrases d'une intention doivent rester dans son split.

Les chunks positifs sont eux aussi exclusifs à un split. Un contrôle de
similarité bloque les quasi-doublons inter-splits à partir de 0,85. La banque
ne contient que des entrées `typed` : aucune suppression artificielle de
lettres n'est présentée comme une sortie ASR.

Deux notions sont séparées :

- `answerable_in_given_context`, pour juger le générateur sur les passages
  effectivement fournis ;
- `answerable_in_corpus`, pour juger le pipeline RAG complet.

Construction et validation :

```powershell
.venv\Scripts\python.exe scripts/build_sft_questions_v03.py
.venv\Scripts\python.exe -m pytest tests/test_sft_questions_v03.py -q
```

## Enrichissement candidat v0.4

La v0.4 multiplie chaque couple intention-langue en cinq formulations :
directe, indirecte, contextuelle longue, fournisseur implicite et message
familier saisi. Elle contient 620 questions : 360 train, 100 validation et 160
test, avec 18, 5 et 8 intentions exclusives. Les quatre langues comptent 155
questions chacune.

Les formulations supplémentaires ont été proposées sous schéma structuré par
Gemini 2.5 Flash puis Gemini 3.5 Flash Lite après épuisement du quota quotidien
du premier modèle. Les caches conservent cette provenance. Toutes les sorties
restent `candidate_human_review`.

Chaque contexte contient quatre chunks. Les cas positifs reçoivent le ou les
gold chunks et des distracteurs proches ; les abstentions reçoivent des
passages incomplets. La répartition mesurée est :

- 24,52 % d'abstentions ;
- 5,16 % de retrieval misses ;
- 10,32 % de questions demandant deux chunks ;
- 20 % de questions contextuelles longues ;
- 20 % de fournisseurs implicites fournis par la facture synthétique.

Le fichier `sft_questions.v0.4.review.jsonl` regroupe les 620 questions en 124
groupes intention-langue de cinq formulations.

```powershell
.venv\Scripts\python.exe scripts/build_sft_questions_v04.py
.venv\Scripts\python.exe -m pytest tests/test_sft_questions_v04.py -q
```

## Ancienne v0.2 rejetée

| Ensemble | Nombre | Usage |
|---|---:|---|
| Questions totales | 496 | banque linguistique auditée |
| Train | 280 | calcul des gradients |
| Validation | 72 | suivi de la loss et choix des réglages |
| Test gelé | 144 | une seule évaluation après gel des réglages |

Les 496 questions sont équilibrées : 124 par langue/écriture. Elles couvrent
huit intentions positives et dix familles d'abstention. Les splits sont faits
par type de formulation et leurs textes sont disjoints. Le test contient 36
cas par langue et ne contient aucune réponse assistant.

La v0.2 remplace les combinaisons de préfixes/suffixes de la v0.1 par des
formulations naturelles, indirectes, contextuelles, avec alias ou fournisseur
implicite, ainsi que du bruit ASR/WhatsApp léger. Elle contient 185 questions
sans nom explicite du fournisseur et 56 questions positives avec une valeur
numérique. Aucun négatif ne dépend des marqueurs `exact`, `bddabt` ou
`بالضبط`.

## Construction locale

```powershell
.venv\Scripts\python.exe scripts/build_sft_dataset_v02.py
```

Les fichiers produits dans `data/training/qralia_sft_v0.2/` sont :

- `train.jsonl` et `validation.jsonl`, avec métadonnées d'audit et messages ;
- `test_prompts.jsonl`, sans réponse assistant ;
- `test_cases.jsonl`, contenant la grille de vérification ;
- `questions.jsonl`, la banque complète avec style et split ;
- `review_queue.jsonl`, 72 groupes exposant les 496 formulations ;
- `manifest.json`, avec effectifs, statut et SHA-256.

Chaque cible assistant est un `GroundedAnswer` JSON. Les citations sont liées
aux chunks officiels, les valeurs critiques restent littérales et une
abstention ne contient aucun claim. Les abstentions d'entraînement reçoivent
un passage officiel distracteur du bon fournisseur : elles apprennent donc à
refuser une information absente d'un contexte plausible, et non seulement à
réagir à un contexte vide.

## Dataset de cibles v0.4

Le builder `scripts/build_sft_dataset_v04.py` a construit les cibles assistant
à partir de 28 chunks. Il génère une seule fiche factuelle multilingue par
chunk, puis assemble les réponses, citations et abstentions de façon
déterministe. Les sorties sont :

- 360 exemples de train et 100 de validation avec réponse structurée ;
- 160 cas et prompts de test sans aucune réponse assistant ;
- 168 groupes de revue des réponses ;
- un manifeste avec les SHA-256 de chaque fichier.

Les affirmations proposées par Gemini 3.5 Flash Lite ont été soumises à une
revue humaine. Les tests vérifient aussi les citations, la conservation des
valeurs, les alphabets, les retrieval misses et l'absence de réponse dans le
test.

Oussama et Meryeme ont accepté les deux files : les 124 groupes de questions
dans `data/evaluation/sft_questions.v0.4.review.jsonl`, puis les 168 groupes de
réponses dans `data/training/qralia_sft_v0.4/review_queue.jsonl` :

1. naturalité de la darija latine et arabe ;
2. absence d'arabe dans les exemples `darija_latn` ;
3. conservation exacte des nombres, horaires, noms et identifiants ;
4. complétude par rapport au passage ;
5. abstention correcte lorsque le passage plausible ne répond pas à la question ;
6. réalisme des alias, fautes légères et formulations implicites.

Après correction dans les builders et reconstruction, chaque groupe accepté
reçoit `human_decision: "accept"`. Le script suivant ne change le statut que
si tous les groupes de réponses sont acceptés :

```powershell
.venv\Scripts\python.exe scripts/finalize_sft_review.py --reviewer Oussama --reviewer Meryeme
```

Le test reste sans réponse assistant dans le notebook.

## Exécution Kaggle — autorisée après revue

Ne pas lancer le notebook avec la v0.2. Le dataset SFT v0.4 porte maintenant
le statut `human_reviewed` et peut être utilisé avec la procédure suivante :

1. Créer un Dataset Kaggle privé avec le contenu relu de
   `data/training/qralia_sft_v0.4/`.
2. Créer un Notebook Kaggle et activer un GPU.
3. Ajouter le Dataset privé comme Input.
4. Importer directement `kaggle/qralia_qwen35_qlora.ipynb`, ou copier le
   script source `kaggle/qralia_qwen35_qlora.py`.
5. Sur la session Tesla T4 testée, commencer par un smoke test d'un pas à
   `MAX_LENGTH = 1024`, puis revenir au réglage d'entraînement après validation.
   Une longueur de 512 supprime trop d'exemples de ce dataset avec la stratégie
   `delete` et n'est donc pas un test représentatif.
6. Après entraînement, faire `Save Version` et télécharger
   `qralia-qwen35-lora.zip` depuis les outputs.

Le script utilise MS-SWIFT, QLoRA bitsandbytes NF4, FP16, un rang LoRA de 8,
un batch physique de 1, une accumulation de 16 et deux époques. Le ViT et
l'aligneur sont gelés : QraLia entraîne uniquement le comportement textuel.

### Stabilité sur Tesla T4

Le premier essai chargeait `flash-linear-attention` et `causal-conv1d`, puis
restait à `Train: 0/N` avant que le noyau Kaggle ne redémarre. Le journal a
montré que le modèle, le dataset et l'adaptateur LoRA étaient déjà construits :
le blocage se situait donc au premier calcul des couches GatedDeltaNet.

La configuration T4 désactive désormais ces kernels optionnels et impose
`--attn_impl eager`. Le chemin PyTorch est moins rapide, mais transforme un
plantage silencieux en exécution stable ou en erreur Python traçable. Le
processus MS-SWIFT écrit aussi dans `training.log` pendant que le notebook
affiche un battement compact toutes les vingt secondes ; cela évite de saturer
le canal de sortie Jupyter avec la représentation complète du modèle.

`USE_FLA_KERNELS` reste explicite dans le script et dans le manifeste de run.
Il pourra être activé lors d'un essai séparé sur un GPU récent, sans modifier
silencieusement la configuration de référence T4.

Le test n'est pas évalué dans ce notebook d'entraînement. Après validation de
la loss et gel des hyperparamètres, l'adaptateur est récupéré dans le dépôt et
comparé localement au même Qwen3.5 de base sur les 160 cas de test.
