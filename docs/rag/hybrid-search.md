# Recherche hybride BM25 + Qwen3

## Objectif expérimental

Le pipeline compare trois systèmes sur exactement les mêmes chunks, questions
et routes :

1. BM25 avec la normalisation déterministe `darija_lexicon` ;
2. Qwen3-Embedding-0.6B avec similarité cosinus via LangChain ;
3. fusion hybride des deux classements.

Le jeu télécom a été écrit et validé avant la première exécution. Il contient
56 cas : 14 en français, 14 en arabe classique, 14 en darija latine et 14 en
darija arabe. Les 40 cas positifs couvrent les 25 chunks télécom et les 16 cas
négatifs vérifient notamment les mélanges entre opérateurs.

## Routage avant scoring

Les candidats sont filtrés avant BM25 et Qwen par :

```text
domain + provider_id + service_type + service_zone_id + audience
```

Les anciens cas `0.2.0` sont projetés en mémoire vers ce contrat. Les fichiers
historiques ne sont pas réécrits. Les routes négatives SRM qui ne disposent
d'aucun chunk actif restent valides et doivent conduire à l'abstention.

## Fusion Weighted RRF

Les scores BM25 et cosinus ne sont pas directement additionnés. Le rang d'un
chunk dans chaque branche est transformé avec :

```text
score_RRF(chunk) = w_bm25 / (k + rang_bm25)
                 + w_qwen / (k + rang_qwen)
```

La baseline utilise `w_bm25 = 0.5`, `w_qwen = 0.5` et `k = 60`, fixés avant
l'observation du jeu test. Un chunk trouvé par les deux moteurs est favorisé ;
un chunk trouvé par une seule branche peut tout de même entrer dans le Top-k.

Le score RRF ordonne les résultats. Une confiance bornée distincte combine le
score BM25 transformé et le cosinus Qwen pour calibrer un seuil d'abstention
sur le split `calibration` uniquement. Cette confiance n'est pas une
probabilité.

## Protocole global

Le rapport actuel indexe 39 chunks actifs et regroupe 227 cas : jeux ONEE,
robustesse darija, Redal, paires darija et télécom. Il calcule Recall@1/3/5,
Binary Recall@1/3/5, MRR, métriques d'abstention, latence et résultats groupés
par langue, catégorie, domaine et fournisseur.

Commande reproductible :

```powershell
$env:PYTHONPATH = "src"
\.venv\Scripts\python.exe -m qralia.rag.hybrid_evaluation `
  --chunks data/processed/onee-regler-facture-residentiel/chunks.jsonl `
  --chunks data/processed/redal/chunks.jsonl `
  --chunks data/processed/telecom/chunks.jsonl `
  --evaluation data/evaluation/onee_retrieval_questions.v0.1.jsonl `
  --evaluation data/evaluation/onee_retrieval_darija_robustness.v0.1.jsonl `
  --evaluation data/evaluation/redal_retrieval_questions.v0.1.jsonl `
  --evaluation data/evaluation/darija_script_pairs.v0.1.jsonl `
  --evaluation data/evaluation/telecom_retrieval_questions.v0.1.jsonl `
  --output outputs/hybrid-all-corpora-qwen3.json `
  --device cuda --top-k 5 --candidate-k 20 `
  --rrf-k 60 --bm25-weight 0.5 --semantic-weight 0.5
```

## Résultats de la baseline gelée

Sur les 123 cas du split test, dont 86 positifs :

| Système | Recall@5 | Binary Recall@5 | MRR | Balanced accuracy abstention |
| --- | ---: | ---: | ---: | ---: |
| BM25 normalisé | 0,905 | 0,919 | **0,785** | **0,745** |
| Qwen3 | 0,950 | 0,953 | 0,704 | 0,619 |
| Hybride RRF | **0,973** | **0,977** | 0,745 | 0,588 |

Sur le seul domaine télécom, Qwen3 et l'hybride obtiennent un Recall@5 de
0,905 contre 0,810 pour BM25. Qwen3 possède aussi le meilleur MRR télécom
(0,754). Sur l'ensemble du corpus, BM25 conserve cependant le meilleur premier
rang et la meilleure abstention.

La conclusion honnête est donc : la fusion égale améliore la couverture, mais
elle ne domine pas toutes les métriques. Les poids ne doivent pas être ajustés
sur le split test observé. Une future optimisation doit choisir les poids sur
la calibration, puis ne mesurer le test qu'une seule fois.
