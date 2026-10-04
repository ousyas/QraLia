# Reranking Qwen3 du Top-N hybride

## Objectif

Le reranker ajoute une quatrième configuration au benchmark existant :

```text
question + route sûre
        |
        +--> BM25 normalisé --------+
        |                            |
        +--> Qwen3-Embedding --------+--> Weighted RRF --> Top-20
                                                            |
                                      Qwen3-Reranker <-------+
                                                            |
                                                     Top-5 réordonné
```

BM25 et le modèle d'embeddings servent à produire rapidement des candidats.
Le reranker relit ensuite conjointement la question et chacun de ces passages.
Il comprend donc mieux leur interaction qu'un bi-encodeur, qui calcule les deux
vecteurs séparément. En contrepartie, il est beaucoup plus lent et ne doit pas
être appliqué à tout le corpus.

Le routage `domain + provider_id + service_type + service_zone_id + audience`
reste appliqué **avant** le reranking. Le modèle ne peut donc pas réintroduire
un chunk d'un autre opérateur ou d'une autre zone.

## Modèle et reproductibilité

- modèle : `Qwen/Qwen3-Reranker-0.6B` ;
- révision figée : `e61197ed45024b0ed8a2d74b80b4d909f1255473` ;
- exécution locale en FP16 sur CUDA ;
- longueur maximale explicite : 1 024 tokens ;
- aucune donnée d'entraînement QraLia et aucun fine-tuning.

Le modèle reçoit une paire `(question, passage)`. Sentence Transformers produit
un logit, puis le code lui applique une sigmoïde :

```text
score = sigmoid(logit) = 1 / (1 + exp(-logit))
```

Le résultat est compris entre 0 et 1 et sert au classement ainsi qu'à la
calibration de l'abstention. Il ne faut pas l'interpréter comme une probabilité
parfaitement calibrée : seule sa comparaison expérimentale sur notre jeu gelé
permet de juger son utilité.

## Lecture du code

Le module est `src/qralia/rag/reranker_evaluation.py`.

### `Qwen3CrossEncoderReranker`

Cette classe charge le `CrossEncoder` et expose seulement `score(pairs)`. Sur
CUDA, `torch.float16` divise approximativement par deux la mémoire des poids.
La méthode `predict` traite les paires par lots et applique `torch.nn.Sigmoid`.
Le chargement est local après le premier téléchargement du modèle.

### `collect_hybrid_candidates`

Pour chaque cas d'évaluation, cette fonction :

1. applique le routage de sécurité ;
2. lance BM25 et Qwen3-Embedding ;
3. fusionne leurs rangs par Weighted RRF ;
4. conserve le Top-N hybride pour le reranker ;
5. enregistre aussi les Top-5 BM25, sémantique et hybride.

Les quatre systèmes sont ainsi comparés sur exactement la même question et la
même route. Le reranker ne bénéficie d'aucun cas supplémentaire.

### `rerank_with_scores`

Cette fonction pure associe un score à chaque candidat puis trie par :

1. score reranker décroissant ;
2. ancien rang hybride croissant en cas d'égalité ;
3. `chunk_id` pour un dernier départage déterministe.

Elle conserve `hybrid_rank`, `hybrid_score` et `hybrid_confidence`. On peut donc
auditer chaque déplacement au lieu de perdre la décision du premier étage.
Elle ne peut ni inventer ni ajouter un candidat absent.

### `candidate_set_metrics`

Cette métrique calcule le plafond du reranker sur le Top-N candidat. Si le bon
chunk n'est pas présent dans cet ensemble, le meilleur reranker possible ne
peut pas le retrouver. Il faut donc lire cette couverture avant d'attribuer un
échec au cross-encoder.

### Mémoire GPU

Le bi-encodeur construit d'abord l'index et tous les candidats. Il est ensuite
supprimé, le cache CUDA est vidé, puis seulement le cross-encoder est chargé.
Ce séquencement évite de garder les deux modèles Qwen simultanément sur une
carte de 4 Go.

### Latence

Le reranker score toutes les paires en lots pour une mesure réaliste du débit.
Sa latence par cas est reportée comme :

```text
latence hybride du cas + temps total de reranking / nombre de cas
```

Il s'agit donc d'une latence moyenne amortie, pas d'une mesure requête par
requête en ligne. Le rapport l'indique explicitement.

## Protocole d'évaluation

Les 227 cas restent inchangés. Le seuil d'abstention est choisi exclusivement
sur le split `calibration`, puis mesuré sur `test`. Le rapport contient :

- Recall@1/3/5 et Binary Recall@1/3/5 ;
- MRR ;
- balanced accuracy d'abstention ;
- résultats par langue, catégorie, domaine et fournisseur ;
- comparaison darija latine / darija arabe ;
- latence moyenne, p95 et maximale ;
- couverture du Top-N candidat.

Le Top-20 candidat atteint un Recall et un Binary Recall de `1,000` sur les 86
cas positifs du split test : tous les passages attendus sont donc disponibles
pour le reranker. Le plafond de candidats n'explique aucun de ses échecs.

Résultats sur les 123 cas de test :

| Système | Recall@1 | Recall@3 | Recall@5 | Binary Recall@5 | MRR | Abstention BA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 normalisé | 0,667 | 0,820 | 0,905 | 0,919 | 0,785 | **0,745** |
| Qwen3-Embedding | 0,529 | 0,784 | 0,950 | 0,953 | 0,704 | 0,619 |
| Hybride RRF | 0,535 | **0,900** | 0,973 | 0,977 | 0,745 | 0,588 |
| Hybride + reranker | **0,669** | 0,817 | **0,981** | **0,988** | **0,799** | 0,627 |

Le reranker donne donc le meilleur premier rang, le meilleur MRR et la
meilleure couverture à cinq. Il ne domine toutefois pas tout : l'hybride reste
meilleur à Recall@3 et BM25 reste nettement meilleur pour l'abstention.

Par langue, le reranker obtient Recall@5 `1,000` en français, arabe classique
et darija arabe, et `0,963` en darija latine. Son MRR vaut respectivement
`0,938`, `0,900`, `0,699` et `0,784`. Les sous-groupes arabe classique et
français sont petits ; ces chiffres sont des indications, pas une preuve de
généralisation.

La contrepartie principale est la latence locale mesurée : environ `3 777 ms`
par cas en moyenne pour le pipeline complet, contre `84 ms` pour l'hybride et
`0,27 ms` pour BM25. La mesure reranker est amortie sur un traitement par lots
de 16, donc elle ne représente pas encore une API temps réel optimisée.

Le rapport complet se trouve dans
`outputs/reranker-all-corpora-qwen3.json`. Aucun hyperparamètre ne doit être
ajusté à partir du split test de ce rapport.

## Commande reproductible

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m qralia.rag.reranker_evaluation `
  --chunks data/processed/onee-regler-facture-residentiel/chunks.jsonl `
  --chunks data/processed/redal/chunks.jsonl `
  --chunks data/processed/telecom/chunks.jsonl `
  --evaluation data/evaluation/onee_retrieval_questions.v0.1.jsonl `
  --evaluation data/evaluation/onee_retrieval_darija_robustness.v0.1.jsonl `
  --evaluation data/evaluation/redal_retrieval_questions.v0.1.jsonl `
  --evaluation data/evaluation/darija_script_pairs.v0.1.jsonl `
  --evaluation data/evaluation/telecom_retrieval_questions.v0.1.jsonl `
  --output outputs/reranker-all-corpora-qwen3.json `
  --device cuda --top-k 5 --candidate-k 20 --rerank-k 20 `
  --reranker-batch-size 16 --rrf-k 60 `
  --bm25-weight 0.5 --semantic-weight 0.5
```
