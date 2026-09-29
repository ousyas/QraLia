# Baseline de récupération BM25

## But

La baseline mesure ce qu'une recherche lexicale simple peut récupérer dans les
cinq chunks actifs, avant d'introduire des embeddings ou un modèle de langage.
Le code se trouve dans `src/qralia/rag/bm25_evaluation.py`.

## Exécution

```powershell
.\.venv\Scripts\python.exe -m qralia.rag.bm25_evaluation `
  --chunks "data/processed/onee-regler-facture-residentiel/chunks.jsonl" `
  --evaluation "data/evaluation/onee_retrieval_questions.v0.1.jsonl" `
  --output "artifacts/evaluation/bm25-v0.1.json" `
  --top-k 5
```

Le rapport complet est généré localement dans `artifacts/`, dossier ignoré par
Git. Le code, le corpus source, le jeu d'évaluation et ce résumé sont versionnés.

## Pipeline

```text
chunks JSONL actifs -----> validation -----> index BM25
questions JSONL ---------> validation -----> recherche par audience
                                             |
                                             v
                                  classement et scores par cas
                                             |
                    calibration du seuil sur split=calibration
                                             |
                                             v
                                  métriques sur split=test
```

Seuls les champs `metadata.heading_path` et `text` sont indexés. Les chunks sont
filtrés par audience avant le classement. Un résultat de score nul est retiré,
car il ne partage aucun token avec la question et son rang serait arbitraire.

## Tokenisation

La tokenisation conserve les séquences alphanumériques Unicode et applique
`casefold()` pour neutraliser la casse. Elle ne réalise ni racinisation, ni
traduction, ni suppression de mots vides. Cette simplicité rend la baseline
facile à expliquer, mais limite naturellement la récupération entre langues.

## Score BM25

Pour chaque terme de la question présent dans un chunk, BM25 combine :

- sa rareté dans le corpus (`IDF`) ;
- sa fréquence dans le chunk ;
- la longueur du chunk comparée à la longueur moyenne ;
- `k1 = 1.5`, qui contrôle la saturation de fréquence ;
- `b = 0.75`, qui contrôle la normalisation par longueur.

La somme des contributions des termes forme le score du chunk. Les résultats
sont triés par score décroissant, puis par identifiant pour garantir un ordre
déterministe en cas d'égalité.

## Métriques de classement

Les métriques de classement sont calculées uniquement sur les questions qui ont
au moins une section pertinente.

- `Recall@k` : proportion de toutes les sections pertinentes présentes dans les
  `k` premiers résultats.
- `Binary Recall@k` ou Hit Rate@k : vaut 1 si au moins une section pertinente
  est présente dans les `k` premiers résultats, sinon 0, puis moyenne des cas.
- `MRR` : moyenne de l'inverse du rang de la première section pertinente.

Les questions d'abstention n'ont aucune section pertinente et sont exclues de
ces métriques pour éviter une division par zéro.

## Seuil d'abstention

BM25 produit un score, mais ne sait pas naturellement décider si une réponse est
suffisamment étayée. Le programme teste des seuils construits à partir des
meilleurs scores des dix cas de calibration et retient celui qui maximise la
`balanced accuracy`. En cas d'égalité, le seuil le plus élevé est choisi afin de
préférer l'abstention prudente.

Ce seuil est ensuite appliqué sans modification aux 25 cas de test. Les mesures
rapportées sont `answer_recall`, `abstention_recall`, `accuracy` et
`balanced_accuracy`, avec la matrice `TP/FP/TN/FN`.

## Résultats v0.1

Résultats sur les 25 cas de test, dont 18 questions positives :

| Métrique | Valeur |
|---|---:|
| Recall@1 | 0,583 |
| Recall@3 | 0,694 |
| Recall@5 | 0,694 |
| Binary Recall@1 | 0,667 |
| Binary Recall@3 | 0,722 |
| Binary Recall@5 | 0,722 |
| MRR | 0,694 |
| Seuil d'abstention | 2,592509 |
| Balanced accuracy d'abstention | 0,635 |

Pour l'abstention sur les 25 cas de test : `TP=10`, `FP=2`, `TN=5`, `FN=8`.
`answer_recall=0,556` et `abstention_recall=0,714`.

Par langue, le MRR vaut `0,938` en français, `1,000` en darija latine,
`0,000` en arabe et `0,000` en darija arabe. Le bon résultat en darija latine
provient surtout de mots partagés avec le français, par exemple `chèque`,
`virement` ou `acompte`; il ne prouve pas une compréhension de la darija.

Ces résultats justifient de tester ensuite une récupération multilingue, tout en
gardant exactement le même corpus, les mêmes filtres et le même jeu de test.
