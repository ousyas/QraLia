# Benchmark apparié darija latine / darija arabe

## Question étudiée

Ce benchmark teste une question précise : à intention identique, la recherche
retrouve-t-elle mieux le passage officiel lorsque la darija est écrite en
alphabet latin ou en alphabet arabe ? Il ne mesure pas la qualité générale de
la langue produite par un LLM.

## Construction contrôlée

Le fichier `data/evaluation/darija_script_pairs.v0.1.jsonl` contient 24
intentions et 48 cas :

- 24 formulations en darija latine et 24 en darija arabe ;
- 16 intentions positives et 8 demandes d'abstention ;
- 12 intentions de calibration et 12 intentions de test ;
- des routes ONEE et SRM-RSK, zone ex-Redal ;
- un même `pair_id`, une même route et les mêmes sections pertinentes pour les
  deux écritures d'une intention.

Les deux variantes d'une paire restent toujours dans le même split. Cela évite
d'utiliser une écriture pour choisir le seuil puis son équivalent presque
identique comme prétendu exemple indépendant de test.

Le fichier est validé structurellement, mais reste en statut pédagogique : ses
formulations doivent être relues par au moins un locuteur marocain qui n'a pas
rédigé les cas, avant de présenter les chiffres comme une évaluation externe.

## Commande Qwen3 sur GPU

```powershell
.venv\Scripts\python.exe -m qralia.rag.semantic_evaluation `
  --chunks data/processed/onee-regler-facture-residentiel/chunks.jsonl `
  --chunks data/processed/redal/chunks.jsonl `
  --evaluation data/evaluation/darija_script_pairs.v0.1.jsonl `
  --output outputs/evaluation/darija-script-pairs-semantic-qwen3-gpu-v0.1.json `
  --device cuda `
  --batch-size 2 `
  --max-seq-length 512 `
  --top-k 5
```

## Résultats sur le split de test

Chaque écriture comporte huit intentions positives dans le split de test.

| Moteur | Écriture | Recall@1 | Recall@5 | MRR |
|---|---|---:|---:|---:|
| BM25 simple | latine | 0,375 | 0,500 | 0,438 |
| BM25 simple | arabe | 0,250 | 0,500 | 0,344 |
| BM25 + lexique | latine | 0,875 | 1,000 | 0,917 |
| BM25 + lexique | arabe | 0,625 | 0,875 | 0,708 |
| Qwen3-Embedding-0.6B | latine | 0,250 | 1,000 | 0,521 |
| Qwen3-Embedding-0.6B | arabe | **0,375** | 0,875 | **0,573** |

Avec Qwen3, l'arabe gagne deux comparaisons de rang, le latin en gagne trois et
trois sont à égalité. La différence moyenne de reciprocal rank en faveur de
l'arabe vaut seulement `+0,052`. L'exactitude de la décision d'abstention vaut
`0,667` en écriture arabe contre `0,417` en écriture latine avec le seuil commun
calibré. La balanced accuracy globale de Qwen3 vaut `0,656`.

## Interprétation honnête

L'hypothèse « Qwen comprend mieux la darija en alphabet arabe » n'est pas
confirmée de manière générale. L'arabe obtient ici davantage de rangs 1 et un
MRR légèrement supérieur, mais le latin conserve un meilleur Recall@5 et gagne
plus de paires. Qwen apporte surtout une récupération robuste dans les deux
écritures par rapport au BM25 simple.

Le lexique BM25 obtient un MRR élevé parce que ce jeu emploie volontairement des
formes déjà couvertes par le dictionnaire (`factora`, `chik`, ainsi que leurs
équivalents arabes). Ce résultat mesure donc l'efficacité sur un vocabulaire
connu, pas une généralisation à de nouvelles formulations.

Une conclusion plus forte demandera davantage d'intentions rédigées et revues
indépendamment par des locuteurs marocains, sans consulter les chunks ni le
lexique pendant leur rédaction.
