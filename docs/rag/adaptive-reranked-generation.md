# Génération avec contexte reranké adaptatif

## Objectif

Cette expérience vérifie si retirer les passages distracteurs avant la génération
améliore Qwen3.5-4B sur les questions en darija arabe, avec une réponse demandée
en arabe standard.

Le baseline transmet quatre passages par question. La variante classe ces quatre
candidats avec `Qwen/Qwen3-Reranker-0.6B`, puis transmet entre un et trois passages.

## Politique

Le reranker reçoit une instruction en anglais spécialisée pour les factures
multilingues. Les scores sont calculés par paire `(question, passage)`.

La politique conserve toujours le meilleur passage. Elle ajoute le deuxième et le
troisième lorsque leur score vaut au moins 40 % du meilleur score. Le ratio de 40 %
a été choisi sur les 25 cas de validation en darija arabe. Les labels du test n'ont
pas été utilisés pour choisir ce ratio.

Sur le test de 40 cas, la distribution obtenue est :

- 16 contextes à un passage ;
- 10 contextes à deux passages ;
- 14 contextes à trois passages ;
- moyenne : 1,95 passage au lieu de 4.

## Résultats

| Mesure | 4 passages | Reranker adaptatif | Écart |
|---|---:|---:|---:|
| Core pass | 77,5 % | 60,0 % | -17,5 pts |
| Strict grounded pass | 45,0 % | 30,0 % | -15,0 pts |
| Exactitude d'abstention | 85,0 % | 87,5 % | +2,5 pts |
| Réponses dangereuses | 5 | 4 | -1 |
| Précision des citations gold | 73,2 % | 49,5 % | -23,7 pts |
| Rappel des citations gold | 81,8 % | 50,0 % | -31,8 pts |
| Cas citant un distracteur | 15 | 13 | -2 |
| Rappel des valeurs critiques | 84,8 % | 75,8 % | -9,1 pts |
| Latence moyenne | 26,89 s | 23,22 s | -3,68 s |
| Tokens moyens du prompt | 1 292 | 963 | -329 |

Le reranker ne conserve au moins un passage gold que pour 25 des 33 questions
répondables. Huit questions perdent donc toute preuve correcte avant la génération.
Sur les 40 cas, trois résultats stricts s'améliorent, neuf régressent et vingt-huit
restent inchangés.

## Interprétation

La réduction du contexte fonctionne techniquement : le prompt est plus court de
25,5 % et la génération moyenne est plus rapide de 13,7 %. Elle réduit légèrement
les citations de distracteurs et les réponses dangereuses.

Cependant, avec ce reranker de 0,6 milliard de paramètres et ces questions en
darija arabe, le coût en rappel est trop élevé. Une preuve supprimée ne peut pas être
recréée par le générateur. La variante adaptative ne doit donc pas remplacer le
baseline à ce stade.

## Décision

Conserver le reranker comme composant expérimental, mais garder le contexte fixe
Top 4 pour la version actuelle. Avant une nouvelle évaluation finale, il faudra :

1. améliorer le reranking cross-lingue ou tester un modèle plus robuste ;
2. ajouter un mécanisme conservateur qui élargit à trois passages lorsque le score
   est faible ou instable ;
3. calibrer ce mécanisme sur un jeu de validation plus large ;
4. évaluer une seule fois sur un nouveau test gelé, car le test actuel a désormais
   servi au diagnostic.

## Dernière tentative conservatrice

Une seconde politique a ensuite été testée : trois passages par défaut, deux
seulement avec une confiance forte et un seul avec une confiance exceptionnelle.
Elle produit 1 contexte à un passage, 2 à deux passages et 37 à trois passages,
soit une moyenne de 2,9. Elle conserve au moins une preuve gold dans 90,9 % des
questions répondables.

| Mesure | 4 passages | Adaptatif agressif | Conservateur |
|---|---:|---:|---:|
| Core pass | 77,5 % | 60,0 % | 72,5 % |
| Strict grounded pass | 45,0 % | 30,0 % | 37,5 % |
| Exactitude d'abstention | 85,0 % | 87,5 % | 85,0 % |
| Réponses dangereuses | 5 | 4 | 4 |
| Précision citations gold | 73,2 % | 49,5 % | 59,1 % |
| Rappel citations gold | 81,8 % | 50,0 % | 63,6 % |
| Cas citant un distracteur | 15 | 13 | 16 |
| Rappel des valeurs critiques | 84,8 % | 75,8 % | 75,8 % |
| Latence moyenne | 26,89 s | 23,22 s | 21,41 s |
| Tokens moyens du prompt | 1 292 | 963 | 1 116 |

La variante conservatrice récupère une grande partie de la qualité perdue et réduit
la latence moyenne de 20,4 %. Elle reste toutefois inférieure au Top 4 sur les deux
mesures principales et sur les citations gold. La conclusion qualité reste donc de
conserver quatre passages. Cette seconde comparaison est exploratoire, car le même
test avait déjà été observé lors du diagnostic de la première variante.

## Reproduction

```powershell
py -3.12 scripts/prepare_adaptive_reranked_sft.py `
  --output-dir artifacts/evaluation/adaptive-reranked-darija-arab-v02 `
  --language darija_arab --device cpu --batch-size 16 `
  --max-length 512 --minimum-chunks 1

$env:PYTHONPATH = "src"
py -3.12 scripts/evaluate_sft_ollama.py `
  --model qwen3.5:4b `
  --prompts artifacts/evaluation/adaptive-reranked-darija-arab-v02/test_prompts.jsonl `
  --cases artifacts/evaluation/adaptive-reranked-darija-arab-v02/test_cases.jsonl `
  --language darija_arab --answer-language ar --num-predict 400 `
  --output artifacts/evaluation/qwen35-4b-adaptive-reranked-darija-arab-to-msa-v02.json
```
