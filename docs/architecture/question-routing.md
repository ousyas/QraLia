# Routage sémantique facture / corpus officiel

## Objectif du premier incrément

Une seule requête locale à `qwen3.5:4b` effectue deux tâches :

1. décomposer la demande en un à trois besoins atomiques ;
2. proposer pour chaque besoin une source probable : `invoice`,
   `official_rag`, `both` ou `unknown`.

Le modèle ne reçoit aucune valeur extraite et ne répond jamais à la question.
Sa sortie est une hypothèse de routage, pas une preuve.

## Contrôles déterministes

Le code rejette une sortie si le JSON est invalide, si un span ne provient pas
exactement de la question, si un nombre disparaît, si le nombre de besoins et
`is_composite` se contredisent ou si la source ne fait pas partie de
l'énumération autorisée. Après deux sorties invalides, le système retourne un
besoin unique `unknown`.

La route globale n'est jamais demandée au modèle. Elle est dérivée en Python :

- seulement `invoice` -> `invoice` ;
- seulement `official_rag` -> `official_rag` ;
- présence de `both`, ou mélange `invoice` + `official_rag` -> `both` ;
- seulement `unknown`, ou mélange partiellement inconnu -> `unknown`.

## Jeu d'évaluation v0.1

Les 48 questions sont entièrement synthétiques et reprennent seulement les
types de champs observés dans la mise en page des factures fournies : montant,
échéance, identifiants, tarif, hors taxes, taxes et total en espèces. Aucune
valeur, aucun identifiant et aucune donnée personnelle du PDF ne sont copiés.

Le jeu est équilibré :

- 16 questions en français, 16 en arabe standard et 16 en darija arabe ;
- 12 cas pour chacune des quatre routes ;
- questions atomiques, questions mixtes et abstentions.

## Résultat Qwen3.5-4B

Sur les 48 cas gelés :

| Mesure | Score |
|---|---:|
| Exactitude du nombre de besoins | 85,42 % |
| Exactitude de la route globale | 83,33 % |
| Exactitude de la source atomique | 82,29 % |
| Exactitude stricte | 66,67 % |
| Taux de fallback | 2,08 % |

Exactitude de routage par langue : français 93,75 %, arabe standard 93,75 %,
darija arabe 62,50 %. Les erreurs principales sont la sur-segmentation des
listes de champs, la confusion `official_rag` / `both` pour la contestation et
des abstentions incorrectes sur des formulations darija.

## Limite et prochaine couche

Cette version ne doit pas router directement une réponse en production. La
décision finale doit être confirmée par une couche de preuves : disponibilité
et statut de validation des champs OCR côté facture, puis score, métadonnées et
seuil d'abstention côté RAG. Un signal Qwen contradictoire ne doit jamais
autoriser l'invention d'un montant, d'une date ou d'un identifiant.
