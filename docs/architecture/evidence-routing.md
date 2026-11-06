# EvidenceRouter : décision fondée sur les preuves

## Rôle

`EvidenceRouter` reçoit un besoin déjà décomposé, une indication Qwen, les
champs structurés de la facture et les candidats issus du pipeline RAG. Il ne
génère aucune réponse et ne modifie aucune valeur. Il retourne une route, les
identifiants de preuves sélectionnées, deux confiances séparées et des raisons
auditables.

## Canal facture

Qwen3-Embedding compare la question aux descriptions multilingues des types de
champs, jamais aux valeurs. Un champ ne peut devenir une preuve que si son
statut est `validated`, sa confiance OCR franchit le minimum, sa provenance est
présente et sa liste d'erreurs est vide. Une correspondance d'alias explicite
est un signal déterministe fort après ces quatre portes.

## Canal RAG

Le routeur consomme les scores du reranker et de la recherche hybride. Avant
le score, il exige une correspondance du fournisseur, du service, de la zone
et de l'audience. La source doit être active, de risque faible ou moyen et de
rôle `operator_primary`, `legal_primary` ou `regulator`. Une source de presse,
un mauvais fournisseur ou un chunk périmé est rejeté même avec un score élevé.

## Fusion conservatrice

- Une seule preuve forte et une indication compatible donnent `invoice` ou
  `official_rag`.
- Une preuve exceptionnellement forte ou un alias exact validé peut corriger
  une indication Qwen erronée.
- `both` exige les deux canaux et l'indication que les deux sont nécessaires.
- Une contradiction non résolue, un champ ambigu ou une preuve manquante donne
  `unknown`.

La confiance de `both` est le minimum des deux canaux. Cela empêche une très
bonne récupération RAG de masquer une lecture OCR faible, et inversement.

## Calibration et évaluation

Les seuils sont sélectionnés sur le split `calibration` par grille déterministe
et sont ensuite figés. Le rapport sépare un test déjà observé d'un holdout
`challenge` ajouté après les corrections. Le challenge contient des
paraphrases sans alias exact, des hints Qwen faux, des chunks de mauvais
fournisseurs, des sources périmées et des preuves proches mais insuffisantes.

Le premier holdout challenge atteint 50 % d'exactitude de route. Les erreurs
sont des abstentions sur les paraphrases éloignées ; aucune décision non
abstentionniste n'est dirigée vers la mauvaise source. Ce résultat indique une
bonne sûreté initiale mais une couverture insuffisante. Il ne justifie pas une
utilisation en production.

## Amélioration V2

La V2 accepte aussi un champ sans alias lorsque quatre signaux concordent : le
hint vise la facture, le champ est premier, sa marge sur le second est
suffisante et toutes les portes OCR sont franchies. Un hint `unknown` ne peut
pas utiliser cette règle. Les questions composées sont routées besoin par
besoin puis agrégées ; la présence d'un besoin facture et d'un besoin officiel
produit `both` sans diluer leurs embeddings dans une même phrase.

Sur le challenge déjà observé, l'exactitude passe de 50 % à 91,67 %. Un nouveau
holdout V2, créé après cette modification, obtient aussi 91,67 %, avec 88,89 %
de rappel répondable, 100 % de précision parmi les décisions prises et 0 % de
faux routage.

## Vérification avec le vrai pipeline RAG

Un adaptateur relie désormais directement le Top-N de BM25 normalisé +
Qwen3-Embedding + RRF au Qwen3-Reranker, puis enrichit les candidats avec les
métadonnées requises par le routeur. Sur six questions nouvelles, le bon chunk
est premier dans cinq cas sur six, mais le routeur n'en accepte que deux avec
le seuil `0.450317` gelé sur les 104 cas de calibration du reranker. Cette
expérience confirme que la prochaine limite est la calibration multilingue du
RAG, surtout en arabe et darija, et non une raison de baisser post-hoc le seuil.
