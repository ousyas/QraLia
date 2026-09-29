# Répartition initiale du binôme

La répartition indique un responsable principal, pas un silo. Chaque livraison reçoit une revue croisée et une courte séance où l’auteur explique les choix, les métriques et les limites.

| Domaine | Responsable principal | Pair/revue | Livrable d’apprentissage |
|---|---|---|---|
| Corpus, schéma documentaire, retrieval | Oussama | Meryeme | expliquer BM25, embeddings, hybride et filtres |
| Jeu d’évaluation et métriques RAG | Oussama | Meryeme | justifier Recall@k, MRR, citations et abstention |
| Factures synthétiques et vérité terrain | Meryeme | Oussama | documenter variations et pièges OCR |
| Benchmark OCR/extraction | Meryeme | Oussama | comparer qualité, confidentialité, latence et coût |
| Validateurs montants/dates/identifiants | travail en binôme | revue croisée | démontrer les cas limites par tests |
| API et intégration | travail en binôme | revue croisée | tracer le flux de bout en bout |
| Docker/CI | rotation par sprint | l’autre membre | reconstruire et expliquer l’environnement |
| Documentation/démo | travail en binôme | revue croisée | distinguer résultats mesurés et ambitions |

## Rituels légers

- début de sprint : objectif mesurable et critères d’acceptation ;
- chaque pull request : auteur, reviewer, test ou métrique affectée ;
- fin de semaine : démonstration de dix minutes par la personne qui n’a pas écrit le composant ;
- journal des décisions : une ADR courte pour chaque choix difficile à inverser ;
- aucun chiffre sur le CV avant qu’un script et un artefact permettent de le reproduire.

