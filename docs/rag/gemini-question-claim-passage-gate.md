# Barrière Gemini : question ↔ claim ↔ passage

## Objectif

Une citation peut être exacte tout en étant hors sujet. La barrière vérifie donc
deux relations indépendantes avant d'afficher une réponse :

1. `question -> claim` : l'affirmation répond-elle réellement à la demande ?
2. `claim -> passages cités` : les citations prouvent-elles entièrement
   l'affirmation ?

Le générateur reste Qwen3.5-4B local. Gemini ne rédige pas la réponse et ne peut
modifier ni les claims ni les citations ; il renvoie uniquement un verdict
structuré.

## Architecture

```text
Question + passages récupérés
            |
            v
       Qwen3.5-4B
            |
            v
 Réponse JSON avec claims et citations
            |
            +--> garde déterministe
            |      - langue déclarée
            |      - claims présents ou abstention
            |      - citations dans le contexte
            |      - nombres présents dans les citations
            |
            +--> Gemini claim verifier
                   - répond à la question ?
                   - soutenu par les citations ?
                   - bon fournisseur ?
                   - valeurs conservées ?
            |
            v
       show / abstain / regenerate
```

La décision finale `show` exige l'accord des deux barrières. Gemini ne peut donc
pas autoriser un nombre absent de la source ou une citation inconnue.

## Schémas de sortie

`ClaimVerdict` contient, pour chaque claim :

- `answers_question` ;
- `entailed_by_citations` ;
- `provider_consistent` ;
- `critical_values_preserved` ;
- `accepted`, forcé par Pydantic à être la conjonction des quatre contrôles ;
- une justification courte.

`CaseClaimVerdict` ajoute :

- `evidence_sufficient` ;
- `abstention_correct` ;
- `response_relevant` ;
- `safe_to_show` ;
- `recommended_action` parmi `show`, `abstain` et `regenerate`.

## Isolation des responsabilités

Le juge ne reçoit ni `gold_chunk_ids`, ni réponse de référence, ni label
`should_abstain`. Il doit prendre sa décision uniquement à partir de la question,
du fournisseur demandé, de la réponse candidate et des passages disponibles.
Les labels gold ne servent qu'après l'appel pour mesurer le juge.

Cette séparation évite une fuite d'évaluation et reproduit mieux les conditions
de production, où la réponse gold n'existe pas.

## Contrôle déterministe

`online_deterministic_gate` est volontairement limité aux propriétés vérifiables
sans LLM :

- la langue déclarée correspond à la langue de sortie ;
- une abstention ne contient aucun claim ;
- une réponse contient au moins un claim ;
- chaque claim possède une citation ;
- chaque citation appartient aux passages récupérés ;
- chaque nombre du claim apparaît dans ses passages cités.

Ce contrôle ne sait pas juger que « paiement disponible 24 h/24 » est hors sujet
pour une question sur une coupure. C'est précisément le rôle de Gemini.

## Pourquoi des lots de trois

Le quota gratuit observé est de 20 requêtes quotidiennes pour
`gemini-2.5-flash`. Plusieurs cas sont donc regroupés dans un seul appel. Un lot
de cinq s'est montré lent et sujet aux reprises internes ; trois cas constitue
ici un meilleur compromis entre quota, taille du prompt et temps de réponse.

Le script :

- impose un timeout réseau de 90 secondes ;
- désactive les retries implicites multiples du SDK ;
- conserve trois retries explicites et visibles ;
- écrit un fichier `.partial.jsonl` après chaque lot ;
- reprend seulement les `case_id` manquants après une interruption ou un reset
  de quota.

## Résultats au 4 octobre 2026

39 des 40 réponses darija arabe -> arabe standard ont été jugées. Le cas
`sftq-v04-0620` reste en attente du prochain reset du quota.

| Mesure | Résultat |
|---|---:|
| Réponses autorisées, abstentions correctes incluses | 18/39 — 46,2 % |
| Réponses envoyées vers régénération | 19/39 |
| Réponses transformables en abstention | 2/39 |
| Hard negatives protégés | 6/6 — 100 % |
| Réponses incorrectes comparées au grounded strict et rejetées | 19/21 |
| Réponses grounded strict acceptées | 16/18 |
| Accord avec le grounded strict | 35/39 — 89,7 % |
| Précision de l'autorisation contre ce proxy | 16/18 — 88,9 % |
| Rappel de l'autorisation contre ce proxy | 16/18 — 88,9 % |

Le grounded strict reste un proxy imparfait, pas une vérité humaine. Les quatre
désaccords sont instructifs :

- Gemini accepte deux réponses Redal que la métrique stricte pénalise pour des
  valeurs de référence absentes, alors que les réponses paraissent utiles ;
- Gemini rejette une formulation IAM ambiguë et une réponse Orange qui omet le
  qualificatif important « au moins » devant 10 jours, toutes deux acceptées par
  la métrique stricte.

Une revue humaine doit départager ces quatre cas avant de mesurer définitivement
la qualité du juge.

## Exemple décisif

Pour une question ONEE sur l'avance, Qwen produit deux claims :

1. l'acompte permet de déposer une avance — accepté ;
2. le virement permet le prélèvement mensuel — vrai et sourcé, mais hors sujet.

Gemini marque le deuxième claim `answers_question=false` et demande une
régénération. C'est exactement la défaillance que le simple contrôle de citation
ne pouvait pas détecter.

## Sécurité et confidentialité

Cette expérience envoie seulement des questions synthétiques et des passages
publics. Une future utilisation sur des requêtes réelles doit faire l'objet
d'une validation de confidentialité, d'une suppression des identifiants et
d'une politique explicite avant tout envoi vers Gemini. Les images ou factures
utilisateur ne doivent jamais être envoyées par ce vérificateur.

## Fichiers

- logique réutilisable : `src/qralia/rag/claim_verification.py` ;
- exécution et reprise : `scripts/judge_claim_alignment_gemini.py` ;
- tests : `tests/test_claim_verification.py` ;
- rapport courant :
  `artifacts/evaluation/gemini-claim-gate-darija-arab-to-msa-v01.json` ;
- cache reprenable :
  `artifacts/evaluation/gemini-claim-gate-darija-arab-to-msa-v01.partial.jsonl`.
