# Architecture du MVP

## Frontière du système

Le MVP traite un seul domaine : les factures d’électricité ONEE, avec priorité aux particuliers basse tension. Les pages visant les professionnels, la moyenne tension ou les grands comptes sont conservées hors de l’index principal, sauf besoin explicitement couvert par un test.

## Flux principal

1. **Entrée** — image ou PDF synthétique et question texte.
2. **Prétraitement** — orientation, contrôle de lisibilité et séparation des pages.
3. **OCR/extraction** — texte brut, champs structurés et coordonnées éventuelles.
4. **Validation** — types, formats, cohérence des dates, contrôle des additions disponibles et conservation exacte de la chaîne source.
5. **Récupération** — recherche lexicale et sémantique sur les seuls chunks officiels éligibles, puis reranking.
6. **Composition** — combinaison des champs validés et des passages récupérés sans autoriser le LLM à modifier les valeurs protégées.
7. **Réponse** — langage simple, citation, confiance séparée pour le document et pour la connaissance, ou abstention.
8. **Journal d’évaluation** — latence, documents/chunks retrouvés, citations et résultat des assertions ; jamais le document personnel brut.

## Contrats de confiance

| Donnée | Source autorisée | Protection |
|---|---|---|
| Montant, date, numéro de facture/contrat | document utilisateur | chaîne exacte + validation déterministe |
| Signification d’un champ | corpus officiel | citation obligatoire |
| Modalités de paiement et retard | corpus officiel adapté à l’audience | filtre d’audience + citation |
| Traduction ou simplification | modèle génératif | aucun fait nouveau |
| Valeur illisible | aucune | abstention, avertissement, aide d’un proche |

Deux scores sont exposés : `document_confidence` pour la lecture du document et `retrieval_confidence` pour la qualité des sources retrouvées. Un score global unique masquerait la cause d’une incertitude.

## Composants prévus

- API Python/FastAPI ;
- modèles Pydantic alignés sur les schémas JSON ;
- adaptateur OCR interchangeable ;
- extracteur de champs avec provenance ;
- validateurs purs et testables ;
- registre de sources et pipeline d’acquisition reproductible ;
- index BM25 et embeddings multilingues, choisis après benchmark ;
- reranker facultatif, conservé seulement s’il améliore les métriques ;
- générateur contraint et formateur de citations ;
- banc d’évaluation hors ligne.

Les fournisseurs OCR/LLM, les embeddings, le reranker et la base vectorielle restent volontairement non choisis. Le benchmark devra comparer qualité français/arabe/darija, exécution locale, confidentialité, latence et coût.

## Modèle documentaire

Une source acquise produit :

- un artefact brut local non versionné ;
- un manifeste d’acquisition (URL canonique, horodatage, statut HTTP, type MIME, empreinte SHA-256) ;
- un document normalisé conforme à `schemas/normalized_document.schema.json` ;
- des chunks conformes à `schemas/retrieval_chunk.schema.json`.

Chaque chunk hérite des restrictions d’audience et de validité de sa source. La citation utilise `source_id`, `section_id`, le titre et l’URL canonique, jamais une URL inventée par le modèle.

## Stratégie d’abstention

Le système s’abstient si au moins une condition critique est vraie : champ essentiel illisible, contradictions entre OCR et validation, aucun passage admissible, citation absente, audience de la source incompatible ou source marquée dégradée/non valide.

## Déploiement progressif

Le premier exécutable est local. Docker arrive lorsque le pipeline minimal fonctionne et possède des tests. CI exécute validation des schémas, tests unitaires et jeu d’évaluation réduit. L’observabilité et le cloud viennent après obtention d’une référence locale mesurée.

