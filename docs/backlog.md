# Backlog par phases

## Phase 0 — Fondation (en cours)

- [x] Définir le périmètre du MVP.
- [x] Créer l’architecture et les contrats de données.
- [x] Créer le registre initial des sources.
- [x] Créer 30 cas d’évaluation initiaux.
- [ ] Relire en binôme les sources et les réponses de référence.
- [x] Initialiser le dépôt Git indépendant sur la branche `main`.
- [ ] Protéger la branche principale lors de la création du dépôt distant.
- [ ] Ajouter les commandes de validation locale.

**Sortie :** documentation cohérente, schémas valides, corpus proposé mais non ingéré.

## Phase 1 — Sprint RAG ONEE

- [ ] Approuver la liste blanche des sources.
- [ ] Acquérir uniquement les pages approuvées avec horodatage et SHA-256.
- [ ] Nettoyer navigation, actualités et éléments sans rapport.
- [ ] Découper par sections sémantiques, sans chevauchement arbitraire au départ.
- [ ] Implémenter une baseline BM25.
- [ ] Comparer au moins deux embeddings multilingues sur le même jeu.
- [ ] Tester recherche lexicale, sémantique et hybride.
- [ ] Ajouter le reranking seulement après mesure d’un gain.
- [ ] Produire réponses citées et abstentions.
- [ ] Mesurer Recall@k, MRR, précision des citations et fidélité.

**Sortie :** rapport reproductible avec métriques réellement obtenues.

## Phase 2 — Factures synthétiques et OCR

- [ ] Concevoir 20 à 30 factures fictives variées, sans copie de données réelles.
- [ ] Définir la vérité terrain au niveau champ.
- [ ] Comparer les options OCR/VLM locales et hébergées autorisées.
- [ ] Conserver texte, valeur normalisée, coordonnées et confiance.
- [ ] Tester rotations, flou, ombres, faible lumière et pages multiples.
- [ ] Implémenter les validateurs de montants, dates et identifiants.
- [ ] Atteindre 0 % de modification/hallucination des valeurs critiques dans le jeu retenu.

**Sortie :** extraction mesurée, erreurs classées, cas difficiles documentés.

## Phase 3 — Application intégrée

- [ ] API FastAPI et schémas Pydantic.
- [ ] Interface minimale d’envoi et de question.
- [ ] Fusion contrôlée des faits documentaires et du RAG.
- [ ] Citations visibles et avertissement de confiance.
- [ ] Tests de bout en bout.
- [ ] Docker et guide de lancement local.

## Phase 4 — Qualité et livraison

- [ ] CI : formatage, tests, validation des données, build Docker.
- [ ] Tests de sécurité et de non-régression.
- [ ] Mesure de latence et coût par requête.
- [ ] Démonstration enregistrée et fiche d’architecture.
- [ ] Déploiement limité avec données synthétiques.

## Phase 5 — Extensions conditionnelles

- [ ] Entrée/sortie vocale darija.
- [ ] Observabilité LLM.
- [ ] Mémoire et rappels avec consentement explicite.
- [ ] Cercle de confiance.
- [ ] Orchestration agentique limitée si elle résout un besoin mesuré.

Ces extensions ne commencent qu’après un MVP évalué et stable.
