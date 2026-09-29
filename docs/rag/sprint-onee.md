# Sprint RAG ONEE

## But

Construire une baseline de récupération et de citation sur un corpus officiel minuscule avant de choisir une base vectorielle ou un framework d’orchestration.

## État des sources

Le registre machine lisible est `data/sources/onee_sources.v0.1.json`. Vérification effectuée le 29 septembre 2026 :

1. **Guide interactif de facture** — pertinent pour les noms et positions des champs ; accès direct instable, donc acquisition à confirmer manuellement.
2. **Comprendre votre facture** — accessible ; explications de rubriques, mais page de navigation ancienne et audience à confirmer.
3. **Règlement et retard** — accessible ; contient délais et séquence de coupure, audience affichée « collectivités locales », donc exclue des réponses résidentielles tant que la portée n’est pas confirmée.
4. **Facturation estimée MT/BT** — accessible ; vise les clients industriels, agricoles et tertiaires, donc hors index résidentiel principal.
5. **FAQ Wakalaty** — contenu encore indexé, mais URL directe en 404 ; ne pas ingérer avant de trouver une URL officielle active ou une confirmation de maintien.
6. **Régler votre facture — résidentiel** — approuvée ; HTML brut archivé localement, document normalisé et cinq chunks validés humainement le 29 septembre 2026.

Une seule source a été acquise. Les cinq autres n’ont pas été téléchargées et conservent leur décision individuelle dans le registre.

## Protocole d’acquisition

Pour chaque source approuvée :

1. vérifier domaine, URL finale, statut HTTP, type MIME et audience ;
2. archiver la réponse brute localement dans `data/raw` ;
3. calculer SHA-256 et enregistrer l’horodatage UTC ;
4. retirer navigation, actualités et éléments sans rapport tout en conservant les titres ;
5. produire un document conforme au schéma ;
6. faire une comparaison humaine entre page et texte normalisé ;
7. attribuer un statut de validité et une date de prochaine vérification ;
8. n’indexer que les documents `active` et compatibles avec l’audience de la requête.

Le collecteur respecte les conditions du site et utilise une fréquence faible. Les captures ne sont pas republiées sans vérifier les droits applicables.

## Découpage initial

- une section cohérente par chunk lorsque sa taille le permet ;
- titres hiérarchiques recopiés dans les métadonnées ;
- listes conservées ensemble ;
- pas de découpage au milieu d’une règle ou d’une séquence de délai ;
- identifiant déterministe dérivé de `source_id`, `section_id` et version ;
- texte original conservé, sans traduction automatique dans l’index de référence.

## Baselines et expériences

1. BM25 seul ;
2. embeddings multilingues seuls ;
3. fusion hybride ;
4. hybride + reranker, uniquement si 3 laisse une marge d’amélioration.

Toutes les variantes utilisent exactement le même corpus, les mêmes filtres et le même jeu d’évaluation. Les choix de bibliothèque, modèle et stockage seront consignés après mesure.

## Métriques

- Hit Rate/Recall@1, @3 et @5 ;
- MRR pour le rang du premier passage pertinent ;
- précision des citations ;
- fidélité aux passages ;
- exactitude de l’abstention ;
- conservation exacte des chaînes de montant et date ;
- latence médiane et p95 ;
- coût, si un service payant est utilisé.

## Critères de fin du sprint

- chaque source indexée possède audience, validité, empreinte et date de vérification ;
- les 30 cas initiaux sont relus par les deux membres ;
- aucune réponse destinée à un particulier ne cite une règle professionnelle non signalée ;
- les résultats sont produits par une commande reproductible ;
- les erreurs sont analysées par catégorie, sans présenter une métrique non mesurée comme acquise.
