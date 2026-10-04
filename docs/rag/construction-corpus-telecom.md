# Construction du corpus télécom

## Périmètre

Le corpus ajoute 25 chunks officiels sur la facturation télécom marocaine :

- Maroc Telecom : 7 règles contractuelles (facturation, échéance, erreur,
  prélèvement, retard, suspension et résiliation) ;
- Orange Maroc : 10 règles (consultation, délai et huit canaux de paiement) ;
- inwi : 6 règles (cinq canaux de paiement et la procédure inwi money) ;
- ANRT : 2 règles de réclamation et de recours.

Les pages d'assistance peu liées aux factures et les pages purement
transactionnelles sans texte explicatif ne sont pas ingérées.

## Architecture de routage 0.3.0

Chaque chunk porte cinq dimensions indépendantes :

```text
domain -> provider_id -> service_type -> service_zone_id -> audiences
```

Exemple Orange :

```json
{
  "domain": "telecommunications",
  "provider_id": "orange_ma",
  "service_type": "consumer_postpaid",
  "service_zone_id": "morocco_national",
  "audiences": ["residential"]
}
```

Cette structure empêche une question Orange de récupérer une règle ONEE,
Redal, IAM ou inwi même si les textes contiennent tous « facture » et
« paiement ». `source_role` distingue en plus une règle d'opérateur, un
partenaire de paiement, une règle contractuelle et le régulateur.

## Parsing fidèle

Les cinq pages téléchargeables sont archivées avec URL, horodatage, statut
HTTP, type MIME et SHA-256. La page IAM est protégée par Cloudflare : seuls les
passages contractuels rendus dans le navigateur et revus ont été archivés, avec
la méthode indiquée dans son manifeste.

Le parseur normalise uniquement les espaces. Chaque citation attendue doit être
présente dans l'archive ; sinon le pipeline échoue et exige une revue humaine.
Le nombre attendu par source est également contrôlé avant l'écriture.

## Sorties

Six `document.json` conservent une provenance par page. Le fichier agrégé
`data/processed/telecom/chunks.jsonl` contient exactement 25 lignes. Les
archives et sorties restent ignorées par Git ; le registre, le code, les
schémas et les tests sont versionnés.

Le corpus actif complet contient maintenant 39 chunks : 5 ONEE, 9 Redal et 25
télécom. Il ne faut pas évaluer la recherche télécom avec les anciens jeux
ONEE/Redal : un jeu télécom versionné doit être écrit avant toute optimisation.
