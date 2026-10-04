# Construction du corpus Redal

## Périmètre

Deux pages publiques officielles sont utilisées : « E-paiement » et
« Questions fréquentes ». L’acquisition et la transformation restent séparées :

```text
URL approuvée -> HTML brut + manifeste -> document normalisé -> chunks JSONL
```

Le manifeste conserve l’URL demandée et finale, le statut HTTP, le type MIME,
l’horodatage, la taille et le SHA-256. Les fichiers bruts et transformés restent
ignorés par Git ; le code, les schémas et les tests sont versionnés.

## Parsing déterministe

`src/qralia/rag/redal_corpus.py` cible seulement :

- les blocs `tabs-item` de la page de paiement ;
- les blocs `accordion-item` de la FAQ.

Le menu, les actualités, les scripts et le pied de page ne sont pas ingérés.
Les URL externes de connexion aux banques sont retirées du texte indexé, tandis
que les noms des services restent présents. Si un onglet ou une question
attendue disparaît, le traitement échoue et impose une revue humaine.

## Découpage obtenu

La page de paiement produit cinq chunks : paiement Redal par carte, banque en
ligne, mobile JIWAR, prélèvement et GAB. La FAQ produit quatre chunks : relève
et auto-relève, lecture de facture, facturation mensuelle, délai et suspension.

Chaque chunk porte notamment :

- `domain: utilities` : domaine eau/électricité ;
- `provider_id: srm_rsk` : opérateur régional actuel ;
- `service_type: multi_utility` : eau et électricité ;
- `service_zone_id: rabat_sale_ex_redal` : périmètre exact de la source ;
- `legacy_operator: Redal` : nom visible sur la page historique ;
- `source_role: operator_primary` : source officielle de l’opérateur ;
- `audiences: [residential]` : public auquel la règle s’applique.

BM25 filtre distributeur, zone et audience avant de calculer les scores.

## Reproduction

```powershell
\.venv\Scripts\python.exe -m qralia.rag.redal_corpus `
  --payment-html data/raw/redal-e-paiement/page-20260930.html `
  --payment-manifest data/raw/redal-e-paiement/acquisition-20260930.json `
  --faq-html data/raw/redal-faq/page-20260930.html `
  --faq-manifest data/raw/redal-faq/acquisition-20260930.json `
  --output-dir data/processed `
  --verified-at 2026-09-30 `
  --validity-status active
```

Le résultat agrégé contient neuf lignes dans
`data/processed/redal/chunks.jsonl`. Les deux documents normalisés sont gardés
séparément afin que chaque chunk conserve une provenance et une citation exactes.
