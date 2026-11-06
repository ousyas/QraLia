# ADR-0002 — Étendre le MVP aux factures d'abonnement télécom

- Statut : accepté
- Date : 2026-11-03

## Décision

QraLia lit aussi les factures d'abonnement postpayé de Maroc Telecom, Orange
Maroc et inwi. Chaque facture lue porte un `provider_id` (`onee`, `iam`,
`orange_ma`, `inwi`) qui fixe son contexte de routage RAG : domaine, service,
zone et audience. Le registre `qralia.domain.providers` est la seule source de
ces correspondances.

## Raisons

- Les abonnements sont la deuxième famille de factures la plus courante pour
  le public visé.
- Le corpus officiel télécom (25 passages) et son routage par fournisseur
  existent déjà ; il manquait seulement la lecture du document.
- Un registre unique évite de dupliquer « onee » en dur dans l'OCR, la
  validation, l'API et la conversation.

## Conséquences

- Le contrat `ExtractedInvoice` passe en 0.2.0 : `provider_id` et trois champs
  télécom facultatifs (`plan_name`, `line_number`, `out_of_plan_amount`). Les
  extractions 0.1.0 restent valides et sont lues comme ONEE.
- Les champs critiques dépendent du fournisseur : le tarif n'est critique que
  pour l'ONEE, où il décide du périmètre résidentiel.
- Le taux de TVA attendu est de 14 % pour l'électricité et 20 % pour le
  télécom ; un écart reste un avertissement.
- Un document dont le fournisseur n'est pas reconnu, ou en mentionne plusieurs,
  n'est pas attribué au hasard : il est signalé comme non pris en charge.
- Les factures de test sont entièrement synthétiques, sans logo, et portent
  une mention « SPÉCIMEN — facture synthétique ». Aucune facture réelle
  d'opérateur n'est utilisée.
- L'évaluation OCR est mesurée séparément par fournisseur, avant toute
  conclusion globale.
