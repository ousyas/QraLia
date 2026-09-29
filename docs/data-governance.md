# Gouvernance des données

## Données autorisées maintenant

- pages publiques officielles figurant dans la liste blanche approuvée ;
- factures et images entièrement synthétiques ;
- annotations créées par le binôme, sans information personnelle réelle.

## Données interdites dans le développement

- tout parcours ou indexation d’un dossier personnel ;
- relevés bancaires, pièces d’identité, passeports, données sociales, scolaires, familiales, médicales ou de revenus ;
- téléversement d’un document réel vers un service externe ;
- secrets, jetons, mots de passe et identifiants clients dans Git, logs ou traces.

## Garde-fous

- acquisition par liste blanche d’URL, jamais par exploration d’un disque utilisateur ;
- dossiers `data/raw`, `data/processed`, `data/indexes` et `data/user_documents` ignorés par Git ;
- exemples publics/synthétiques explicitement marqués ;
- journaux sans contenu documentaire brut ;
- suppression simple des artefacts reconstruisibles ;
- revue de confidentialité avant toute expérimentation avec documents réels.

## Conditions préalables aux documents réels

Avant toute utilisation : consentement explicite, finalité documentée, minimisation, chiffrement en transit et au repos, contrôle d’accès, durée de conservation, suppression vérifiée, politique du fournisseur et procédure d’incident.

Pour un document médical, QraLia peut lire ou reformuler fidèlement mais ne donne aucun diagnostic ni conseil médical.

