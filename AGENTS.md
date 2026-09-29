# Instructions de projet QraLia

## Mission

QraLia est un projet pédagogique réalisé par Oussama et Meryeme. Le premier MVP aide à lire et comprendre des factures d’électricité ONEE à partir de documents synthétiques, avec extraction fidèle des champs, validation déterministe et réponses fondées sur des sources officielles citées.

## Priorités

1. Avancer par petits incréments mesurables et explicables en entretien.
2. Maintenir une séparation stricte entre les faits du document utilisateur et les règles du corpus RAG.
3. Ne jamais laisser un LLM modifier ou inventer un montant, une date ou un identifiant.
4. Construire et versionner l’évaluation avant d’optimiser les modèles.
5. Préférer un pipeline simple et testé à une architecture multi-agent prématurée.
6. Expliquer pédagogiquement chaque choix afin qu’Oussama et Meryeme comprennent tout le système.

## Données et sécurité

- Ne jamais parcourir, indexer ou téléverser un dossier personnel.
- Utiliser uniquement des pages publiques officielles approuvées et des factures entièrement synthétiques.
- Ne jamais envoyer de document réel à un service externe sans politique de confidentialité explicitement validée.
- Les dossiers locaux de données brutes, documents utilisateurs et index restent ignorés par Git.
- Pour un document médical futur, lire fidèlement sans donner de diagnostic ni conseil médical.

## État de référence

- Architecture, backlog, responsabilités et gouvernance documentés.
- Six sources ONEE candidates enregistrées, mais aucune téléchargée ni ingérée.
- Trente cas d’évaluation trilingues disponibles et validés par leur schéma.
- Le corpus résidentiel principal doit exclure les règles destinées aux professionnels ou aux collectivités.
- La prochaine étape est une revue humaine de la liste blanche, puis une acquisition contrôlée avec horodatage et SHA-256.

Lire `README.md` et `docs/conversation-handoff.md` avant une modification importante.

