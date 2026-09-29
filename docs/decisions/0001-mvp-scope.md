# ADR-0001 — Limiter le MVP aux factures ONEE

- Statut : accepté
- Date : 2026-09-29

## Décision

Le premier MVP couvre la lecture de factures ONEE synthétiques et les questions associées. Il sépare extraction documentaire, validation déterministe et récupération de règles officielles.

## Raisons

Un domaine étroit permet de construire des jeux de vérité terrain, mesurer les erreurs critiques et comparer les composants. Il rend également explicable en entretien le rôle respectif de l’OCR, du RAG et du LLM.

## Conséquences

- pas de dossiers administratifs généraux au départ ;
- aucune donnée personnelle réelle dans le développement ;
- pas de multi-agent tant qu’un pipeline simple n’a pas de baseline ;
- les pages professionnelles ne répondent pas aux questions des particuliers sans filtre explicite ;
- toute extension de périmètre crée un nouvel ADR et de nouveaux tests.

