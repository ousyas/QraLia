# FastAPI et React — première intégration

## Objectif

Cette version expose une boucle locale complète et prudente : téléversement
d'une facture synthétique, extraction, validation, consultation, question et
suppression. Elle n'est pas encore une version publique de production.

## Contrats HTTP

- `GET /health` confirme seulement que le processus HTTP répond.
- `GET /ready` contrôle séparément l'API, Tesseract, le corpus officiel et la
  configuration du chemin RAG/LLM.
- `POST /v1/invoices` accepte PDF, PNG ou JPEG, avec confirmation explicite
  que le document est synthétique.
- `GET /v1/invoices/{invoice_id}` retourne les champs et la validation.
- `POST /v1/invoices/{invoice_id}/questions` répond depuis la facture, le
  corpus officiel, les deux, ou s'abstient.
- `DELETE /v1/invoices/{invoice_id}` détruit immédiatement la session.

Les octets téléversés sont écrits dans un fichier temporaire isolé, lus par
l'OCR puis supprimés dans un bloc `finally`. Le magasin applicatif ne conserve
que l'extraction structurée pendant 30 minutes et possède une limite de 25
documents.

## Réponse aux questions

Le chemin IA local est maintenant branché derrière le même contrat HTTP :

1. Qwen3.5-4B décompose la demande et propose seulement une source probable ;
2. les champs OCR validés sont comparés à un catalogue de champs, jamais à
   leurs valeurs protégées ;
3. les règles officielles passent par BM25 normalisé + Qwen3-Embedding-0.6B,
   Weighted RRF puis Qwen3-Reranker-0.6B ;
4. l'EvidenceRouter confirme ou refuse la proposition du routeur ;
5. Qwen3.5-4B formule une réponse à partir des seules preuves retenues ;
6. le code valide citations, alphabet et nombres. Toute sortie douteuse est
   remplacée par une réponse extractive déterministe.

Le RAG résidentiel reste désactivé pour une facture reconnue comme
professionnelle. Une question portant sur un montant validé peut encore être
traitée, mais aucune règle résidentielle n'est appliquée à ce document.

Les modèles de retrieval sont chargés paresseusement et restent en mémoire.
Le premier appel est donc sensiblement plus lent. `QRALIA_AI_ENABLED=false`
réactive le chemin déterministe pour le diagnostic ou les tests HTTP.

## Lancement local

```powershell
python -m pip install -e ".[api,invoice,semantic,generation,dev]"
ollama serve
ollama pull qwen3.5:4b
qralia-api
```

Dans un second terminal :

```powershell
cd frontend
npm install
npm run dev
```

Swagger est disponible sur `http://127.0.0.1:8000/docs` et l'interface sur
`http://127.0.0.1:5174`.

