# Construction du premier corpus ONEE

## Objectif

Le module `src/qralia/rag/onee_corpus.py` réalise deux transformations distinctes :

```text
archive HTML brute -> document normalisé -> chunks de recherche
```

Il travaille uniquement sur un fichier local déjà approuvé. Il ne contacte pas
le site ONEE et n'utilise aucun LLM.

## Installation locale

Depuis la racine du dépôt :

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Exécution

```powershell
.\.venv\Scripts\python.exe -m qralia.rag.onee_corpus `
  --input "data/raw/onee-regler-facture-residentiel/page-20260929-212757.html" `
  --document-output "data/processed/onee-regler-facture-residentiel/document.json" `
  --chunks-output "data/processed/onee-regler-facture-residentiel/chunks.jsonl" `
  --retrieved-at "2026-09-29T19:28:25Z" `
  --http-status 200 `
  --verified-at "2026-09-29" `
  --validity-status unknown
```

Les valeurs d'acquisition doivent provenir du téléchargement observé. Le premier
téléchargement manuel n'a pas conservé ses en-têtes HTTP : son statut et son type
MIME devront être capturés automatiquement par le futur collecteur.

## Étape 1 : normalisation

Le parseur décode le fichier en `iso-8859-1`, localise `td#contenu`, ignore le
reste de la page, puis exige un titre, une introduction et exactement quatre
moyens de paiement dans l'ordre attendu. Une structure différente provoque une
erreur et demande une revue humaine.

Le document produit contient la provenance, l'empreinte SHA-256, le fournisseur
concerné (`provider_id: onee`), le service (`electricity`), la classification
résidentielle et cinq sections. L'éditeur de la source et le fournisseur sont
deux notions séparées : une source publiée
par le CMI peut, par exemple, décrire une règle applicable à la RADEEF. Le
document est validé avec `schemas/normalized_document.schema.json`.

## Étape 2 : création des chunks

Une section normalisée devient un chunk autonome. Chaque chunk recopie le texte
sans reformulation et ajoute les métadonnées nécessaires aux filtres, citations
et audits, dont `domain`, `provider_id`, `service_type`, `service_zone_id` et
`source_role`. La récupération doit obligatoirement filtrer le fournisseur, la zone et l'audience
avant de calculer les scores. Chaque chunk est
validé avec `schemas/retrieval_chunk.schema.json`, puis écrit sur une ligne du
fichier JSONL.

## Revue humaine obligatoire

La première exécution utilise `--validity-status unknown`. Il faut comparer les
cinq sections au HTML brut. Si elles sont fidèles, relancer la commande avec :

```text
--validity-status active
```

Seuls les chunks `active` pourront ensuite entrer dans l'index RAG.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Les tests utilisent un HTML synthétique et vérifient notamment que la navigation
est ignorée, que quatre moyens de paiement sont extraits et que cinq chunks sont
créés.
