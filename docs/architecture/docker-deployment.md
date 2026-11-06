# Conteneurisation locale de QraLia

## Périmètre

Cette configuration exécute deux conteneurs :

- `api` : FastAPI, OCR, validation, retrieval et génération ;
- `web` : bundle React servi par Nginx, avec proxy `/api` vers FastAPI.

Ollama continue de s'exécuter sur la machine hôte. Les chunks officiels sont
montés en lecture seule et les modèles Hugging Face sont conservés dans un
volume Docker. Aucune facture et aucun secret ne sont copiés dans les images.

## Prérequis

1. Installer et démarrer Docker Desktop.
2. Démarrer Ollama sur la machine hôte.
3. Vérifier que `qwen3.5:4b` est disponible avec `ollama list`.
4. Vérifier la présence de
   `data/processed/onee-regler-facture-residentiel/chunks.jsonl`.

## Construction et démarrage

```powershell
docker compose build
docker compose up -d
docker compose ps
```

L'interface est disponible sur `http://127.0.0.1:8080`. La documentation
FastAPI reste accessible sur `http://127.0.0.1:8000/docs`.

Au premier appel RAG, les modèles publics Qwen d'embedding et de reranking
peuvent être téléchargés dans le volume `qralia_huggingface-cache`. Les appels
suivants réutilisent ce cache. Pour interdire tout téléchargement, démarrer
avec `QRALIA_HF_LOCAL_FILES_ONLY=true` après avoir prérempli ce volume.

## Vérifications

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/ready

.\.venv\Scripts\python.exe scripts\docker_smoke_test.py `
  output\pdf\qralia_facture_onee_residentielle_synthetique.pdf
```

Le smoke test vérifie les deux endpoints de santé, téléverse la facture,
contrôle que l'extraction est valide, puis supprime la session créée.

## Tests continus

Le workflow `.github/workflows/docker.yml` exécute trois contrôles successifs :

1. les tests Python déterministes, sans téléchargement ni appel de modèle ;
2. l'installation reproductible du frontend, son contrôle TypeScript et son build ;
3. la construction des deux images, puis un smoke test de la pile complète.

Le smoke test démarre FastAPI et Nginx avec l'IA désactivée, fabrique une
facture entièrement synthétique, vérifie `/health` et `/ready`, envoie le PDF
à l'OCR, contrôle que la validation aboutit, puis supprime la facture. Les logs
des conteneurs sont affichés même en cas d'échec. Ollama et les modèles Qwen ne
sont volontairement pas requis dans la CI : leurs évaluations lourdes restent
des expériences versionnées séparées. Comme le corpus de travail est ignoré par
Git, la CI monte uniquement `tests/fixtures/onee_chunks.jsonl`, un passage
minimal sans donnée utilisateur destiné au contrôle de disponibilité.

## Diagnostic

```powershell
docker compose logs -f api
docker compose logs -f web
```

Si Ollama est inaccessible depuis le conteneur, vérifier qu'il écoute sur une
interface joignable depuis Docker Desktop. L'URL par défaut est
`http://host.docker.internal:11434/api/chat` et peut être remplacée par la
variable `QRALIA_OLLAMA_URL`.

## Arrêt

```powershell
docker compose down
```

Ne pas ajouter `-v` sauf si la suppression du cache de modèles est souhaitée.

## Limites actuelles

- Le stockage des factures est en mémoire : un seul worker Uvicorn est lancé.
- Un redémarrage de `api` invalide les identifiants de session.
- `/ready` valide l'existence du pipeline mais ne charge pas les modèles à
  l'avance ; le premier appel RAG reste plus lent.
- Cette pile locale ne fournit encore ni HTTPS, ni authentification, ni
  registre d'images, ni déploiement cloud.

