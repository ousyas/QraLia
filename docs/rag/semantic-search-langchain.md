# Recherche sémantique locale avec LangChain et Qwen3

## Séparation des moteurs

La baseline BM25 écrite à la main reste inchangée. La recherche sémantique est
un deuxième moteur indépendant, évalué sur les mêmes chunks et les mêmes cas :

```text
question -> BM25 manuel ---------> classement lexical
question -> Qwen3 + LangChain ---> classement sémantique
```

Cette séparation permet une comparaison honnête avant toute fusion hybride.

## Composants LangChain utilisés

- `Embeddings` : contrat commun `embed_documents` / `embed_query` ;
- `Document` : texte indexé accompagné des métadonnées de provenance ;
- `InMemoryVectorStore` : index local calculant la similarité cosinus.

`Qwen3LangChainEmbeddings` encode les documents sans prompt et les questions
avec le prompt `query` du modèle. Tous les vecteurs sont normalisés. Le texte
encodé concatène `heading_path` et le texte officiel ; les métadonnées ne sont
pas ajoutées au contenu, mais servent au routage et aux citations.

## Filtrage de sécurité

La fonction passée au vector store exige simultanément :

```text
validity_status == active
distributor_id == route demandée
service_zone_id == route demandée
audience présente dans metadata.audiences
```

Un passage hors route n'est donc pas admissible, même si sa similarité serait
forte. Chaque cas d'évaluation porte sa propre route. Une question visant une
route sans chunks produit donc zéro candidat et doit conduire à l'abstention.
Le nom du distributeur n'a pas besoin d'apparaître dans la question si le
contexte applicatif a déjà déterminé la route.

## Installation

```powershell
.venv\Scripts\python.exe -m pip install -e ".[dev,semantic]"
```

Le premier lancement télécharge les poids publics du modèle dans le cache local
Hugging Face. Aucun chunk QraLia n'est envoyé à un service externe.

Pour utiliser un GPU NVIDIA, PyTorch doit être installé avec une roue CUDA, et
`torch.cuda.is_available()` doit retourner `True`. Le paramètre `--device cuda`
ne peut pas transformer une installation PyTorch CPU en installation GPU.

## Évaluation ONEE

```powershell
.venv\Scripts\python.exe -m qralia.rag.semantic_evaluation `
  --chunks data/processed/onee-regler-facture-residentiel/chunks.jsonl `
  --evaluation data/evaluation/onee_retrieval_questions.v0.1.jsonl `
  --output outputs/evaluation/onee-semantic-qwen3-v0.1.json `
  --distributor-id onee `
  --service-zone-id onee_be_national `
  --device cpu `
  --top-k 5
```

Plusieurs fichiers `--chunks` peuvent être fournis. Le filtre conserve ensuite
seulement la route autorisée pour chaque recherche.

## Évaluation Redal sur GPU

```powershell
.venv\Scripts\python.exe -m qralia.rag.semantic_evaluation `
  --chunks data/processed/onee-regler-facture-residentiel/chunks.jsonl `
  --chunks data/processed/redal/chunks.jsonl `
  --evaluation data/evaluation/redal_retrieval_questions.v0.1.jsonl `
  --output outputs/evaluation/redal-semantic-qwen3-gpu-v0.1.json `
  --device cuda `
  --batch-size 2 `
  --max-seq-length 512 `
  --top-k 5
```

On ne fournit volontairement aucun filtre global de route ici : chaque cas
porte `distributor_id` et `service_zone_id`. Cela permet de conserver dans le
même jeu les hard negatives SRM-FM, SRM-CS, SRM-TTA et SRM-MS. Les filtres CLI
restent disponibles pour exécuter seulement une route précise.

## Reproductibilité

Le nom et la révision du modèle sont enregistrés dans le rapport. Le seuil
d'abstention est choisi uniquement sur le split `calibration`, puis mesuré sur
le split `test`, comme pour BM25. Le rapport conserve aussi le temps de
chargement du modèle, la durée d'indexation et les latences moyenne, p95 et
maximale des recherches.

## Première mesure ONEE

Sur les 35 cas du jeu `onee_retrieval_questions.v0.1.jsonl`, Qwen3 obtient
Recall@5 `1.000` et MRR `0.886`. BM25 avec lexique Darija obtient respectivement
`0.844` et `0.852`. L'abstention sémantique reste moins bonne (`0.591` de
balanced accuracy contre `0.619`) : les hard negatives demeurent donc
indispensables et le seuil ne doit pas être choisi sur le split de test.

## Première mesure Redal sur GPU

Sur les 21 cas du split de test Redal, Qwen3 obtient Recall@5 `0.889`, Binary
Recall@5 `0.917` et MRR `0.757`. La balanced accuracy d'abstention vaut `0.625`.
Sur la GTX 1650 Ti, l'indexation des 14 chunks prend environ `2.56 s`, puis une
question prend `129 ms` en moyenne et `145 ms` au p95. Le chargement initial du
modèle et de CUDA (`28.43 s` sur la mesure conservée) est un coût de démarrage,
pas la latence de chaque question. La limite de 512 tokens ne tronque aucun des
14 chunks : le plus long en compte 327 avec le tokenizer du modèle.

Les tests n'utilisent pas Qwen : un faux modèle d'embeddings déterministe teste
la plomberie LangChain sans réseau, GPU ni téléchargement lourd.
