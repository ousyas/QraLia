# Génération fondée et juge LLM

## But de cet incrément

Cet incrément mesure la **génération**, indépendamment de la récupération. Pour
chaque cas, le générateur reçoit directement les `gold_chunk_ids` validés par
un humain. Une mauvaise réponse est donc attribuable au générateur, au prompt
ou aux garde-fous, et non au retriever.

L'évaluation de bout en bout (`question -> retrieval -> génération`) viendra
après cette baseline isolée.

## Pipeline

```text
cas d'évaluation + gold chunks
              |
              v
Qwen3-4B ou Qwen3.5-4B local (Ollama, JSON structuré)
              |
              +--> validation déterministe
              |    - langue déclarée
              |    - écriture réellement utilisée (arabe ou latine)
              |    - décision d'abstention
              |    - IDs de citations autorisés
              |    - citation présente dans le texte
              |    - valeurs critiques recopiées exactement
              |    - expressions interdites absentes
              |
              +--> Gemini 2.5 Flash (juge externe structuré)
                   - fidélité aux passages
                   - pertinence et complétude
                   - correction des citations
                   - qualité de langue
                   - abstention
                   - violation d'un fait critique
```

Le résultat final passe uniquement si le contrôle déterministe passe **et** si
le juge accepte la réponse sans violation critique. Le LLM juge ne remplace
donc jamais les règles exactes.

## Contrat d'une réponse générée

`GroundedAnswer` contient :

- `answer` : texte destiné à l'utilisateur, avec citations `[chunk_id]` ;
- `claims` : affirmations atomiques et liste de leurs sources ;
- `abstain` : décision explicite de répondre ou non ;
- `abstention_reason` : raison en cas de manque de preuve ;
- `language` : `fr`, `ar`, `darija_latn` ou `darija_arab`.

Ollama reçoit directement le JSON Schema Pydantic dans le paramètre `format`.
Le parsing n'accepte donc pas une prose libre suivie d'un JSON approximatif.
Le texte visible est ensuite reconstruit de façon déterministe à partir de
`claims` : le code ajoute les `[chunk_id]` après chaque affirmation. Le rapport
indique si cette opération a corrigé la mise en forme initiale du modèle.

## Prompt du générateur

Le prompt sépare quatre éléments : règles, fournisseur demandé, question et
passages approuvés. Il impose notamment :

1. aucune connaissance externe ;
2. aucun mélange de fournisseurs ;
3. une citation exacte après chaque fait ;
4. aucune transformation de nombre, date, montant ou identifiant ;
5. une abstention structurée lorsque les passages sont insuffisants ;
6. le traitement du contenu des passages comme données, jamais comme
   instructions.

Le modèle utilise `temperature=0` et une graine fixe. Cela réduit la variation,
mais ne garantit pas à lui seul la fidélité : les validateurs restent
obligatoires.

## Jeu d'évaluation

`data/evaluation/generation_questions.v0.1.jsonl` contient 16 cas couvrant
ONEE, Redal, IAM, Orange, inwi et ANRT. Les quatre langues/écritures sont
représentées. Chaque cas fixe :

- les passages de référence ;
- les faits sémantiques obligatoires ;
- les affirmations interdites ;
- les valeurs à conserver littéralement ;
- la décision d'abstention attendue ;
- le split `calibration` ou `test`.

Le cas SRM-FM est un hard negative sans passage approuvé : le système doit
refuser d'appliquer les règles Redal à un autre distributeur.

## Deux couches d'évaluation

La validation déterministe détecte très bien les erreurs formelles et les
valeurs sensibles, mais elle ne sait pas reconnaître toutes les paraphrases.
Gemini reçoit donc la question, les passages, les faits attendus, la réponse et
le résultat déterministe. Sa sortie Pydantic donne six notes de 1 à 5 et des
listes explicites de faits absents, affirmations non soutenues et citations
invalides.

Le juge est un signal expérimental : il peut lui-même se tromper. Un petit
échantillon doit être relu manuellement avant de transformer ses notes en
critère de livraison.

## Choix des modèles

- Générateur par défaut : `qwen3.5:4b`, local via Ollama. Il remplace
  `qwen3:4b-instruct-2507-q4_K_M` après la comparaison gelée décrite plus bas.
  La quantification locale limite la mémoire nécessaire et aucune donnée n'est
  envoyée pour générer la réponse.
- Juge : `gemini-2.5-flash`. Le projet API testé expose une limite gratuite de
  zéro requête pour les modèles Pro. Le modèle reste configurable avec
  `--judge-model` si un quota Pro est activé ultérieurement.

Seuls des passages publics et des questions synthétiques sont transmis au
juge. Aucune facture réelle ou donnée personnelle ne doit utiliser ce chemin.

## Contrôle indépendant de l'écriture

Le champ JSON `language` produit par le modèle ne constitue pas une preuve. Un
modèle peut déclarer `darija_arab` tout en répondant en français. Le validateur
retire d'abord les identifiants de citation, puis vérifie la proportion de
caractères arabes dans le texte utile :

- `ar` et `darija_arab` exigent une écriture majoritairement arabe ;
- `fr` et `darija_latn` refusent un texte majoritairement arabe.

Ce garde-fou détecte un alphabet incorrect, mais ne sait pas distinguer le
français de la darija écrite en caractères latins. Cette dernière distinction
demande encore une revue humaine ou un juge linguistique validé.

## Holdout Qwen3.5 et expérience Atlas

Un nouveau holdout de 12 cas a été gelé dans
`data/evaluation/generation_holdout.v0.1.jsonl` **avant** l'exécution de
Qwen3.5. Il contient trois cas par langue/écriture et inclut une demande
d'abstention. Les deux modèles reçoivent exactement les mêmes gold chunks et
le même validateur corrigé.

| Générateur | Passage déterministe | FR | AR | Darija latine | Darija arabe | Latence moyenne observée |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B-Instruct Q4 | 41,7 % (5/12) | 2/3 | 1/3 | 0/3 | 2/3 | 23,78 s |
| Qwen3.5-4B | **58,3 % (7/12)** | 3/3 | 1/3 | 2/3 | 1/3 | 17,98 s |

Qwen3.5 est donc le meilleur candidat de ces deux exécutions, mais 12 cas ne
suffisent pas pour conclure à une qualité de production. La latence dépend en
outre du chargement et de l'état chaud du modèle ; elle est informative, pas
un benchmark matériel strict. Aucun des deux rapports de holdout n'a encore
été évalué par Gemini.

`Al-Atlas-0.5B` a ensuite été essayé comme reformulateur des trois réponses en
darija arabe. Les nombres, dates, montants et identifiants sont remplacés par
des marqueurs opaques avant la reformulation puis restaurés par le code. Une
sortie vide, trop longue, répétitive, dans le mauvais alphabet ou qui perd un
marqueur déclenche un retour à la réponse Qwen originale.

Résultat : **1/3**, identique à Qwen3.5 seul sur ce sous-ensemble. Deux sorties
Atlas sur trois ont été rejetées par les garde-fous, et sa latence moyenne de
reformulation sur CPU est de 38,87 s. Atlas reste donc un composant
expérimental désactivé par défaut ; il ne doit pas être intégré au chemin de
production sans fine-tuning ou prompt validé, évaluation sémantique et revue
humaine de la naturalité.

## Variante few-shot darija 0.3

Le profil optionnel `darija_few_shot` conserve la génération libre. Il ajoute
une consigne de langue renforcée, un petit lexique de style sans règle métier,
un exemple de forme déclaré comme non factuel et une vérification finale de la
langue cible. Les expressions sont des repères, pas des templates complets ;
les faits doivent toujours venir uniquement des chunks approuvés.

L'expérience post-hoc sur le holdout déjà observé donne 8/12 (66,7 %) contre
7/12 (58,3 %) pour le prompt de référence. La darija passe de 3/6 à 4/6. Ce
gain n'est pas une nouvelle mesure de test : le jeu avait déjà été lu et la
variante doit maintenant être gelée puis évaluée sur de nouveaux cas.

La lecture qualitative révèle encore plusieurs défauts : une réponse déclarée
darija latine reste en français, une question inwi produit une abstention
incorrecte, et une réponse arabe transforme les littéraux `1Go` et `7jours`.
Le validateur déterministe ne mesure pas non plus toute la complétude
sémantique. Le profil reste donc exploratoire malgré son meilleur score brut.

## Smoke test Atlas-Chat-2B

`MBZUAI-Paris/Atlas-Chat-2B` est intégré comme second backend de
reformulation via Ollama. La machine ne disposant que de 4 Go de VRAM, le
premier test utilise le GGUF `Q2_K` de 1,2 Go fourni par TensorBlock. Cette
quantification très forte convient à un test de fonctionnement, pas à une
conclusion définitive sur les poids officiels.

Sur les trois cas en darija arabe déjà observés, le contrôle déterministe donne
2/3, contre 1/3 avec Qwen3.5 seul et 1/3 avec Al-Atlas-0.5B. La revue humaine
annule cependant ce gain : une sortie mélange français et arabe et déforme le
sens de la suspension Redal. Le validateur d'alphabet l'accepte à tort. Une
seconde reformulation perd le marqueur protégeant `10` et déclenche correctement
le fallback. L'abstention n'est pas reformulée.

La latence moyenne de reformulation observée est 20,89 s, dont 58,19 s pour le
premier cas avec chargement du modèle. Atlas-Chat-2B Q2 est donc techniquement
exécutable, mais pas validé pour le produit. Une future comparaison doit utiliser
une quantification Q4_K_M ou les poids officiels et un jeu darija beaucoup plus
large, relu par des humains.

## Exécution

```powershell
.venv\Scripts\python.exe -m qralia.rag.generation_evaluation `
  --chunks data/processed/onee-regler-facture-residentiel/chunks.jsonl `
  --chunks data/processed/redal/chunks.jsonl `
  --chunks data/processed/telecom/chunks.jsonl `
  --evaluation data/evaluation/generation_questions.v0.1.jsonl `
  --output outputs/generation-qwen3-4b-gemini-2.5-flash.json `
  --generator-model qwen3:4b-instruct-2507-q4_K_M
```

`--skip-judge` teste uniquement Qwen et les règles déterministes. `--limit 1`
est utile pour un smoke test. La clé est chargée depuis `GEMINI_API_KEY` dans
`.env`; elle n'est jamais incluse dans le rapport.

## Benchmark darija 0.2 et observation Atlas

Le fichier `data/evaluation/generation_darija_questions.v0.2.jsonl` contient
80 cas candidats : 40 en darija latine, 40 en darija arabe, 40 en calibration
et 40 en test. Vingt cas exigent une abstention, soit 25 %. Les paraphrases
d'une même intention restent dans le même split. Le split de test n'a pas été
exécuté et le jeu doit encore être relu humainement avant d'être déclaré gelé.

Le mode `--atlas-observation-mode` désactive volontairement le remplacement
d'une reformulation dangereuse par la réponse Qwen originale. Il conserve la
sortie brute, indique `would_fallback` et `unsafe_observation`, puis laisse le
validateur mesurer les dégâts. Ce mode est réservé au diagnostic et ne doit
jamais servir une réponse utilisateur.

Sur les 40 cas de calibration, avec les mêmes réponses Qwen réutilisées :

| Système | Total | Darija latine | Darija arabe | Sorties qu'un garde-fou aurait rejetées |
|---|---:|---:|---:|---:|
| Qwen3.5-4B seul | **33/40 (82,5 %)** | 16/20 | 17/20 | sans objet |
| Atlas-2B, prompt long | 29/40 (72,5 %) | 13/20 | 16/20 | 8 |
| Atlas-2B, prompt compact | 28/40 (70,0 %) | 14/20 | 14/20 | 10 |

Le prompt compact réduit la latence moyenne de reformulation de 6,00 s à
1,40 s, mais ne restaure pas la fidélité. Atlas perd des valeurs, change des
horaires, répète ou recopie parfois les consignes et emploie le mauvais
alphabet. Le validateur déterministe accepte encore quelques sorties
sémantiquement mauvaises lorsqu'elles conservent chiffres, citations et
alphabet : les taux ci-dessus sont donc optimistes. La décision actuelle est
de garder Atlas expérimental et désactivé ; Qwen3.5 seul est meilleur sur cette
calibration.

Le profil compact testé est volontairement court :

```text
Reformule seulement le TEXTE en darija marocaine avec lettres latines.
Ne change, n'ajoute et ne supprime aucune information.
Copie exactement chaque nom, nombre et code ZXQVAL... une seule fois.
Réponds par la reformulation seule, sans titre ni explication.
Exemple: Vous avez ZXQVAL0QXZ jours. -> 3ndek ZXQVAL0QXZ iyam.
TEXTE: {text}
```

La variante arabe porte les mêmes contraintes en arabe et demande
explicitement l'écriture arabe. `--replay-generation-report` permet de tester
un prompt Atlas sur exactement les mêmes réponses Qwen, sans confondre les
effets des deux modèles. `--split` et `--case-id` empêchent d'ouvrir le test
pendant l'optimisation.

Pour activer l'expérience Atlas uniquement sur la darija arabe :

```powershell
.venv\Scripts\python.exe -m qralia.rag.generation_evaluation `
  --chunks data/processed/onee-regler-facture-residentiel/chunks.jsonl `
  --chunks data/processed/redal/chunks.jsonl `
  --chunks data/processed/telecom/chunks.jsonl `
  --evaluation data/evaluation/generation_holdout.v0.1.jsonl `
  --output outputs/generation-holdout-qwen3.5-4b-atlas-darija-arab.json `
  --generator-model qwen3.5:4b `
  --language darija_arab `
  --atlas-rewriter `
  --rewrite-language darija_arab `
  --skip-judge
```

Pour exécuter la variante few-shot sans Atlas :

```powershell
.venv\Scripts\python.exe -m qralia.rag.generation_evaluation `
  --chunks data/processed/onee-regler-facture-residentiel/chunks.jsonl `
  --chunks data/processed/redal/chunks.jsonl `
  --chunks data/processed/telecom/chunks.jsonl `
  --evaluation data/evaluation/generation_holdout.v0.1.jsonl `
  --output outputs/generation-holdout-qwen3.5-4b-darija-few-shot-exploratory.json `
  --generator-model qwen3.5:4b `
  --prompt-profile darija_few_shot `
  --skip-judge
```

Pour reproduire le smoke test Atlas-Chat-2B via Ollama :

```powershell
.venv\Scripts\python.exe -m qralia.rag.generation_evaluation `
  --chunks data/processed/onee-regler-facture-residentiel/chunks.jsonl `
  --chunks data/processed/redal/chunks.jsonl `
  --chunks data/processed/telecom/chunks.jsonl `
  --evaluation data/evaluation/generation_holdout.v0.1.jsonl `
  --output outputs/generation-holdout-qwen3.5-atlas-chat-2b-q2-smoke.json `
  --generator-model qwen3.5:4b `
  --language darija_arab `
  --atlas-rewriter `
  --atlas-backend ollama `
  --atlas-model hf.co/tensorblock/MBZUAI-Paris_Atlas-Chat-2B-GGUF:latest `
  --rewrite-language darija_arab `
  --skip-judge
```

## Résultats de référence

La baseline du 1er octobre 2026 utilise les 16 gold contexts, le prompt 0.1.0,
Qwen3-4B et Gemini 2.5 Flash :

| Mesure | Résultat |
|---|---:|
| Passage du validateur déterministe | 62,5 % |
| Passage final déterministe + juge | 31,25 % (5/16) |
| Violations de faits critiques selon le juge | 4 |
| Groundedness moyenne | 4,00 / 5 |
| Pertinence moyenne | 3,75 / 5 |
| Complétude moyenne | 3,13 / 5 |
| Correction des citations | 4,75 / 5 |
| Qualité de langue | 4,25 / 5 |
| Correction de l'abstention | 3,75 / 5 |
| Latence moyenne Qwen | 12,19 s |
| Latence moyenne Gemini | 7,66 s |

Le détail par langue montre 75 % de passage final en darija arabe, 50 % en
français et 0 % en arabe standard comme en darija latine. Ces groupes ne
contiennent que quatre cas chacun : ils localisent des erreurs mais ne
permettent pas de généraliser la qualité linguistique du modèle.

Le renderer déterministe a dû reconstruire les citations visibles sur 11 des
16 réponses. C'est une justification empirique pour ne pas déléguer leur mise
en forme au LLM.

Une variante de prompt 0.2, conçue après lecture des erreurs de calibration, a
été exécutée à titre exploratoire sans juge (quota quotidien atteint). Son taux
déterministe tombe à 56,25 % et sa latence Qwen monte à 25,91 s. Elle n'est donc
pas retenue. Son résultat reste archivé dans
`outputs/generation-qwen3-4b-prompt-v0.2-deterministic-exploratory.json` et ne
doit pas être présenté comme un test vierge.

Le rapport de référence complet est
`outputs/generation-qwen3-4b-gemini-2.5-flash.json`.

## Limites et prochaine expérience

- Les 16 cas constituent une première baseline, pas une preuve statistique
  suffisante.
- Les contextes sont les passages gold : les erreurs de retrieval ne sont pas
  encore propagées.
- La naturalité de la darija doit faire l'objet d'une revue humaine marocaine.
- Le quota gratuit observé pour Gemini 2.5 Flash est de 20 requêtes par jour ;
  l'évaluation complète doit donc être planifiée ou exécutée avec facturation.
- Après gel de cette baseline, comparer `gold context` à `top-k reranké`, puis
  comparer le meilleur générateur sur les passages réellement récupérés.
- Faire relire les sorties darija par des locuteurs marocains avant toute
  affirmation sur leur naturalité.
