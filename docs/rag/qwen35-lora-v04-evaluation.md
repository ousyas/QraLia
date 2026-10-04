# Évaluation du QLoRA Qwen3.5-4B — test gelé SFT v0.4

## Verdict

L'adaptateur produit de bonnes réponses en français et en arabe standard lorsque
la sortie est complète, mais il n'est pas prêt à être intégré dans QraLia. Les
deux principaux problèmes sont la génération en darija et une tendance à
s'abstenir alors que le passage utile est présent. De plus, 24 sorties sur 160
ont atteint exactement la limite de 512 tokens et ont été coupées : les scores
bruts de format JSON doivent donc être considérés comme une borne basse jusqu'à
leur nouvelle génération.

Décision : **ne pas remplacer le modèle de base par ce LoRA**. Conserver
l'expérience et ses artefacts, corriger le protocole de génération, puis
entraîner une v0.5 avec des cibles plus courtes et une darija revue humainement.

## Protocole

- Jeu de test gelé : 160 cas, 40 par langue.
- Langues : français, arabe standard, darija latine et darija arabe.
- Cas issus d'intents absents de l'entraînement afin de mesurer la
  généralisation, pas la mémorisation.
- Contextes de plusieurs passages comprenant le passage gold et des
  distracteurs, ou seulement des passages insuffisants pour les abstentions.
- Décodage déterministe sur Kaggle, limite de génération de 512 tokens.
- Empreinte SHA-256 des prédictions :
  `96625AF881891F2221FB6EE5266F74D5ECB9A72836E1368825911547B5A8BEC9`.

Le score « structurel » vérifie le JSON, le schéma, la langue déclarée,
l'écriture, la décision d'abstention et la sûreté des citations. Le score
« grounded strict » exige en plus que les citations correspondent uniquement
aux passages gold et que les valeurs critiques soient conservées. Ces tests
déterministes ne mesurent pas complètement la fluidité ni l'entailment
sémantique ; une réponse grammaticalement mauvaise peut encore les réussir.

## Résultats globaux

| Mesure | Résultat |
|---|---:|
| Sortie JSON et schéma valides | 136/160 — 85,0 % |
| Passage structurel, toutes sorties | 104/160 — 65,0 % |
| Passage grounded strict, toutes sorties | 85/160 — 53,1 % |
| Décision d'abstention correcte, toutes sorties | 107/160 — 66,9 % |
| Passage structurel parmi les 136 sorties valides | 76,5 % |
| Passage grounded strict parmi les sorties valides | 62,5 % |
| Abstention correcte parmi les sorties valides | 78,7 % |
| Fausses abstentions | 26 |
| Réponses données alors qu'il fallait s'abstenir | 3 |
| Cas citant au moins un distracteur | 18 |
| Rappel des valeurs critiques, sorties répondables valides | 86,2 % |
| Latence moyenne / médiane / p95 | 31,7 s / 31,2 s / 67,8 s |

La métrique principale pour décider d'une intégration est le grounded strict,
pas le simple taux de JSON valide. À 53,1 % sur l'ensemble du test, le modèle
reste trop fragile pour une application de factures.

## Effet de la troncature

Les 24 JSON invalides ont tous exactement 512 tokens de complétion. Les intents
les plus touchés sont `inwi_money_steps` (9 sorties) et
`redal_mobile_wallet` (6 sorties). Le modèle recopie souvent trop de détails du
passage, ce qui augmente simultanément la latence et le risque de coupure.

Cette observation ne permet pas de déclarer que les 24 cas auraient réussi avec
plus de tokens. Elle montre seulement que l'échec JSON n'est pas définitivement
attribuable à la compréhension. Il faut régénérer uniquement ces cas avec 800
ou 1 024 tokens, tout en considérant la verbosité comme un défaut réel.

## Résultats par langue

| Langue | JSON valide | Structurel | Grounded strict | Grounded strict parmi sorties valides | Fausses abstentions | Citations distractrices |
|---|---:|---:|---:|---:|---:|---:|
| Français | 85,0 % | 77,5 % | 77,5 % | 91,2 % | 3 | 0 |
| Arabe standard | 90,0 % | 77,5 % | 75,0 % | 83,3 % | 3 | 2 |
| Darija latine | 85,0 % | 65,0 % | 35,0 % | 41,2 % | 6 | 11 |
| Darija arabe | 80,0 % | 40,0 % | 25,0 % | 31,3 % | 14 | 5 |

Le français et l'arabe standard sont prometteurs une fois les sorties tronquées
écartées. La darija latine conserve généralement la décision répondre/refuser,
mais choisit souvent le mauvais passage et produit parfois du texte peu
compréhensible. La darija arabe est le point le plus faible : elle cumule les
troncatures et surtout 14 fausses abstentions.

Le résultat contredit l'hypothèse selon laquelle ce fine-tuning aurait résolu la
darija. Il a plutôt appris un patron d'abstention très fort en darija arabe.

## Réponse versus abstention

- Sur les 132 cas où le passage gold est présent, le grounded strict n'est que
  de 46,2 % et le modèle produit 26 fausses abstentions.
- Sur les 20 questions réellement absentes du corpus, 18 sont correctement
  refusées, soit 90 %.
- Sur les 8 simulations d'échec du retrieval, 6 sont correctement refusées,
  mais 2 reçoivent une réponse non fondée.

Le modèle est donc conservateur : il sait relativement bien dire « je ne sais
pas », mais applique aussi ce comportement lorsque la réponse existe. Ce défaut
est particulièrement important pour QraLia, car une abstention correcte protège
contre une invention, tandis qu'une fausse abstention rend le service inutile.

## Intentions les plus faibles et les plus fortes

- `redal_outage_missing` est le meilleur groupe : 90 % grounded strict. Il
  s'agit toutefois principalement d'abstentions, donc d'une tâche plus simple.
- `orange_invoice_download` atteint 75 % grounded strict sans troncature.
- `onee_advance` n'atteint que 45 % grounded strict et produit 8 fausses
  abstentions, malgré 100 % de JSON valides.
- `iam_debit_dispute` n'atteint que 40 % grounded strict et ne conserve les
  valeurs critiques que dans 33,3 % des sorties répondables valides.
- `redal_mobile_wallet` tombe à 40 % grounded strict avec 6 troncatures.
- `inwi_money_steps` tombe à 35 % grounded strict avec 9 troncatures ; c'est le
  symptôme le plus net de réponses trop longues.

## Types de formulation

Les questions implicites sur le fournisseur sont les plus difficiles pour la
décision : 59,4 % d'abstentions correctes. Les formulations informelles ont le
meilleur grounded strict brut (59,4 %), ce qui montre que « langage informel »
n'est pas, à lui seul, la cause des erreurs. Les formulations longues ont un bon
score structurel parmi les sorties complètes, mais 7 sorties sont tronquées et
leur latence moyenne atteint 40 secondes.

## Exemples diagnostiques

### Bonne réponse

À « Puis-je déposer une avance pour mes prochaines factures ONEE ? », le modèle
répond correctement qu'un montant au choix peut être déposé auprès de l'agence
ONEE Branche Électricité et cite uniquement le passage `paiement-acompte`.

### Fausse abstention

À « نقدر نخلي تسبيق على فواتير المكتب الجاية؟ », le passage `paiement-acompte`
est présent, mais le modèle répond que les passages ne suffisent pas. C'est un
échec de génération/compréhension, pas un échec du retrieval.

### Réponse dangereuse à une question non couverte

À la question arabe demandant la période maximale pour payer une facture
Orange, le modèle affirme qu'un nombre illimité de factures peut être payé à
l'avance en citant Tassbik. Le passage ne donne pourtant pas le délai demandé.
La citation est syntaxiquement valide, mais elle n'étaye pas la réponse.

### Darija peu intelligible

À une question sur la durée d'une coupure Redal, le modèle répond avec une
phrase darija latine incohérente et cite le passage sur la facturation
mensuelle. Cet exemple illustre pourquoi les métriques déterministes doivent
être complétées par une revue sémantique humaine.

## Comparaison contrôlée avec Qwen3.5-4B de base

Sur les 24 cas exactement communs au précédent benchmark direct en français,
arabe standard et darija arabe :

| Modèle | Passage déterministe |
|---|---:|
| Qwen3.5-4B de base | 20/24 — 83,3 % |
| Qwen3.5-4B + LoRA | 17/24 — 70,8 % |
| LoRA, grounded strict | 15/24 — 62,5 % |

Le modèle de base réussit quatre cas que le LoRA échoue ; le LoRA ne récupère
qu'un cas échoué par la base. Cette comparaison est indicative, car les moteurs
d'inférence et la quantification ne sont pas identiques, mais elle ne fournit
aucune preuve d'un gain apporté par le LoRA. La latence ne doit pas être comparée
entre les deux exécutions, réalisées sur du matériel différent.

## Diagnostic de l'entraînement

Le faible `eval_loss` et l'accuracy token élevée du checkpoint ne se traduisent
pas par une bonne généralisation sur les intents gelés. Les causes probables
sont :

1. seulement 360 exemples d'entraînement, dont 100 abstentions ;
2. des patrons de réponse et d'abstention répétés, faciles à mémoriser ;
3. des réponses cibles trop longues qui recopient plusieurs phrases du chunk ;
4. une qualité linguistique insuffisante des cibles darija synthétiques ;
5. une sélection du checkpoint par la loss plutôt que par une génération
   complète sur un jeu de validation représentatif ;
6. une généralisation inter-intent réellement difficile, ce qui est souhaitable
   pour un test honnête.

Le rang LoRA 8 et `alpha=16` ne sont pas les premiers paramètres à augmenter.
Le goulot d'étranglement est d'abord la qualité, la variété et la concision des
données.

## Plan recommandé

1. Régénérer les 24 sorties tronquées avec `max_new_tokens=1024` et recalculer
   exactement le même rapport.
2. Faire une revue humaine de toutes les sorties darija et des 85 cas marqués
   grounded strict afin de détecter les faux positifs sémantiques.
3. Construire une v0.5 avec des réponses de 1 à 3 claims, sans recopie inutile
   du chunk.
4. Ajouter des paires contrastives : même fournisseur et mêmes passages, mais
   une question répondable et une question non couverte.
5. Faire corriger les exemples darija par des locuteurs marocains et ajouter
   des paraphrases réellement différentes, notamment pour les fournisseurs
   implicites.
6. Réentraîner avec un petit sweep contrôlé, mais sélectionner le checkpoint à
   l'aide du grounded strict, des fausses abstentions et d'une note humaine, pas
   uniquement de la loss.
7. Comparer base et LoRA sur les 160 mêmes cas, avec le même moteur, la même
   quantification et les mêmes paramètres de décodage.

## Artefacts reproductibles

- Prédictions copiées localement :
  `artifacts/evaluation/qralia_lora_test_predictions.jsonl`
- Rapport JSON détaillé :
  `artifacts/evaluation/qralia_lora_test_analysis.json`
- Analyseur : `scripts/analyze_sft_predictions.py`
