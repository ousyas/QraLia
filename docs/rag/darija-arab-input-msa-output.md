# Expérience : question en darija arabe, réponse en arabe standard

## Hypothèse

Qwen3.5-4B peut recevoir la question originale en darija arabe, utiliser les
passages inchangés, puis répondre en arabe standard simple. Cette stratégie
évite de lui demander de générer une darija qu'il maîtrise mal.

## Protocole

- Modèle : `qwen3.5:4b` Q4_K_M local via Ollama, sans LoRA.
- 40 questions `darija_arab` du test SFT v0.4 gelé.
- Questions, fournisseurs, passages et distracteurs inchangés.
- Seules modifications du prompt : suppression des few-shots de style darija
  et sortie imposée en arabe standard avec `language="ar"`.
- Température 0, seed 42, thinking désactivé, 400 tokens maximum.
- Les citations manquantes dans le texte visible sont ajoutées par le renderer
  déterministe de QraLia à partir des claims structurés.

## Résultats

| Mesure | Base, sortie arabe standard | LoRA précédent, sortie darija arabe |
|---|---:|---:|
| JSON valide | 100 % | 80 % |
| Passage structurel | 77,5 % | 40 % |
| Grounded strict | 45 % | 25 % |
| Décision répondre/s'abstenir | 85 % | 42,5 % |
| Fausses abstentions | 1 | 14 |
| Réponses dangereuses | 5 | 1 |
| Cas citant un distracteur | 15 | 5 |
| Latence moyenne | 26,9 s | 25,8 s |

La stratégie améliore nettement la langue, le format et les fausses
abstentions. Le grounded strict gagne 20 points. En revanche, le modèle devient
plus affirmatif : il cite plus souvent un passage voisin mais non pertinent et
répond cinq fois lorsqu'il devrait s'abstenir.

Sur les huit questions directes déjà exécutées avec le Qwen de base en sortie
darija, le passage structurel progresse de 5/8 à 7/8. Le grounded strict de la
nouvelle variante est de 5/8. Cette petite intersection confirme surtout le
gain de stabilité de format ; elle ne suffit pas à prouver un gain sémantique
sur tous les cas.

## Exemples réussis

Question : `نقدر نخلي تسبيق على فواتير المكتب الجاية؟`

Réponse : `يمكنك دفع فاتورتك إما عن طريق الإيداع المبدئي في فرع ONEE الكهرباء
الخاص بك.` La réponse utilise le passage ONEE sur l'acompte et ne s'abstient
plus à tort.

Question : `IAM قطعات ليا الفلوس من الكونط شحال عندي باش نعترض بالكتابة؟`

Réponse : `يجب على العميل تقديم اعتراض مكتوب للبنك خلال 7 أيام من تاريخ الخصم.`
Le délai de 7 jours et le bon passage sont conservés.

Question : `كيفاش نخلص الفاتورة بـ inwi money خطوة بخطوة؟`

Réponse : `أدخل فئة الدفع «تيليكوم» والفاتوري «فينوي» ورقم الهاتف. تأكد من رقم
الهاتف ثم تأكد برمزك السري.` Le passage Inwi Money correct est sélectionné.

## Échec de sûreté représentatif

À une question sur le délai de rétablissement d'une coupure Redal, information
absente du corpus, le modèle répond que le paiement en ligne est disponible en
permanence. La phrase est sourcée, mais la source ne répond pas à la question.
Cela montre qu'une citation valide ne garantit pas l'entailment.

## Décision

L'approche est prometteuse pour l'expérience utilisateur, mais ne doit pas être
activée seule en production. Elle doit être associée à une consigne plus stricte
sur la pertinence exacte, une limite de 1 à 2 claims, un seuil de retrieval et
un contrôle claim-question-passage avant affichage.

Prochaine expérience recommandée : renforcer le prompt avec « réponds seulement
au fait demandé ; un passage du bon fournisseur mais hors sujet est
insuffisant », puis réévaluer en priorité les hard negatives et les 15 cas ayant
cité un distracteur.
