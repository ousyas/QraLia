# Chatbot conversationnel QraLia v0.1

## Objectif

Ajouter des questions de suivi sans transformer les anciennes réponses du LLM
en sources de vérité. Chaque question autonome reconstruite repasse dans le
pipeline facture/RAG et dans ses garde-fous existants.

## Flux

```text
message courant + 6 derniers tours + état sûr de la facture
                         |
                         v
        standalone / rewrite / clarify
                         |
             validation déterministe
                         |
                         v
        pipeline QuestionAnswerService existant
                         |
                         v
      nouvelles preuves + réponse + historique
```

## Mémoire

La mémoire v0.1 est en RAM, bornée à 50 conversations, avec une durée de vie de
30 minutes renouvelée à chaque message. Une conversation est liée à un unique
`invoice_id`. Supprimer une facture supprime aussi ses conversations. Aucun
octet du document n'est conservé dans la mémoire conversationnelle.

Chaque tour conserve le texte original, la langue, et pour audit seulement la
question autonome, la route et les IDs de preuves. Les anciennes réponses
assistant servent à comprendre le dialogue mais ne sont jamais injectées comme
preuves dans la génération suivante.

## Résolution

- Une question autonome est transmise sans modification.
- Un suivi est envoyé à Qwen3.5 avec une sortie JSON contrainte.
- La sortie est rejetée si elle change de langue déclarée, invente un nombre ou
  change de fournisseur.
- Si Qwen est indisponible ou rejeté, des règles à haute précision couvrent les
  suivis simples (date limite, montant, chèque, virement, espèces).
- Un suivi non résolu déclenche une clarification, jamais une supposition.

La normalisation BM25 intervient après cette étape, dans le pipeline de
recherche existant. Le texte historique original n'est pas remplacé par des
tokens normalisés.

## API

```http
POST   /v1/invoices/{invoice_id}/conversations
POST   /v1/conversations/{conversation_id}/messages
GET    /v1/conversations/{conversation_id}
DELETE /v1/conversations/{conversation_id}
```

L'ancienne route stateless `/v1/invoices/{invoice_id}/questions` reste
compatible.

Le frontend React crée automatiquement une conversation après l'envoi de la
facture, affiche les tours sous forme de bulles, rend visible la question
autonome lorsqu'un suivi a été reformulé et conserve les sources au niveau de
chaque réponse. Une demande de clarification est affichée comme un message
assistant et n'appelle pas le pipeline de génération.

## Évaluation v0.1

Le fichier `data/evaluation/conversation_followups.v0.1.jsonl` contient 20 cas
équilibrés entre français, arabe standard, darija arabe et darija latine. Il
couvre les questions autonomes, le chèque, le virement, la date limite et les
ambiguïtés qui doivent provoquer une clarification.

Les métriques sont : exactitude de l'action, usage correct de l'historique,
rappel des concepts obligatoires et exactitude stricte. Le résultat de la
baseline déterministe est enregistré dans
`outputs/conversation-followups-v0.1.json`. Ce score mesure seulement ce petit
jeu contrôlé ; il ne prouve pas encore une robustesse conversationnelle réelle.

La baseline déterministe obtient 20/20 sur ce jeu construit autour de ses cas
à haute précision. Un premier passage réel de Qwen3.5 a utilisé Qwen pour 16
cas et le chemin direct pour quatre : exactitude d'action 0,80, usage déclaré
de l'historique 0,20, rappel des concepts 0,683 et exactitude stricte 0,20.
Ces résultats faibles, particulièrement en darija latine, ont conduit à
rejeter désormais toute sortie `rewrite/clarify` qui déclare
`history_used=false`. Le score du système avec fallback ne doit pas être
présenté comme le score propre de Qwen.

## Holdout candidat v0.2 et validation sémantique

Le fichier `data/evaluation/conversation_followups.v0.2.jsonl` contient 80
conversations synthétiques candidates : 20 familles de scénarios déclinées en
français, arabe standard, darija arabe et darija latine. Il comporte 20 cas
autonomes, 48 reformulations et 12 clarifications. Le manifeste associé fixe
les effectifs, la provenance et le SHA-256. Son statut reste
`candidate_human_review` : il ne doit pas être présenté comme un échantillon de
conversations réelles avant relecture par des locuteurs.

Le validateur ne contrôle plus seulement le JSON, la langue, les nombres et le
fournisseur. Il compare aussi la question courante, l'historique utilisateur et
la reformulation selon un contrat sémantique : concepts métier obligatoires,
dimension interrogative (`où`, `comment`, `quand`, `combien`, `quel`), alphabet
attendu et registre darija. Il rejette l'introduction d'un moyen de paiement et
demande une clarification lorsque plusieurs antécédents sont plausibles.

La première mesure, exécutée une seule fois sans LLM sur le holdout figé,
constitue une baseline honnête : action 0,6625 ; usage de l'historique 0,825 ;
rappel des concepts 0,79375 ; dimension 0,7625 ; exactitude stricte 0,55. Ces
résultats ne servent pas à retoucher ce holdout. Les prochaines améliorations
doivent être conçues sur des tests de développement séparés, puis mesurées sur
un nouveau holdout ou après protocole de versionnement explicite.

### Expérience Qwen-first

Un profil expérimental `qwen_first` supprime le filtre lexical préalable dès
qu'un historique existe. Qwen décide alors librement entre `standalone`,
`rewrite` et `clarify`. Les garde-fous placés après le modèle restent identiques
(nombres, fournisseur, concepts critiques, dimension et ambiguïtés). Le profil
de production `conservative` n'est donc pas modifié par cette expérience.

| Profil | Action | Historique | Concepts | Dimension | Strict |
|---|---:|---:|---:|---:|---:|
| Déterministe sans LLM | 0,6625 | 0,8250 | 0,7938 | 0,7625 | 0,5500 |
| Qwen conservateur | 0,5000 | 0,5625 | 0,8813 | 0,8625 | 0,4125 |
| Qwen-first | 0,5500 | 0,6250 | 0,8688 | 0,8750 | 0,4250 |

Qwen-first accepte 48 décisions du modèle, contre 33 avec le filtre
conservateur. Il améliore légèrement l'action et la réussite stricte du profil
Qwen, ainsi que la dimension interrogative, mais ne dépasse pas la baseline
déterministe. Sa faiblesse principale concerne les clarifications : 0,1667
d'exactitude contre 0,9167 pour la baseline déterministe. Qwen traite souvent
un fragment dépendant du contexte comme une question autonome. La conclusion
est de conserver `qwen_first` comme profil expérimental et de garder un détecteur
déterministe d'ambiguïté avant tout appel de reformulation en production.

Commande reproductible :

```powershell
python -m qralia.rag.conversation_evaluation `
  --cases data/evaluation/conversation_followups.v0.2.jsonl `
  --output outputs/conversation-followups-v0.2-qwen-first.json `
  --use-ollama --routing-profile qwen_first
```
