"""Enrichit la banque v0.3 et construit des contextes Top-k réalistes.

Les paraphrases Gemini sont des candidats, jamais des vérités automatiques.
Les splits, les gold chunks, les distracteurs, les retrieval misses et tous
les contrôles de fuite sont déterministes et indépendants du LLM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field

from qralia.rag.bm25_evaluation import load_active_chunks


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/evaluation/sft_questions.v0.3.jsonl"
CACHE = ROOT / "data/evaluation/sft_question_paraphrases.v0.4.jsonl"
MULTI_CACHE = ROOT / "data/evaluation/sft_question_multichunk.v0.4.jsonl"
OUTPUT = ROOT / "data/evaluation/sft_questions.v0.4.jsonl"
MANIFEST = ROOT / "data/evaluation/sft_questions.v0.4.manifest.json"
REVIEW = ROOT / "data/evaluation/sft_questions.v0.4.review.jsonl"
SCHEMA = ROOT / "schemas/sft_question_v0.4.schema.json"
CHUNK_SCHEMA = ROOT / "schemas/retrieval_chunk.schema.json"
CHUNK_PATHS = [
    ROOT / "data/processed/onee-regler-facture-residentiel/chunks.jsonl",
    ROOT / "data/processed/redal/chunks.jsonl",
    ROOT / "data/processed/telecom/chunks.jsonl",
]
MODEL = "gemini-3.5-flash-lite"
KINDS = ("indirect", "contextual_long", "provider_implicit", "informal_typed")


class Candidate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    language: Literal["fr", "ar", "darija_latn", "darija_arab"]
    formulation_kind: Literal["indirect", "contextual_long", "provider_implicit", "informal_typed"]
    question: str = Field(min_length=5)
    register_name: Literal["formal", "neutral", "informal", "whatsapp"] = Field(alias="register")
    provider_mentioned: bool


class CandidateBatch(BaseModel):
    candidates: list[Candidate] = Field(min_length=16, max_length=16)


class MultiCandidate(BaseModel):
    intent_id: str
    language: Literal["fr", "ar", "darija_latn", "darija_arab"]
    question: str = Field(min_length=15)


class MultiBatch(BaseModel):
    candidates: list[MultiCandidate] = Field(min_length=64, max_length=64)


SPLIT_OVERRIDE = {
    "redal_mobile_wallet": "test",
    "iam_billing_error": "test",
    "orange_invoice_download": "test",
}

# La variante contextual_long de ces intentions demande deux informations.
# Les deux chunks sont garantis dans le même split ou réservés à ce split.
MULTI_SECOND = {
    "onee_cash": ("onee", "paiement-cheque"),
    "onee_date_holiday": ("onee", "paiement-virement"),
    "redal_online_card": ("srm_rsk", "paiement-prelevement"),
    "redal_direct_debit": ("srm_rsk", "paiement-en-ligne-carte"),
    "iam_monthly_billing": ("iam", "exigibilite-non-reception"),
    "iam_invoice_not_received": ("iam", "facturation-mensuelle"),
    "orange_recharge": ("orange_ma", "paiement-max-it-carte"),
    "orange_money": ("orange_ma", "paiement-recharge"),
    "inwi_agency_hours": ("inwi", "paiement-agents"),
    "inwi_payment_agents": ("inwi", "paiement-agence"),
    "redal_atm": ("srm_rsk", "paiement-ebanking"),
    "iam_late_penalty": ("iam", "suspension-retablissement"),
    "inwi_myinwi_reward": ("inwi", "paiement-ebanking"),
    "iam_billing_error": ("iam", "prelevement-contestation"),
    "orange_payment_delay": ("orange_ma", "consulter-facture"),
    "inwi_money_steps": ("inwi", "paiement-prelevement"),
}

# Une seule formulation par langue pour huit intentions simule un Top-k raté :
# le gold existe dans le corpus, mais est volontairement absent du contexte.
MISS_INTENTS = {
    "onee_cheque",
    "redal_direct_debit",
    "iam_monthly_billing",
    "orange_recharge",
    "inwi_agency_hours",
    "redal_atm",
    "iam_debit_dispute",
    "orange_payment_delay",
}

QUESTION_OVERRIDES = {
    ("iam_invoice_not_received", "darija_latn", "contextual_long"): "Ma wslatnich factora dyal IAM had chher w l'échéance qrebat. Wach khasni nkhlles bach ma nti7ch f chi mochkil wakha ma chfthach?",
    ("inwi_agency_hours", "darija_latn", "contextual_long"): "Ana khdam tol simana w ma 3ndich lwe9t 7ta l sebt f l3chiya. Wach n9der nmchi l agence dik sa3a bach nkhlles?",
    ("iam_agency_hours_missing", "darija_latn", "indirect"): "Bghit n3ref wach lmkatb dyal charika kayst9blo nass nhar l7ed.",
    ("inwi_failed_payment_missing", "darija_latn", "contextual_long"): "Kont kan nkhlles factora f l'application w lflous t9t3o mn compte, walakin l'opération ma dazetsh. Imta y9dro yrj3o lia flousi?",
    ("redal_atm", "darija_latn", "contextual_long"): "Achmen bankat kaykhelliw nkhlles Redal f GAB, w achmen khadamat dyal banque à distance n9der نستعمل?".replace("نستعمل", "nsta3mel"),
    ("iam_late_penalty", "darija_latn", "contextual_long"): "Kifach IAM katteb9 gharamat retard, w f achmen 7alat t9der tqte3 service kamel wla chi joz2 mno?",
    ("onee_cash", "fr", "contextual_long"): "Je veux comparer deux possibilités : où puis-je payer en espèces, et à quel ordre dois-je établir un chèque ?",
    ("onee_cash", "ar", "contextual_long"): "أريد المقارنة بين طريقتين: أين يمكنني الأداء نقدا، وباسم من أحرر الشيك؟",
    ("onee_cash", "darija_latn", "contextual_long"): "Bghit n9aren bin jouj toro9: fin n9der nkhlles b cash, w 3la smit chkon khasni nkteb chik?",
    ("onee_cash", "darija_arab", "contextual_long"): "بغيت نقارن بين جوج طرق: فين نقدر نخلص بالكاش، وعلى سميت شكون خاصني نكتب الشيك؟",
    ("onee_date_holiday", "fr", "contextual_long"): "Si ma date de paiement tombe un jour férié, quand dois-je payer, et comment mettre en place le règlement automatique par la banque ?",
    ("onee_date_holiday", "ar", "contextual_long"): "إذا صادف تاريخ الأداء يوم عطلة، فمتى أؤدي، وكيف أفعل الأداء التلقائي عن طريق البنك؟",
    ("onee_date_holiday", "darija_latn", "contextual_long"): "Ila date dyal lkhlass jat f nhar 3otla, imta nkhlles, w kifach ndir bach lbanka tkhlles lia automatiquement?",
    ("onee_date_holiday", "darija_arab", "contextual_long"): "إلا تاريخ الخلاص جا فنهار عطلة، إمتى نخلص، وكيفاش ندير باش البنك يخلص ليا أوتوماتيكيا؟",
    ("redal_online_card", "darija_arab", "contextual_long"): "إلا خلصت Redal بالكارط فالإنترنت، فين نلقى الوصل، وكيفاش ندير الاقتطاع البنكي التلقائي؟",
    ("redal_direct_debit", "darija_arab", "contextual_long"): "كننسى نخلص الفاتورة؛ كيفاش نفعل الاقتطاع التلقائي، وشنو طرق الأداء من موقع Redal؟",
    ("iam_billing_error", "ar", "contextual_long"): "ماذا تفعل اتصالات المغرب إذا تأكد خطأ في الفاتورة، وما أجل الاعتراض على الاقتطاع البنكي؟",
    ("inwi_myinwi_reward", "darija_latn", "contextual_long"): "Ila khllest f My inwi, achmen avantage kayn, w kifach n9der nkhlles men espace client dyal lbanka?",
    ("inwi_myinwi_reward", "darija_arab", "contextual_long"): "إلا خلصت فـ My inwi، شنو الامتياز اللي كاين، وكيفاش نقدر نخلص من فضاء الزبون ديال البنك؟",
    ("inwi_money_steps", "fr", "contextual_long"): "Quelles étapes dois-je suivre pour payer avec inwi money, et comment mettre en place le prélèvement bancaire automatique ?",
    ("inwi_money_steps", "ar", "contextual_long"): "ما الخطوات اللازمة للأداء بواسطة inwi money، وكيف أفعل الاقتطاع البنكي التلقائي؟",
    ("inwi_money_steps", "darija_latn", "contextual_long"): "Kifach nkhlles b inwi money khotwa b khotwa, w kifach ndir prélèvement bancaire automatique?",
    ("inwi_money_steps", "darija_arab", "contextual_long"): "كيفاش نخلص بـ inwi money خطوة بخطوة، وكيفاش نفعل الاقتطاع البنكي التلقائي؟",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def stable_order(seed: str, values: list[str]) -> list[str]:
    return sorted(values, key=lambda value: hashlib.sha256(f"{seed}|{value}".encode()).hexdigest())


def prompt_for(intent_rows: list[dict[str, Any]], chunk_text: str) -> str:
    bases = {row["language"]: row["question"] for row in intent_rows}
    return f"""Tu construis des questions marocaines naturelles pour QraLia.

INTENTION: {intent_rows[0]['intent_id']}
FOURNISSEUR: {intent_rows[0]['provider_id']}
PASSAGE OFFICIEL (pour comprendre le sujet, sans recopier sa réponse):
{chunk_text}

QUESTIONS DE BASE:
{json.dumps(bases, ensure_ascii=False, indent=2)}

Produis exactement quatre nouvelles formulations pour chacune des quatre
langues, donc exactement 16 objets. Pour chaque langue, utilise une fois chacun:
- indirect: évite autant que possible les mots exacts du passage;
- contextual_long: deux propositions naturelles avec une situation vécue;
- provider_implicit: ne nomme ni le fournisseur ni un alias; le fournisseur
  viendra des métadonnées d'une facture synthétique;
- informal_typed: message réellement plausible, familier mais lisible.

Contraintes:
- ne donne jamais la réponse dans la question;
- ne change aucun nombre ou identifiant déjà nécessaire à l'intention;
- pas de fautes artificielles par suppression aléatoire de lettres;
- pas de faux ASR, pas de transcription audio inventée;
- darija_latn uniquement en caractères latins/chiffres;
- darija_arab principalement en écriture arabe;
- le français doit inclure des formulations familières, pas seulement formelles;
- contextual_long doit être sensiblement plus longue que la base;
- provider_mentioned=false uniquement pour provider_implicit.
"""


def generate_cache(rows: list[dict[str, Any]]) -> None:
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY absente de .env")
    from google import genai

    chunks = [chunk for path in CHUNK_PATHS for chunk in load_active_chunks(path, CHUNK_SCHEMA)]
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    existing = {row["intent_id"]: row for row in read_jsonl(CACHE)} if CACHE.exists() else {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["intent_id"]].append(row)
    client = genai.Client(api_key=api_key)
    for intent_id, intent_rows in grouped.items():
        if intent_id in existing:
            continue
        source_id = next((row["source_chunk_id"] for row in intent_rows if row["source_chunk_id"]), None)
        context_id = intent_rows[0]["context_chunk_ids"][0]
        chunk_text = chunks_by_id[source_id or context_id]["text"]
        response = None
        for attempt, delay in enumerate((0, 5, 15), start=1):
            if delay:
                time.sleep(delay)
            try:
                response = client.models.generate_content(
                    model=MODEL,
                    contents=prompt_for(intent_rows, chunk_text),
                    config={"temperature": 0.7, "response_mime_type": "application/json", "response_schema": CandidateBatch},
                )
                break
            except Exception:
                if attempt == 3:
                    raise
        parsed = response.parsed
        batch = parsed if isinstance(parsed, CandidateBatch) else CandidateBatch.model_validate(parsed)
        keys = {(candidate.language, candidate.formulation_kind) for candidate in batch.candidates}
        expected = {(language, kind) for language in ("fr", "ar", "darija_latn", "darija_arab") for kind in KINDS}
        if keys != expected:
            raise ValueError(f"Couverture incomplète pour {intent_id}: {keys ^ expected}")
        existing[intent_id] = {"intent_id": intent_id, "model": MODEL, "prompt_version": "v0.4.0", "candidates": [candidate.model_dump(by_alias=True) for candidate in batch.candidates], "review_status": "candidate_human_review"}
        write_jsonl(CACHE, [existing[key] for key in sorted(existing)])
        print(f"generated {intent_id} ({len(existing)}/{len(grouped)})", flush=True)

    if not MULTI_CACHE.exists():
        specs = []
        for intent_id, second_key in MULTI_SECOND.items():
            intent_rows = grouped[intent_id]
            first_id = next(row["source_chunk_id"] for row in intent_rows if row["source_chunk_id"])
            second = next(chunk for chunk in chunks if (chunk["metadata"]["provider_id"], chunk["section_id"]) == second_key)
            specs.append({"intent_id": intent_id, "provider_id": intent_rows[0]["provider_id"], "base_questions": {row["language"]: row["question"] for row in intent_rows}, "passage_1": chunks_by_id[first_id]["text"], "passage_2": second["text"]})
        multi_prompt = f"""Crée des questions QraLia qui nécessitent réellement DEUX passages.
Pour chacun des {len(specs)} blocs et chacune des quatre langues fr, ar,
darija_latn, darija_arab, produis une question longue et naturelle qui demande
une information du passage_1 ET une information du passage_2. Retourne donc
exactement {len(specs) * 4} objets. N'inclus jamais la réponse. La darija_latn
ne doit contenir aucune lettre arabe. N'invente pas d'ASR ni de faute aléatoire.

{json.dumps(specs, ensure_ascii=False)}
"""
        response = client.models.generate_content(model=MODEL, contents=multi_prompt, config={"temperature": 0.5, "response_mime_type": "application/json", "response_schema": MultiBatch})
        parsed = response.parsed
        batch = parsed if isinstance(parsed, MultiBatch) else MultiBatch.model_validate(parsed)
        expected = {(intent_id, language) for intent_id in MULTI_SECOND for language in ("fr", "ar", "darija_latn", "darija_arab")}
        actual = {(item.intent_id, item.language) for item in batch.candidates}
        if actual != expected:
            raise ValueError(f"Couverture multi-chunk invalide: {actual ^ expected}")
        write_jsonl(MULTI_CACHE, [{"intent_id": item.intent_id, "language": item.language, "question": item.question, "model": MODEL, "review_status": "candidate_human_review"} for item in batch.candidates])
        print(f"generated {len(batch.candidates)} multi-chunk candidates", flush=True)


def choose_distractors(*, seed: str, chunks: list[dict[str, Any]], provider: str, domain: str, excluded: set[str], count: int) -> list[str]:
    same_provider = [chunk["chunk_id"] for chunk in chunks if chunk["metadata"]["provider_id"] == provider and chunk["chunk_id"] not in excluded]
    same_domain = [chunk["chunk_id"] for chunk in chunks if chunk["metadata"]["domain"] == domain and chunk["chunk_id"] not in excluded and chunk["chunk_id"] not in same_provider]
    candidates = stable_order(seed, same_provider) + stable_order(seed + "|domain", same_domain)
    if len(candidates) < count:
        raise ValueError(f"Pas assez de distracteurs pour {provider}/{domain}")
    return candidates[:count]


def build() -> None:
    base_rows = read_jsonl(SOURCE)
    cache_by_intent = {row["intent_id"]: row for row in read_jsonl(CACHE)}
    multi_by_key = {(row["intent_id"], row["language"]): row for row in read_jsonl(MULTI_CACHE)}
    chunks = [chunk for path in CHUNK_PATHS for chunk in load_active_chunks(path, CHUNK_SCHEMA)]
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    by_provider_section = {(chunk["metadata"]["provider_id"], chunk["section_id"]): chunk for chunk in chunks}
    validator = Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8")))
    rows: list[dict[str, Any]] = []
    number = 1

    for base in base_rows:
        split = SPLIT_OVERRIDE.get(base["intent_id"], base["split"])
        generated = {(item["language"], item["formulation_kind"]): item for item in cache_by_intent[base["intent_id"]]["candidates"]}
        base_register = "neutral" if base["register"] == "standard" else base["register"]
        variants = [{"question": base["question"], "formulation_kind": "direct", "register": base_register, "provider_mentioned": True, "generator": "human_curated_v0.3"}]
        variants += [{**generated[(base["language"], kind)], "generator": MODEL} for kind in KINDS]
        for variant in variants:
            if variant["formulation_kind"] == "contextual_long" and base["intent_id"] in MULTI_SECOND:
                multi = multi_by_key[(base["intent_id"], base["language"])]
                variant["question"] = multi["question"]
                variant["generator"] = multi["model"]
            key = (base["intent_id"], base["language"], variant["formulation_kind"])
            if key in QUESTION_OVERRIDES:
                variant["question"] = QUESTION_OVERRIDES[key]
                variant["generator"] = "human_correction_v0.4"
        for variant_index, variant in enumerate(variants):
            gold = [] if base["should_abstain"] else [base["source_chunk_id"]]
            if variant["formulation_kind"] == "contextual_long" and base["intent_id"] in MULTI_SECOND and gold:
                second = by_provider_section[MULTI_SECOND[base["intent_id"]]]["chunk_id"]
                if second not in gold:
                    gold.append(second)
            retrieval_miss = bool(gold and base["intent_id"] in MISS_INTENTS and variant["formulation_kind"] == "informal_typed")
            retrieval_outcome = "miss" if retrieval_miss else ("not_in_corpus" if base["should_abstain"] else "hit")
            should_abstain = base["should_abstain"] or retrieval_miss
            excluded = set(gold)
            distractors = choose_distractors(seed=f"v04-{number}", chunks=chunks, provider=base["provider_id"], domain=base["domain"], excluded=excluded, count=4 if should_abstain else max(2, 4 - len(gold)))
            context = distractors if should_abstain else gold + distractors
            context = stable_order(f"context-v04-{number}", context)[:5]
            provider_source = "synthetic_invoice_metadata" if variant["formulation_kind"] == "provider_implicit" else "explicit_or_alias_in_question"
            row = {"schema_version": "0.4.0", "case_id": f"sftq-v04-{number:04d}", "intent_id": base["intent_id"], "semantic_family_id": base["semantic_family_id"], "language": base["language"], "domain": base["domain"], "provider_id": base["provider_id"], "provider_source": provider_source, "channel": "typed", "register": variant["register"], "formulation_kind": variant["formulation_kind"], "question": variant["question"], "split": split, "gold_chunk_ids": gold, "context_chunk_ids": context, "distractor_chunk_ids": [cid for cid in context if cid not in gold], "retrieval_outcome": retrieval_outcome, "should_abstain": should_abstain, "answerable_in_given_context": not should_abstain, "answerable_in_corpus": bool(gold) or base["answerable_in_corpus"], "generation_provenance": variant["generator"], "review_status": "candidate_human_review"}
            validator.validate(row)
            rows.append(row)
            number += 1

    write_jsonl(OUTPUT, rows)
    review_groups: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["intent_id"], row["language"])
        group = review_groups.setdefault(key, {"review_group_id": f"{row['intent_id']}--{row['language']}", "intent_id": row["intent_id"], "language": row["language"], "split": row["split"], "questions": [], "human_decision": None, "human_comment": ""})
        group["questions"].append({field: row[field] for field in ("case_id", "formulation_kind", "register", "provider_source", "question", "retrieval_outcome", "gold_chunk_ids", "context_chunk_ids")})
    write_jsonl(REVIEW, list(review_groups.values()))
    manifest = {"schema_version": "0.4.0", "status": "candidate_human_review", "paraphrase_models": sorted({item["model"] for item in cache_by_intent.values()} | {item["model"] for item in read_jsonl(MULTI_CACHE)}), "counts": {"questions": len(rows), "intents": len({r['intent_id'] for r in rows}), "train": sum(r['split']=='train' for r in rows), "validation": sum(r['split']=='validation' for r in rows), "test": sum(r['split']=='test' for r in rows), "review_groups": len(review_groups)}, "language_counts": dict(sorted(Counter(r["language"] for r in rows).items())), "split_intent_counts": {split: len({r['intent_id'] for r in rows if r['split']==split}) for split in ("train", "validation", "test")}, "abstention_count": sum(r["should_abstain"] for r in rows), "abstention_rate": sum(r["should_abstain"] for r in rows)/len(rows), "retrieval_miss_count": sum(r["retrieval_outcome"]=="miss" for r in rows), "retrieval_miss_rate": sum(r["retrieval_outcome"]=="miss" for r in rows)/len(rows), "multi_chunk_count": sum(len(r["gold_chunk_ids"])>1 for r in rows), "multi_chunk_rate": sum(len(r["gold_chunk_ids"])>1 for r in rows)/len(rows), "long_context_count": sum(r["formulation_kind"]=="contextual_long" for r in rows), "implicit_provider_count": sum(r["provider_source"]=="synthetic_invoice_metadata" for r in rows), "context_size_counts": dict(sorted(Counter(len(r["context_chunk_ids"]) for r in rows).items())), "files": {"questions": {"path": OUTPUT.name, "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}, "review": {"path": REVIEW.name, "sha256": hashlib.sha256(REVIEW.read_bytes()).hexdigest()}, "paraphrase_cache": {"path": CACHE.name, "sha256": hashlib.sha256(CACHE.read_bytes()).hexdigest()}, "multichunk_cache": {"path": MULTI_CACHE.name, "sha256": hashlib.sha256(MULTI_CACHE.read_bytes()).hexdigest()}}}
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="Appelle Gemini pour créer les candidats manquants.")
    args = parser.parse_args()
    base_rows = read_jsonl(SOURCE)
    if args.generate:
        generate_cache(base_rows)
    build()


if __name__ == "__main__":
    main()
