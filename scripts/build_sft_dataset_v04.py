"""Construit les cibles SFT v0.4 après la banque de questions auditée.

Les résumés multilingues sont générés une fois par chunk et restent candidats
à revue humaine. L'assemblage des citations, abstentions, splits et prompts de
test est entièrement déterministe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from jsonschema import Draft202012Validator
from pydantic import BaseModel, Field

from qralia.rag.bm25_evaluation import load_active_chunks
from qralia.rag.generation_evaluation import AnswerClaim, GroundedAnswer, build_generation_prompt, render_answer_from_claims


ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "data/evaluation/sft_questions.v0.4.jsonl"
TARGET_CACHE = ROOT / "data/training/sft_chunk_targets.v0.4.jsonl"
OUTPUT_DIR = ROOT / "data/training/qralia_sft_v0.4"
SCHEMA = ROOT / "schemas/sft_example_v0.4.schema.json"
CHUNK_SCHEMA = ROOT / "schemas/retrieval_chunk.schema.json"
CHUNK_PATHS = [ROOT / "data/processed/onee-regler-facture-residentiel/chunks.jsonl", ROOT / "data/processed/redal/chunks.jsonl", ROOT / "data/processed/telecom/chunks.jsonl"]
MODEL = "gemini-3.5-flash-lite"
LANGUAGES = ("fr", "ar", "darija_latn", "darija_arab")
ARABIC = re.compile(r"[\u0600-\u06ff]")
# Les chiffres intégrés aux mots de darija latine (3otla, n9der, 7ta) sont des
# lettres de translittération, pas des valeurs critiques.
NUMBER = re.compile(r"(?<![A-Za-z0-9])(\d+(?:[.,]\d+)?)(?:h|j|jours?)?(?![A-Za-z0-9])", re.IGNORECASE)
SOURCE_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


class TargetClaim(BaseModel):
    language: Literal["fr", "ar", "darija_latn", "darija_arab"]
    claims: list[str] = Field(min_length=1, max_length=3)


class ChunkTarget(BaseModel):
    chunk_id: str
    targets: list[TargetClaim] = Field(min_length=4, max_length=4)


class TargetBatch(BaseModel):
    chunks: list[ChunkTarget] = Field(min_length=1)


class LatinCorrection(BaseModel):
    claims: list[str] = Field(min_length=1, max_length=3)


ABSTENTIONS = {
    "fr": "Les passages fournis ne permettent pas de confirmer cette information.",
    "ar": "لا تسمح المقاطع المقدمة بتأكيد هذه المعلومة.",
    "darija_latn": "L passages li t3taw ma kaykfiwch bach n2ekked lik had lma3louma.",
    "darija_arab": "المقاطع اللي تعطاو ما كافيينش باش نأكد ليك هاد المعلومة.",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_targets(chunks: list[dict[str, Any]], required_ids: set[str]) -> None:
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY absente de .env")
    from google import genai

    existing = {row["chunk_id"]: row for row in read_jsonl(TARGET_CACHE)} if TARGET_CACHE.exists() else {}
    pending = [chunk for chunk in chunks if chunk["chunk_id"] in required_ids and chunk["chunk_id"] not in existing]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in pending:
        grouped[chunk["metadata"]["provider_id"]].append(chunk)
    client = genai.Client(api_key=api_key)
    for provider, provider_chunks in grouped.items():
        payload = [{"chunk_id": chunk["chunk_id"], "title": chunk["metadata"]["title"], "text": chunk["text"]} for chunk in provider_chunks]
        prompt = f"""Crée des cibles factuelles courtes pour QraLia à partir des passages officiels ci-dessous.

Pour chaque chunk, retourne exactement quatre TargetClaim, un pour fr, ar,
darija_latn et darija_arab. Chaque langue contient 1 à 3 affirmations atomiques
qui couvrent les faits utiles du passage.

Règles absolues:
- aucune connaissance externe et aucun fait absent du passage;
- conserve exactement chaque nombre, durée, pourcentage, horaire et identifiant;
- darija_latn uniquement en alphabet latin et chiffres, avec ZERO caractère arabe;
- darija_arab principalement en écriture arabe;
- phrases courtes, naturelles et autonomes;
- ne mets aucune citation dans le texte: le code les ajoutera;
- ne transforme pas une possibilité en obligation ni une liste en exclusivité.

FOURNISSEUR: {provider}
CHUNKS:
{json.dumps(payload, ensure_ascii=False)}
"""
        response = None
        for attempt, delay in enumerate((0, 5, 15), start=1):
            if delay:
                time.sleep(delay)
            try:
                response = client.models.generate_content(model=MODEL, contents=prompt, config={"temperature": 0.2, "response_mime_type": "application/json", "response_schema": TargetBatch})
                break
            except Exception:
                if attempt == 3:
                    raise
        parsed = response.parsed
        batch = parsed if isinstance(parsed, TargetBatch) else TargetBatch.model_validate(parsed)
        expected = {chunk["chunk_id"] for chunk in provider_chunks}
        if {item.chunk_id for item in batch.chunks} != expected:
            raise ValueError(f"Chunks cibles incomplets pour {provider}")
        for item in batch.chunks:
            language_map = {target.language: target.claims for target in item.targets}
            if set(language_map) != set(LANGUAGES):
                raise ValueError(f"Langues incomplètes pour {item.chunk_id}")
            source_numbers = set(SOURCE_NUMBER.findall(next(chunk["text"] for chunk in provider_chunks if chunk["chunk_id"] == item.chunk_id)))
            for language, claims in language_map.items():
                if language == "darija_latn" and any(ARABIC.search(claim) for claim in claims):
                    correction = client.models.generate_content(
                        model=MODEL,
                        contents=(
                            "Réécris ces affirmations en darija marocaine avec uniquement "
                            "l'alphabet latin et les chiffres de translittération. Zéro caractère "
                            "arabe. Conserve strictement les faits et nombres, sans ajout: "
                            + json.dumps(claims, ensure_ascii=False)
                        ),
                        config={"temperature": 0, "response_mime_type": "application/json", "response_schema": LatinCorrection},
                    )
                    parsed_correction = correction.parsed
                    corrected = parsed_correction if isinstance(parsed_correction, LatinCorrection) else LatinCorrection.model_validate(parsed_correction)
                    claims = corrected.claims
                    language_map[language] = claims
                    if any(ARABIC.search(claim) for claim in claims):
                        raise ValueError(f"Arabe persistant dans darija_latn: {item.chunk_id}")
                generated_numbers = {number for claim in claims for number in NUMBER.findall(claim)}
                if not generated_numbers.issubset(source_numbers):
                    raise ValueError(f"Nombre inventé dans {item.chunk_id}: {generated_numbers - source_numbers}")
            existing[item.chunk_id] = {"chunk_id": item.chunk_id, "provider_id": provider, "model": MODEL, "prompt_version": "target-v0.4.0", "claims": language_map, "review_status": "candidate_human_review"}
        write_jsonl(TARGET_CACHE, [existing[key] for key in sorted(existing)])
        print(f"generated targets for {provider}: {len(provider_chunks)} chunks", flush=True)


def make_answer(question: dict[str, Any], target_by_chunk: dict[str, dict[str, Any]]) -> GroundedAnswer:
    language = question["language"]
    if question["should_abstain"]:
        return GroundedAnswer(answer=ABSTENTIONS[language], claims=[], abstain=True, abstention_reason=ABSTENTIONS[language], language=language)
    claims = []
    for chunk_id in question["gold_chunk_ids"]:
        for text in target_by_chunk[chunk_id]["claims"][language]:
            claims.append(AnswerClaim(text=text, citations=[chunk_id]))
    return render_answer_from_claims(GroundedAnswer(answer="", claims=claims, abstain=False, abstention_reason=None, language=language))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-targets", action="store_true")
    args = parser.parse_args()
    questions = read_jsonl(QUESTIONS)
    chunks = [chunk for path in CHUNK_PATHS for chunk in load_active_chunks(path, CHUNK_SCHEMA)]
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    required_ids = {chunk_id for question in questions for chunk_id in question["gold_chunk_ids"]}
    if args.generate_targets:
        generate_targets(chunks, required_ids)
    target_by_chunk = {row["chunk_id"]: row for row in read_jsonl(TARGET_CACHE)}
    missing = required_ids - set(target_by_chunk)
    if missing:
        raise RuntimeError(f"Cibles manquantes: {sorted(missing)}. Lancez --generate-targets.")
    validator = Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8")))

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []
    test_prompts: list[dict[str, Any]] = []
    for question in questions:
        contexts = [chunks_by_id[chunk_id] for chunk_id in question["context_chunk_ids"]]
        case = {"language": question["language"], "provider_id": question["provider_id"], "question": question["question"]}
        user_prompt = build_generation_prompt(case, contexts, "darija_few_shot")
        if question["split"] == "test":
            tests.append(question)
            test_prompts.append({"case_id": question["case_id"], "messages": [{"role": "user", "content": user_prompt}]})
            continue
        answer = make_answer(question, target_by_chunk)
        critical_values = sorted({number for claim in answer.claims for number in NUMBER.findall(claim.text)})
        record = {"schema_version": "0.4.0", "sample_id": question["case_id"].replace("sftq-", "sft-"), "source_question_case_id": question["case_id"], "intent_id": question["intent_id"], "semantic_family_id": question["semantic_family_id"], "language": question["language"], "provider_id": question["provider_id"], "formulation_kind": question["formulation_kind"], "split": question["split"], "retrieval_outcome": question["retrieval_outcome"], "should_abstain": question["should_abstain"], "gold_chunk_ids": question["gold_chunk_ids"], "context_chunk_ids": question["context_chunk_ids"], "critical_values": critical_values, "review_status": "candidate_human_review", "messages": [{"role": "user", "content": user_prompt}, {"role": "assistant", "content": answer.model_dump_json()}]}
        validator.validate(record)
        (train if question["split"] == "train" else validation).append(record)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {name: OUTPUT_DIR / f"{name}.jsonl" for name in ("train", "validation", "test_cases", "test_prompts", "review_queue")}
    write_jsonl(paths["train"], train)
    write_jsonl(paths["validation"], validation)
    write_jsonl(paths["test_cases"], tests)
    write_jsonl(paths["test_prompts"], test_prompts)

    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in train + validation:
        answer = json.loads(record["messages"][1]["content"])
        key = (record["intent_id"], record["language"], record["retrieval_outcome"], tuple(record["gold_chunk_ids"]))
        group = groups.setdefault(key, {"review_group_id": hashlib.sha256(repr(key).encode()).hexdigest()[:16], "intent_id": record["intent_id"], "language": record["language"], "retrieval_outcome": record["retrieval_outcome"], "gold_chunk_ids": record["gold_chunk_ids"], "questions": [], "assistant_answer": answer["answer"], "human_decision": None, "human_comment": ""})
        group["questions"].append({"sample_id": record["sample_id"], "formulation_kind": record["formulation_kind"], "question": next(q["question"] for q in questions if q["case_id"] == record["source_question_case_id"])})
    write_jsonl(paths["review_queue"], list(groups.values()))

    manifest = {"schema_version": "0.4.0", "status": "candidate_human_review", "base_model": "Qwen/Qwen3.5-4B", "target_models": sorted({row["model"] for row in target_by_chunk.values()}), "counts": {"train": len(train), "validation": len(validation), "test": len(tests), "review_groups": len(groups)}, "train_abstention_count": sum(r["should_abstain"] for r in train), "validation_abstention_count": sum(r["should_abstain"] for r in validation), "test_policy": "Le test ne contient aucune réponse assistant et n'est jamais passé à swift sft.", "files": {name: {"path": path.name, "sha256": sha256(path)} for name, path in paths.items()}}
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
