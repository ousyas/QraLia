"""Recherche sémantique LangChain avec Qwen3 et évaluation reproductible.

BM25 reste un moteur indépendant. Ce module convertit les chunks QraLia en
``Document`` LangChain, les indexe dans ``InMemoryVectorStore`` et évalue la
similarité cosinus avec le même protocole que la baseline lexicale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from qralia.rag.bm25_evaluation import (
    abstention_metrics,
    calibrate_threshold,
    grouped_ranking_metrics,
    load_active_chunks,
    load_evaluation_cases,
    ranking_metrics,
)


DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
# Une révision explicite rend une expérience reproductible même si le dépôt du
# modèle évolue. Elle reste surchargeable depuis la ligne de commande.
DEFAULT_MODEL_REVISION = "3d106eabb5535a84de3ae88f45887a78259b52de"


class Qwen3LangChainEmbeddings(Embeddings):
    """Adaptateur local Qwen3 conforme au contrat ``Embeddings`` de LangChain."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        revision: str | None = DEFAULT_MODEL_REVISION,
        device: str = "cpu",
        batch_size: int = 8,
        max_seq_length: int = 1024,
        local_files_only: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size doit être au moins égal à 1.")
        if max_seq_length < 1:
            raise ValueError("max_seq_length doit être au moins égal à 1.")
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Dépendances sémantiques absentes. Installez : "
                ".venv\\Scripts\\python.exe -m pip install -e \".[semantic]\""
            ) from exc

        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self._model = SentenceTransformer(
            model_name,
            revision=revision,
            device=device,
            local_files_only=local_files_only,
        )
        # Nos chunks sont courts. Cette borne réduit le coût et rend la
        # troncature explicite au lieu de dépendre d'un défaut du modèle.
        self._model.max_seq_length = max_seq_length

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Encode les passages, sans prompt de requête."""

        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Encode une question avec le prompt de retrieval fourni par Qwen."""

        vector = self._model.encode(
            text,
            prompt_name="query",
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vector.tolist()


def searchable_text(chunk: dict[str, Any]) -> str:
    """Construit le passage encodé : contexte hiérarchique puis texte officiel."""

    headings = chunk["metadata"].get("heading_path", [])
    return " > ".join(headings) + "\n\n" + chunk["text"]


def chunk_to_document(chunk: dict[str, Any]) -> Document:
    """Convertit un chunk validé en objet LangChain sans perdre sa provenance."""

    metadata = dict(chunk["metadata"])
    metadata.update(
        {
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "source_id": chunk["source_id"],
            "section_id": chunk["section_id"],
        }
    )
    return Document(
        id=chunk["chunk_id"],
        page_content=searchable_text(chunk),
        metadata=metadata,
    )


class LangChainSemanticIndex:
    """Index dense local avec filtres de sécurité appliqués avant le classement."""

    def __init__(self, chunks: Sequence[dict[str, Any]], embeddings: Embeddings) -> None:
        if not chunks:
            raise ValueError("La recherche sémantique exige au moins un chunk actif.")
        self.chunks = tuple(chunks)
        self.embeddings = embeddings
        self.vector_store = InMemoryVectorStore(embedding=embeddings)
        documents = [chunk_to_document(chunk) for chunk in chunks]
        self.vector_store.add_documents(
            documents=documents,
            ids=[chunk["chunk_id"] for chunk in chunks],
        )

    @staticmethod
    def _route_filter(
        *, audience: str, distributor_id: str, service_zone_id: str,
        domain: str | None = None, service_type: str | None = None,
    ):
        def allowed(document: Document) -> bool:
            metadata = document.metadata
            return (
                metadata.get("validity_status") == "active"
                and metadata.get("provider_id", metadata.get("distributor_id")) == distributor_id
                and metadata.get("service_zone_id") == service_zone_id
                and audience in metadata.get("audiences", [])
                and (domain is None or metadata.get("domain") == domain)
                and (service_type is None or metadata.get("service_type") == service_type)
            )

        return allowed

    def search(
        self,
        query: str,
        *,
        audience: str,
        distributor_id: str,
        service_zone_id: str,
        domain: str | None = None,
        service_type: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k doit être au moins égal à 1.")

        matches = self.vector_store.similarity_search_with_score(
            query=query,
            k=top_k,
            filter=self._route_filter(
                audience=audience,
                distributor_id=distributor_id,
                service_zone_id=service_zone_id,
                domain=domain,
                service_type=service_type,
            ),
        )
        return [
            {
                "chunk_id": document.metadata["chunk_id"],
                "section_id": document.metadata["section_id"],
                "score": float(score),
                "title": document.metadata["title"],
                "canonical_url": document.metadata["canonical_url"],
                "provider_id": document.metadata.get("provider_id", document.metadata.get("distributor_id")),
                "provider_name": document.metadata.get("provider_name", document.metadata.get("distributor_name")),
                "service_zone_id": document.metadata["service_zone_id"],
                "retriever": "semantic_qwen3_langchain",
            }
            for document, score in matches
        ]


def evaluate_case(
    index: LangChainSemanticIndex,
    case: dict[str, Any],
    *,
    top_k: int,
) -> dict[str, Any]:
    started_at = perf_counter()
    ranking = index.search(
        case["question"],
        audience=case["audience"],
        distributor_id=case["distributor_id"],
        service_zone_id=case["service_zone_id"],
        top_k=top_k,
    )
    latency_ms = (perf_counter() - started_at) * 1000.0
    relevant = set(case["relevant_section_ids"])
    retrieved = [result["section_id"] for result in ranking]
    relevant_ranks = [
        rank
        for rank, section_id in enumerate(retrieved, start=1)
        if section_id in relevant
    ]
    recalls = {
        str(k): len(relevant.intersection(retrieved[:k])) / len(relevant)
        if relevant
        else None
        for k in (1, 3, 5)
    }
    binary_recalls = {
        str(k): float(bool(relevant.intersection(retrieved[:k])))
        if relevant
        else None
        for k in (1, 3, 5)
    }
    record = {
        "case_id": case["case_id"],
        "question": case["question"],
        "language": case["language"],
        "category": case["category"],
        "audience": case["audience"],
        "distributor_id": case["distributor_id"],
        "service_zone_id": case["service_zone_id"],
        "split": case["split"],
        "should_abstain": case["should_abstain"],
        "relevant_section_ids": case["relevant_section_ids"],
        "ranking": ranking,
        "top_score": ranking[0]["score"] if ranking else 0.0,
        "has_candidates": bool(ranking),
        "recall_at_k": recalls,
        "binary_recall_at_k": binary_recalls,
        "reciprocal_rank": 1.0 / min(relevant_ranks) if relevant_ranks else 0.0,
        "latency_ms": latency_ms,
    }
    if "pair_id" in case:
        record["pair_id"] = case["pair_id"]
    return record


def paired_script_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compare directement les variantes latine et arabe d'une même intention."""

    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        pair_id = record.get("pair_id")
        if pair_id:
            groups.setdefault(pair_id, {})[record["language"]] = record

    complete = [
        variants
        for variants in groups.values()
        if {"darija_latn", "darija_arab"}.issubset(variants)
    ]
    positive = [
        variants
        for variants in complete
        if not variants["darija_latn"]["should_abstain"]
        and not variants["darija_arab"]["should_abstain"]
    ]
    arab_wins = sum(
        variants["darija_arab"]["reciprocal_rank"]
        > variants["darija_latn"]["reciprocal_rank"]
        for variants in positive
    )
    latin_wins = sum(
        variants["darija_latn"]["reciprocal_rank"]
        > variants["darija_arab"]["reciprocal_rank"]
        for variants in positive
    )
    ties = len(positive) - arab_wins - latin_wins

    accuracy_by_script: dict[str, float | None] = {}
    for language in ("darija_latn", "darija_arab"):
        language_records = [variants[language] for variants in complete]
        accuracy_by_script[language] = (
            mean(
                record["predicted_abstain"] == record["should_abstain"]
                for record in language_records
            )
            if language_records
            else None
        )

    return {
        "complete_pair_count": len(complete),
        "positive_pair_count": len(positive),
        "ranking_wins_by_reciprocal_rank": {
            "darija_arab": arab_wins,
            "darija_latn": latin_wins,
            "ties": ties,
        },
        "mean_reciprocal_rank_difference_arab_minus_latin": (
            mean(
                variants["darija_arab"]["reciprocal_rank"]
                - variants["darija_latn"]["reciprocal_rank"]
                for variants in positive
            )
            if positive
            else None
        ),
        "abstention_accuracy_by_script": accuracy_by_script,
    }


def build_report(
    index: LangChainSemanticIndex,
    cases: Sequence[dict[str, Any]],
    *,
    top_k: int,
    model_name: str,
    model_revision: str | None,
    device: str | None = None,
    batch_size: int | None = None,
    max_seq_length: int | None = None,
    model_load_seconds: float | None = None,
    index_build_seconds: float | None = None,
) -> dict[str, Any]:
    records = [
        evaluate_case(
            index,
            case,
            top_k=top_k,
        )
        for case in cases
    ]
    calibration = [record for record in records if record["split"] == "calibration"]
    test = [record for record in records if record["split"] == "test"]
    threshold, calibration_abstention = calibrate_threshold(calibration)
    for record in records:
        record["predicted_abstain"] = not (
            record["has_candidates"] and record["top_score"] >= threshold
        )
    latencies = sorted(float(record["latency_ms"]) for record in records)
    p95_index = max(0, min(len(latencies) - 1, (95 * len(latencies) + 99) // 100 - 1))

    return {
        "schema_version": "0.1.0",
        "configuration": {
            "algorithm": "dense_cosine_similarity",
            "orchestration": "langchain-core/InMemoryVectorStore",
            "embedding_model": model_name,
            "model_revision": model_revision,
            "query_prompt": "query",
            "normalized_embeddings": True,
            "device": device,
            "batch_size": batch_size,
            "max_seq_length": max_seq_length,
            "routes": sorted(
                {
                    f"{case['distributor_id']}::{case['service_zone_id']}"
                    for case in cases
                }
            ),
            "top_k": top_k,
            "indexed_chunk_count": len(index.chunks),
            "indexed_fields": ["metadata.heading_path", "text"],
            "model_load_seconds": model_load_seconds,
            "index_build_seconds": index_build_seconds,
        },
        "calibration": {
            "case_count": len(calibration),
            "selected_threshold": threshold,
            "abstention_metrics": calibration_abstention,
        },
        "test": {
            "case_count": len(test),
            "ranking_metrics": ranking_metrics(test),
            "abstention_metrics": abstention_metrics(test, threshold),
            "ranking_by_language": grouped_ranking_metrics(test, "language"),
            "ranking_by_category": grouped_ranking_metrics(test, "category"),
            "paired_script_metrics": paired_script_metrics(test),
        },
        "latency": {
            "case_count": len(latencies),
            "mean_ms": mean(latencies),
            "p95_ms": latencies[p95_index],
            "max_ms": latencies[-1],
        },
        "cases": records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Évalue la recherche sémantique Qwen3 orchestrée avec LangChain."
    )
    parser.add_argument(
        "--chunks", required=True, action="append", type=Path,
        help="JSONL de chunks ; l'option peut être répétée.",
    )
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--schemas-dir", type=Path, default=Path("schemas"))
    parser.add_argument("--distributor-id", help="Optionnel : filtre les cas par distributeur.")
    parser.add_argument("--service-zone-id", help="Optionnel : filtre les cas par zone.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    chunks: list[dict[str, Any]] = []
    for path in args.chunks:
        chunks.extend(
            load_active_chunks(path, args.schemas_dir / "retrieval_chunk.schema.json")
        )
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Les chunk_id doivent être uniques dans l'index sémantique.")

    cases = load_evaluation_cases(
        args.evaluation,
        args.schemas_dir / "retrieval_evaluation_case.schema.json",
    )
    if bool(args.distributor_id) != bool(args.service_zone_id):
        raise ValueError("--distributor-id et --service-zone-id doivent être fournis ensemble.")
    if args.distributor_id:
        cases = [
            case for case in cases
            if case["distributor_id"] == args.distributor_id
            and case["service_zone_id"] == args.service_zone_id
        ]
        if not cases:
            raise ValueError("Aucun cas d'évaluation ne correspond à la route demandée.")

    active_routes = {
        (
            chunk["section_id"],
            chunk["metadata"].get("provider_id", chunk["metadata"].get("distributor_id")),
            chunk["metadata"]["service_zone_id"],
            audience,
        )
        for chunk in chunks
        for audience in chunk["metadata"]["audiences"]
    }
    unknown = sorted(
        {
            (section_id, case["distributor_id"], case["service_zone_id"], case["audience"])
            for case in cases
            for section_id in case["relevant_section_ids"]
            if (
                section_id,
                case["distributor_id"],
                case["service_zone_id"],
                case["audience"],
            ) not in active_routes
        }
    )
    if unknown:
        raise ValueError(f"Sections attendues absentes de la route active : {unknown}")

    model_started_at = perf_counter()
    embeddings = Qwen3LangChainEmbeddings(
        model_name=args.model_name,
        revision=args.model_revision,
        device=args.device,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
    )
    model_load_seconds = perf_counter() - model_started_at
    index_started_at = perf_counter()
    index = LangChainSemanticIndex(chunks, embeddings)
    index_build_seconds = perf_counter() - index_started_at
    report = build_report(
        index,
        cases,
        top_k=args.top_k,
        model_name=args.model_name,
        model_revision=args.model_revision,
        device=args.device,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        model_load_seconds=model_load_seconds,
        index_build_seconds=index_build_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    ranking = report["test"]["ranking_metrics"]
    abstention = report["test"]["abstention_metrics"]
    print(f"Rapport : {args.output}")
    print(f"Recall@5 : {ranking['recall_at_5']:.3f}")
    print(f"Binary Recall@5 : {ranking['binary_recall_at_5']:.3f}")
    print(f"MRR : {ranking['mrr']:.3f}")
    print(f"Seuil d'abstention : {report['calibration']['selected_threshold']:.6f}")
    print(f"Balanced accuracy d'abstention : {abstention['balanced_accuracy']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
