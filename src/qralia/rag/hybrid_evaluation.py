"""Recherche hybride BM25 normalisé + Qwen3 et évaluation commune.

La fusion utilise Weighted Reciprocal Rank Fusion (RRF). Elle combine les rangs
plutôt que les scores bruts, car un score BM25 et une similarité cosinus n'ont
ni la même unité ni la même distribution.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Sequence

from langchain_core.embeddings import Embeddings

from qralia.rag.bm25_evaluation import (
    BM25Index,
    abstention_metrics,
    calibrate_threshold,
    grouped_ranking_metrics,
    load_active_chunks,
    load_jsonl,
    ranking_metrics,
)
from qralia.rag.onee_corpus import validate_with_schema
from qralia.rag.semantic_evaluation import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_REVISION,
    LangChainSemanticIndex,
    Qwen3LangChainEmbeddings,
    paired_script_metrics,
)


def _provider_id(case: dict[str, Any]) -> str:
    return str(case.get("provider_id", case.get("distributor_id")))


def canonicalize_case(case: dict[str, Any], chunks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Projette les cas historiques 0.2 vers le routage générique 0.3 en mémoire."""

    canonical = dict(case)
    canonical["provider_id"] = _provider_id(case)
    if "domain" in canonical and "service_type" in canonical:
        return canonical

    possible = {
        (chunk["metadata"]["domain"], chunk["metadata"]["service_type"])
        for chunk in chunks
        if chunk["metadata"]["provider_id"] == canonical["provider_id"]
        and chunk["metadata"]["service_zone_id"] == canonical["service_zone_id"]
    }
    if not possible and canonical["provider_id"].startswith("srm_"):
        # Hard negatives régionaux volontairement sans chunks actifs. Leur
        # domaine reste connu même si aucune source de cette route n'est indexée.
        possible = {("utilities", "multi_utility")}
    elif not possible and canonical["provider_id"] == "onee":
        possible = {("utilities", "electricity")}
    if len(possible) != 1:
        raise ValueError(
            f"Route historique ambiguë pour {case['case_id']} : {sorted(possible)}"
        )
    canonical["domain"], canonical["service_type"] = next(iter(possible))
    return canonical


def load_evaluation_files(
    paths: Sequence[Path], *, schemas_dir: Path, chunks: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Charge ensemble les jeux gelés 0.2 et le nouveau jeu générique 0.3."""

    schemas = {
        "0.2.0": schemas_dir / "retrieval_evaluation_case.schema.json",
        "0.3.0": schemas_dir / "retrieval_evaluation_case_v0.3.schema.json",
    }
    cases: list[dict[str, Any]] = []
    for path in paths:
        for case in load_jsonl(path):
            version = case.get("schema_version")
            if version not in schemas:
                raise ValueError(f"Version de cas non prise en charge : {version!r} dans {path}.")
            validate_with_schema(case, schemas[version])
            cases.append(canonicalize_case(case, chunks))
    ids = [case["case_id"] for case in cases]
    if len(ids) != len(set(ids)):
        duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
        raise ValueError(f"case_id dupliqués entre les jeux : {duplicates}")
    return cases


class HybridRetrievalIndex:
    """Deux moteurs indépendants et une fusion RRF strictement routée."""

    def __init__(
        self,
        chunks: Sequence[dict[str, Any]],
        embeddings: Embeddings,
        *,
        bm25_weight: float = 0.5,
        semantic_weight: float = 0.5,
        rrf_k: int = 60,
        candidate_k: int = 20,
    ) -> None:
        if bm25_weight < 0 or semantic_weight < 0 or bm25_weight + semantic_weight <= 0:
            raise ValueError("Les poids doivent être positifs et leur somme non nulle.")
        if rrf_k < 1 or candidate_k < 1:
            raise ValueError("rrf_k et candidate_k doivent être au moins égaux à 1.")
        total = bm25_weight + semantic_weight
        self.bm25_weight = bm25_weight / total
        self.semantic_weight = semantic_weight / total
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k
        self.chunks = tuple(chunks)
        self.by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
        self.bm25 = BM25Index(chunks, normalization="darija_lexicon")
        self.semantic = LangChainSemanticIndex(chunks, embeddings)

    def search_components(
        self,
        query: str,
        *,
        domain: str,
        provider_id: str,
        service_type: str,
        service_zone_id: str,
        audience: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        route = {
            "domain": domain,
            "distributor_id": provider_id,
            "service_type": service_type,
            "service_zone_id": service_zone_id,
            "audience": audience,
        }
        started = perf_counter()
        bm25 = self.bm25.search(query, top_k=self.candidate_k, **route)
        bm25_ms = (perf_counter() - started) * 1000.0
        started = perf_counter()
        semantic = self.semantic.search(query, top_k=self.candidate_k, **route)
        semantic_ms = (perf_counter() - started) * 1000.0

        bm25_by_id = {item["chunk_id"]: (rank, item) for rank, item in enumerate(bm25, 1)}
        semantic_by_id = {item["chunk_id"]: (rank, item) for rank, item in enumerate(semantic, 1)}
        fused: list[dict[str, Any]] = []
        for chunk_id in bm25_by_id.keys() | semantic_by_id.keys():
            bm = bm25_by_id.get(chunk_id)
            sem = semantic_by_id.get(chunk_id)
            rrf_score = (
                self.bm25_weight / (self.rrf_k + bm[0]) if bm else 0.0
            ) + (
                self.semantic_weight / (self.rrf_k + sem[0]) if sem else 0.0
            )
            bm25_raw = float(bm[1]["score"]) if bm else 0.0
            semantic_raw = float(sem[1]["score"]) if sem else 0.0
            # Cette confiance sert uniquement à calibrer l'abstention. Le rang
            # reste déterminé par RRF. Les transformations bornent les branches
            # sur [0,1] sans les confondre avec une probabilité.
            bm25_confidence = 1.0 - math.exp(-max(0.0, bm25_raw))
            semantic_confidence = max(0.0, min(1.0, semantic_raw))
            confidence = (
                self.bm25_weight * bm25_confidence
                + self.semantic_weight * semantic_confidence
            )
            chunk = self.by_id[chunk_id]
            fused.append({
                "chunk_id": chunk_id,
                "section_id": chunk["section_id"],
                "score": rrf_score,
                "confidence": confidence,
                "bm25_rank": bm[0] if bm else None,
                "bm25_score": bm25_raw if bm else None,
                "semantic_rank": sem[0] if sem else None,
                "semantic_score": semantic_raw if sem else None,
                "title": chunk["metadata"]["title"],
                "canonical_url": chunk["metadata"]["canonical_url"],
                "provider_id": chunk["metadata"]["provider_id"],
                "service_zone_id": chunk["metadata"]["service_zone_id"],
                "retriever": "weighted_rrf",
            })
        fused.sort(key=lambda item: (-item["score"], -item["confidence"], item["chunk_id"]))
        return {
            "bm25": bm25[:top_k],
            "semantic": semantic[:top_k],
            "hybrid": fused[:top_k],
            "latency_ms": {
                "bm25": bm25_ms,
                "semantic": semantic_ms,
                "hybrid_fusion": (perf_counter() - started) * 1000.0 - semantic_ms,
            },
        }


def _record(
    case: dict[str, Any], ranking: list[dict[str, Any]], *, system: str, latency_ms: float
) -> dict[str, Any]:
    relevant = set(case["relevant_section_ids"])
    retrieved = [item["section_id"] for item in ranking]
    ranks = [rank for rank, section in enumerate(retrieved, 1) if section in relevant]
    top_score = 0.0
    if ranking:
        top_score = float(
            ranking[0].get("confidence", ranking[0].get("score", 0.0))
        )
    record = {
        "case_id": case["case_id"],
        "question": case["question"],
        "language": case["language"],
        "category": case["category"],
        "domain": case["domain"],
        "provider_id": case["provider_id"],
        "service_type": case["service_type"],
        "service_zone_id": case["service_zone_id"],
        "audience": case["audience"],
        "split": case["split"],
        "should_abstain": case["should_abstain"],
        "relevant_section_ids": case["relevant_section_ids"],
        "ranking": ranking,
        "top_score": top_score,
        "has_candidates": bool(ranking),
        "recall_at_k": {
            str(k): len(relevant.intersection(retrieved[:k])) / len(relevant) if relevant else None
            for k in (1, 3, 5)
        },
        "binary_recall_at_k": {
            str(k): float(bool(relevant.intersection(retrieved[:k]))) if relevant else None
            for k in (1, 3, 5)
        },
        "reciprocal_rank": 1.0 / min(ranks) if ranks else 0.0,
        "latency_ms": latency_ms,
        "system": system,
    }
    if "pair_id" in case:
        record["pair_id"] = case["pair_id"]
    return record


def evaluate_all(
    index: HybridRetrievalIndex, cases: Sequence[dict[str, Any]], *, top_k: int
) -> dict[str, list[dict[str, Any]]]:
    records = {"bm25": [], "semantic": [], "hybrid": []}
    for case in cases:
        result = index.search_components(
            case["question"],
            domain=case["domain"],
            provider_id=case["provider_id"],
            service_type=case["service_type"],
            service_zone_id=case["service_zone_id"],
            audience=case["audience"],
            top_k=top_k,
        )
        records["bm25"].append(_record(case, result["bm25"], system="bm25", latency_ms=result["latency_ms"]["bm25"]))
        records["semantic"].append(_record(case, result["semantic"], system="semantic", latency_ms=result["latency_ms"]["semantic"]))
        records["hybrid"].append(_record(case, result["hybrid"], system="hybrid", latency_ms=sum(result["latency_ms"].values())))
    return records


def _system_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    calibration = [record for record in records if record["split"] == "calibration"]
    test = [record for record in records if record["split"] == "test"]
    threshold, calibration_metrics = calibrate_threshold(calibration)
    for record in records:
        record["predicted_abstain"] = not (
            record["has_candidates"] and record["top_score"] >= threshold
        )
    latencies = sorted(float(record["latency_ms"]) for record in records)
    p95_index = max(0, min(len(latencies) - 1, math.ceil(0.95 * len(latencies)) - 1))
    return {
        "calibration": {
            "case_count": len(calibration),
            "selected_threshold": threshold,
            "abstention_metrics": calibration_metrics,
        },
        "test": {
            "case_count": len(test),
            "ranking_metrics": ranking_metrics(test),
            "abstention_metrics": abstention_metrics(test, threshold),
            "ranking_by_language": grouped_ranking_metrics(test, "language"),
            "ranking_by_category": grouped_ranking_metrics(test, "category"),
            "ranking_by_domain": grouped_ranking_metrics(test, "domain"),
            "ranking_by_provider": grouped_ranking_metrics(test, "provider_id"),
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


def validate_expected_sections(cases: Sequence[dict[str, Any]], chunks: Sequence[dict[str, Any]]) -> None:
    active = {
        (
            chunk["metadata"]["domain"], chunk["metadata"]["provider_id"],
            chunk["metadata"]["service_type"], chunk["metadata"]["service_zone_id"],
            audience, chunk["section_id"],
        )
        for chunk in chunks for audience in chunk["metadata"]["audiences"]
    }
    missing = sorted({
        (case["domain"], case["provider_id"], case["service_type"], case["service_zone_id"], case["audience"], section)
        for case in cases for section in case["relevant_section_ids"]
        if (case["domain"], case["provider_id"], case["service_type"], case["service_zone_id"], case["audience"], section) not in active
    })
    if missing:
        raise ValueError(f"Sections attendues absentes des routes actives : {missing}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare BM25 normalisé, Qwen3 et leur fusion RRF sur les mêmes cas.")
    parser.add_argument("--chunks", required=True, action="append", type=Path)
    parser.add_argument("--evaluation", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--schemas-dir", type=Path, default=Path("schemas"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--bm25-weight", type=float, default=0.5)
    parser.add_argument("--semantic-weight", type=float, default=0.5)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    chunks = [
        chunk for path in args.chunks
        for chunk in load_active_chunks(path, args.schemas_dir / "retrieval_chunk.schema.json")
    ]
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Les chunk_id doivent être uniques dans l'index hybride.")
    cases = load_evaluation_files(args.evaluation, schemas_dir=args.schemas_dir, chunks=chunks)
    validate_expected_sections(cases, chunks)

    started = perf_counter()
    embeddings = Qwen3LangChainEmbeddings(
        model_name=args.model_name, revision=args.model_revision, device=args.device,
        batch_size=args.batch_size, max_seq_length=args.max_seq_length,
    )
    model_load_seconds = perf_counter() - started
    started = perf_counter()
    index = HybridRetrievalIndex(
        chunks, embeddings, bm25_weight=args.bm25_weight,
        semantic_weight=args.semantic_weight, rrf_k=args.rrf_k,
        candidate_k=args.candidate_k,
    )
    index_build_seconds = perf_counter() - started
    records = evaluate_all(index, cases, top_k=args.top_k)
    report = {
        "schema_version": "0.1.0",
        "configuration": {
            "algorithm": "weighted_reciprocal_rank_fusion",
            "bm25_normalization": "darija_lexicon",
            "embedding_model": args.model_name,
            "model_revision": args.model_revision,
            "device": args.device,
            "bm25_weight": index.bm25_weight,
            "semantic_weight": index.semantic_weight,
            "rrf_k": args.rrf_k,
            "candidate_k": args.candidate_k,
            "top_k": args.top_k,
            "indexed_chunk_count": len(chunks),
            "evaluation_case_count": len(cases),
            "model_load_seconds": model_load_seconds,
            "index_build_seconds": index_build_seconds,
        },
        "systems": {name: _system_report(system_records) for name, system_records in records.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Rapport : {args.output}")
    print(f"Chunks indexés : {len(chunks)} ; cas évalués : {len(cases)}")
    for name in ("bm25", "semantic", "hybrid"):
        metrics = report["systems"][name]["test"]["ranking_metrics"]
        abstention = report["systems"][name]["test"]["abstention_metrics"]
        print(
            f"{name}: Recall@5={metrics['recall_at_5']:.3f} "
            f"BinaryRecall@5={metrics['binary_recall_at_5']:.3f} "
            f"MRR={metrics['mrr']:.3f} "
            f"AbstentionBA={abstention['balanced_accuracy']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
