"""Évalue un reranker Qwen3 au-dessus des candidats de la recherche hybride.

Le bi-encodeur Qwen3-Embedding sert à retrouver rapidement un petit ensemble de
candidats. Le cross-encoder Qwen3-Reranker relit ensuite chaque paire
``(question, passage)`` et réordonne uniquement ces candidats. Les modèles sont
chargés successivement afin de limiter la mémoire GPU nécessaire.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Sequence

from qralia.rag.bm25_evaluation import load_active_chunks
from qralia.rag.hybrid_evaluation import (
    HybridRetrievalIndex,
    _record,
    _system_report,
    load_evaluation_files,
    validate_expected_sections,
)
from qralia.rag.semantic_evaluation import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_REVISION,
    Qwen3LangChainEmbeddings,
    searchable_text,
)


DEFAULT_RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_RERANKER_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"


class Qwen3CrossEncoderReranker:
    """Adaptateur minimal autour du ``CrossEncoder`` de Sentence Transformers."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_RERANKER_MODEL,
        revision: str | None = DEFAULT_RERANKER_REVISION,
        device: str = "cpu",
        batch_size: int = 4,
        max_length: int = 1024,
        instruction: str | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size doit être au moins égal à 1.")
        if max_length < 1:
            raise ValueError("max_length doit être au moins égal à 1.")
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Dépendances sémantiques absentes. Installez : "
                '.venv\\Scripts\\python.exe -m pip install -e ".[semantic]"'
            ) from exc

        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        # float16 réduit fortement l'empreinte sur les GPU grand public. Sur CPU,
        # float32 évite les opérations half mal prises en charge.
        dtype = torch.float16 if device == "cuda" else torch.float32
        self._sigmoid = torch.nn.Sigmoid()
        prompt_kwargs = {}
        if instruction:
            prompt_kwargs = {
                "prompts": {"qralia": instruction},
                "default_prompt_name": "qralia",
            }
        self._model = CrossEncoder(
            model_name,
            revision=revision,
            device=device,
            max_length=max_length,
            model_kwargs={"torch_dtype": dtype},
            trust_remote_code=True,
            **prompt_kwargs,
        )

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """Retourne un score borné dans [0, 1] pour chaque paire question-passage."""

        if not pairs:
            return []
        values = self._model.predict(
            list(pairs),
            batch_size=self.batch_size,
            show_progress_bar=True,
            activation_fn=self._sigmoid,
        )
        return [float(value) for value in values]


def rerank_with_scores(
    candidates: Sequence[dict[str, Any]],
    scores: Sequence[float],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    """Réordonne des candidats déjà routés sans modifier leur ensemble."""

    if len(candidates) != len(scores):
        raise ValueError("Il faut exactement un score par candidat.")
    if top_k < 1:
        raise ValueError("top_k doit être au moins égal à 1.")

    reranked: list[dict[str, Any]] = []
    for hybrid_rank, (candidate, reranker_score) in enumerate(zip(candidates, scores), 1):
        item = dict(candidate)
        item["hybrid_rank"] = hybrid_rank
        item["hybrid_score"] = float(candidate["score"])
        item["hybrid_confidence"] = float(candidate["confidence"])
        item["reranker_score"] = float(reranker_score)
        # _record utilise confidence pour calibrer l'abstention. Ici elle est
        # réellement issue du sigmoid du cross-encoder.
        item["score"] = float(reranker_score)
        item["confidence"] = float(reranker_score)
        item["retriever"] = "weighted_rrf_then_qwen3_reranker"
        reranked.append(item)

    reranked.sort(
        key=lambda item: (-item["reranker_score"], item["hybrid_rank"], item["chunk_id"])
    )
    return reranked[:top_k]


def candidate_set_metrics(
    cases: Sequence[dict[str, Any]], candidates_by_case: Sequence[Sequence[dict[str, Any]]]
) -> dict[str, float | int]:
    """Mesure le plafond du reranker : il ne peut retrouver un candidat absent."""

    positive_recalls: list[float] = []
    binary_hits: list[float] = []
    for case, candidates in zip(cases, candidates_by_case):
        relevant = set(case["relevant_section_ids"])
        if not relevant:
            continue
        retrieved = {item["section_id"] for item in candidates}
        positive_recalls.append(len(relevant & retrieved) / len(relevant))
        binary_hits.append(float(bool(relevant & retrieved)))
    return {
        "positive_case_count": len(positive_recalls),
        "recall": mean(positive_recalls) if positive_recalls else 0.0,
        "binary_recall": mean(binary_hits) if binary_hits else 0.0,
    }


def collect_hybrid_candidates(
    index: HybridRetrievalIndex,
    cases: Sequence[dict[str, Any]],
    *,
    top_k: int,
    rerank_k: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[list[dict[str, Any]]], list[float]]:
    """Évalue les trois baselines et conserve le Top-N hybride pour le reranker."""

    records = {"bm25": [], "semantic": [], "hybrid": []}
    candidates_by_case: list[list[dict[str, Any]]] = []
    retrieval_latencies: list[float] = []
    for case in cases:
        result = index.search_components(
            case["question"],
            domain=case["domain"],
            provider_id=case["provider_id"],
            service_type=case["service_type"],
            service_zone_id=case["service_zone_id"],
            audience=case["audience"],
            top_k=rerank_k,
        )
        total_latency = sum(result["latency_ms"].values())
        candidates = result["hybrid"]
        candidates_by_case.append(candidates)
        retrieval_latencies.append(total_latency)
        records["bm25"].append(
            _record(case, result["bm25"][:top_k], system="bm25", latency_ms=result["latency_ms"]["bm25"])
        )
        records["semantic"].append(
            _record(case, result["semantic"][:top_k], system="semantic", latency_ms=result["latency_ms"]["semantic"])
        )
        records["hybrid"].append(
            _record(case, candidates[:top_k], system="hybrid", latency_ms=total_latency)
        )
    return records, candidates_by_case, retrieval_latencies


def release_gpu_memory(*objects: object) -> None:
    """Libère les modèles précédents avant de charger le cross-encoder."""

    del objects
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ModuleNotFoundError:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare BM25, Qwen3, hybride RRF et hybride + Qwen3-Reranker."
    )
    parser.add_argument("--chunks", required=True, action="append", type=Path)
    parser.add_argument("--evaluation", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--schemas-dir", type=Path, default=Path("schemas"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rerank-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--bm25-weight", type=float, default=0.5)
    parser.add_argument("--semantic-weight", type=float, default=0.5)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--reranker-revision", default=DEFAULT_RERANKER_REVISION)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--reranker-batch-size", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--reranker-max-length", type=int, default=1024)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.rerank_k < args.top_k:
        raise ValueError("rerank-k doit être supérieur ou égal à top-k.")
    if args.candidate_k < args.rerank_k:
        raise ValueError("candidate-k doit être supérieur ou égal à rerank-k.")

    chunks = [
        chunk
        for path in args.chunks
        for chunk in load_active_chunks(path, args.schemas_dir / "retrieval_chunk.schema.json")
    ]
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Les chunk_id doivent être uniques dans l'index.")
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    cases = load_evaluation_files(args.evaluation, schemas_dir=args.schemas_dir, chunks=chunks)
    validate_expected_sections(cases, chunks)

    started = perf_counter()
    embeddings = Qwen3LangChainEmbeddings(
        model_name=args.model_name,
        revision=args.model_revision,
        device=args.device,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
    )
    embedding_load_seconds = perf_counter() - started
    started = perf_counter()
    index = HybridRetrievalIndex(
        chunks,
        embeddings,
        bm25_weight=args.bm25_weight,
        semantic_weight=args.semantic_weight,
        rrf_k=args.rrf_k,
        candidate_k=args.candidate_k,
    )
    index_build_seconds = perf_counter() - started
    baseline_records, candidates_by_case, retrieval_latencies = collect_hybrid_candidates(
        index, cases, top_k=args.top_k, rerank_k=args.rerank_k
    )

    # Le reranker ne partage pas le GPU avec le modèle d'embeddings : le pic de
    # mémoire reste compatible avec une carte de 4 Go.
    del index
    del embeddings
    release_gpu_memory()

    started = perf_counter()
    reranker = Qwen3CrossEncoderReranker(
        model_name=args.reranker_model,
        revision=args.reranker_revision,
        device=args.device,
        batch_size=args.reranker_batch_size,
        max_length=args.reranker_max_length,
    )
    reranker_load_seconds = perf_counter() - started
    pairs: list[tuple[str, str]] = []
    slices: list[tuple[int, int]] = []
    for case, candidates in zip(cases, candidates_by_case):
        start = len(pairs)
        pairs.extend(
            (case["question"], searchable_text(chunks_by_id[item["chunk_id"]]))
            for item in candidates
        )
        slices.append((start, len(pairs)))

    started = perf_counter()
    all_scores = reranker.score(pairs)
    reranker_scoring_ms = (perf_counter() - started) * 1000.0
    amortized_scoring_ms = reranker_scoring_ms / len(cases) if cases else 0.0
    reranker_records: list[dict[str, Any]] = []
    for case, candidates, retrieval_ms, (start, end) in zip(
        cases, candidates_by_case, retrieval_latencies, slices
    ):
        ranking = rerank_with_scores(candidates, all_scores[start:end], top_k=args.top_k)
        reranker_records.append(
            _record(
                case,
                ranking,
                system="hybrid_reranker",
                latency_ms=retrieval_ms + amortized_scoring_ms,
            )
        )

    all_records = {**baseline_records, "hybrid_reranker": reranker_records}
    test_indices = [index for index, case in enumerate(cases) if case["split"] == "test"]
    test_cases = [cases[index] for index in test_indices]
    test_candidates = [candidates_by_case[index] for index in test_indices]
    report = {
        "schema_version": "0.1.0",
        "configuration": {
            "pipeline": "bm25_plus_qwen3_embedding_rrf_then_qwen3_cross_encoder",
            "bm25_normalization": "darija_lexicon",
            "embedding_model": args.model_name,
            "embedding_revision": args.model_revision,
            "reranker_model": args.reranker_model,
            "reranker_revision": args.reranker_revision,
            "device": args.device,
            "bm25_weight": args.bm25_weight,
            "semantic_weight": args.semantic_weight,
            "rrf_k": args.rrf_k,
            "candidate_k": args.candidate_k,
            "rerank_k": args.rerank_k,
            "top_k": args.top_k,
            "embedding_batch_size": args.batch_size,
            "reranker_batch_size": args.reranker_batch_size,
            "embedding_max_seq_length": args.max_seq_length,
            "reranker_max_length": args.reranker_max_length,
            "indexed_chunk_count": len(chunks),
            "evaluation_case_count": len(cases),
            "scored_pair_count": len(pairs),
            "embedding_load_seconds": embedding_load_seconds,
            "index_build_seconds": index_build_seconds,
            "reranker_load_seconds": reranker_load_seconds,
            "reranker_scoring_seconds": reranker_scoring_ms / 1000.0,
            "reranker_latency_method": "hybrid retrieval + amortized batched reranker scoring",
        },
        "candidate_set_test": candidate_set_metrics(test_cases, test_candidates),
        "systems": {name: _system_report(records) for name, records in all_records.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Rapport : {args.output}")
    print(f"Chunks indexés : {len(chunks)} ; cas évalués : {len(cases)} ; paires : {len(pairs)}")
    for name in ("bm25", "semantic", "hybrid", "hybrid_reranker"):
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
