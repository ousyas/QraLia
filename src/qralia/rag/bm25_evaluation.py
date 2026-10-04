"""Baseline BM25 et évaluation déterministe de la récupération QraLia."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Sequence
import unicodedata

from qralia.rag.onee_corpus import validate_with_schema


TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)

# Petit lexique volontairement explicite. Il ne cherche pas à "traduire la
# darija" : il rapproche seulement des graphies courantes observées dans les
# questions utilisateur des mots présents dans le corpus français.
DARIJA_LATIN_ALIASES = {
    "factora": "facture",
    "fatoura": "facture",
    "factoura": "facture",
    "fatora": "facture",
    "factura": "facture",
    "fawatir": "facture",
    "chik": "cheque",
    "chek": "cheque",
    "cash": "especes",
    "kach": "especes",
    "flous": "especes",
    "virman": "virement",
    "virmon": "virement",
    "tasbi9": "acompte",
    "tsbi9": "acompte",
    "avance": "acompte",
    "3tla": "ferie",
    "3otla": "ferie",
    "l3id": "ferie",
    "echeance": "date",
    "ajal": "date",
    "fo9ach": "date",
    "banka": "banque",
    "wakala": "agence",
    "nkhlles": "paiement",
    "khllest": "paiement",
    "tkhlles": "paiement",
    "khlass": "paiement",
    "chki": "reclamation",
    "chikaya": "reclamation",
    "t9t3": "suspension",
    "y9t3o": "suspension",
    "richarj": "recharge",
    "الفاتورة": "facture",
    "فاتورة": "facture",
    "الشيك": "cheque",
    "بالشيك": "cheque",
    "الكاش": "especes",
    "بالكاش": "especes",
    "فيرمان": "virement",
    "عطلة": "ferie",
    "الكارط": "carte",
    "بالكارط": "carte",
    "نخلص": "paiement",
    "خلصت": "paiement",
    "تخلص": "paiement",
    "الأداء": "paiement",
    "اداء": "paiement",
    "الوكالة": "agence",
    "فالوكالة": "agence",
    "الشكوى": "reclamation",
    "نشكي": "reclamation",
    "الاقتطاع": "prelevement",
    "التأخير": "retard",
    "التعبئة": "recharge",
}

NORMALIZATION_MODES = ("basic", "darija_lexicon")


def _strip_accents(token: str) -> str:
    """Retire les signes diacritiques afin d'aligner ``chèque`` et ``cheque``."""

    decomposed = unicodedata.normalize("NFKD", token)
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def tokenize(text: str, *, normalization: str = "basic") -> list[str]:
    """Découpe le texte et applique la normalisation demandée.

    ``basic`` conserve exactement la baseline historique. ``darija_lexicon``
    retire les accents des tokens latins puis applique un petit dictionnaire
    d'alias déterministe. Le même traitement est utilisé pour les chunks et
    pour les questions.
    """

    if normalization not in NORMALIZATION_MODES:
        raise ValueError(f"Normalisation inconnue : {normalization}")

    tokens = [match.group(0).casefold() for match in TOKEN_PATTERN.finditer(text)]
    if normalization == "basic":
        return tokens

    normalized: list[str] = []
    for token in tokens:
        accentless = _strip_accents(token)
        canonical = DARIJA_LATIN_ALIASES.get(accentless)
        # En darija latine, l'article défini est souvent collé au nom :
        # ``lfactura`` ou ``lflous``. On ne retire le ``l`` que si le reste du
        # token appartient déjà au lexique, ce qui évite un stemming agressif.
        if canonical is None and accentless.startswith("l"):
            canonical = DARIJA_LATIN_ALIASES.get(accentless[1:])
        normalized.append(canonical or accentless)
    return normalized


@dataclass(frozen=True)
class IndexedChunk:
    """Chunk actif accompagné des tokens utilisés par BM25."""

    payload: dict[str, Any]
    tokens: tuple[str, ...]

    @property
    def section_id(self) -> str:
        return str(self.payload["section_id"])

    @property
    def audiences(self) -> tuple[str, ...]:
        return tuple(self.payload["metadata"]["audiences"])

    @property
    def provider_id(self) -> str:
        metadata = self.payload["metadata"]
        # Les jeux d'évaluation 0.2 parlent encore de distributeur. Les chunks
        # 0.3 utilisent le terme générique fournisseur, valable aussi en télécom.
        return str(metadata.get("provider_id", metadata.get("distributor_id")))

    @property
    def service_zone_id(self) -> str:
        return str(self.payload["metadata"]["service_zone_id"])

    @property
    def domain(self) -> str | None:
        return self.payload["metadata"].get("domain")

    @property
    def service_type(self) -> str | None:
        return self.payload["metadata"].get("service_type")


class BM25Index:
    """Implémentation minimale et explicable de BM25 Okapi."""

    def __init__(
        self,
        chunks: Sequence[dict[str, Any]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        normalization: str = "basic",
    ) -> None:
        if not chunks:
            raise ValueError("BM25 exige au moins un chunk actif.")
        if k1 <= 0:
            raise ValueError("k1 doit être strictement positif.")
        if not 0 <= b <= 1:
            raise ValueError("b doit être compris entre 0 et 1.")
        if normalization not in NORMALIZATION_MODES:
            raise ValueError(f"Normalisation inconnue : {normalization}")

        self.k1 = k1
        self.b = b
        self.normalization = normalization
        self.chunks = tuple(
            IndexedChunk(
                payload=chunk,
                tokens=tuple(tokenize(_searchable_text(chunk), normalization=normalization)),
            )
            for chunk in chunks
        )
        self.average_document_length = sum(len(chunk.tokens) for chunk in self.chunks) / len(self.chunks)
        self.document_frequencies = self._document_frequencies()

    def _document_frequencies(self) -> Counter[str]:
        frequencies: Counter[str] = Counter()
        for chunk in self.chunks:
            frequencies.update(set(chunk.tokens))
        return frequencies

    def inverse_document_frequency(self, term: str) -> float:
        """Calcule l'IDF BM25 positive de Robertson-Sparck Jones."""

        document_count = len(self.chunks)
        document_frequency = self.document_frequencies.get(term, 0)
        return math.log(
            1.0 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
        )

    def score(self, query: str, chunk: IndexedChunk) -> float:
        """Calcule le score BM25 d'une requête pour un chunk."""

        term_frequencies = Counter(chunk.tokens)
        document_length = len(chunk.tokens)
        score = 0.0

        # dict.fromkeys retire les répétitions de la requête tout en gardant l'ordre.
        for term in dict.fromkeys(tokenize(query, normalization=self.normalization)):
            frequency = term_frequencies.get(term, 0)
            if frequency == 0:
                continue
            denominator = frequency + self.k1 * (
                1.0 - self.b + self.b * document_length / self.average_document_length
            )
            score += self.inverse_document_frequency(term) * (
                frequency * (self.k1 + 1.0) / denominator
            )

        return score

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
        """Classe les chunks compatibles avec l'audience et le distributeur."""

        if top_k < 1:
            raise ValueError("top_k doit être au moins égal à 1.")

        candidates = [
            chunk
            for chunk in self.chunks
            if audience in chunk.audiences
            and chunk.provider_id == distributor_id
            and chunk.service_zone_id == service_zone_id
            and (domain is None or chunk.domain == domain)
            and (service_type is None or chunk.service_type == service_type)
        ]
        scored = [
            {
                "chunk_id": chunk.payload["chunk_id"],
                "section_id": chunk.section_id,
                "score": self.score(query, chunk),
                "title": chunk.payload["metadata"]["title"],
                "canonical_url": chunk.payload["metadata"]["canonical_url"],
                "provider_id": chunk.payload["metadata"].get("provider_id", chunk.payload["metadata"].get("distributor_id")),
                "provider_name": chunk.payload["metadata"].get("provider_name", chunk.payload["metadata"].get("distributor_name")),
                "service_zone_id": chunk.payload["metadata"]["service_zone_id"],
            }
            for chunk in candidates
        ]
        # Un score nul signifie qu'aucun token de la question n'apparaît dans
        # le chunk. Le conserver créerait un classement arbitraire et gonflerait
        # artificiellement Recall@k et MRR, surtout pour les requêtes multilingues.
        scored = [result for result in scored if result["score"] > 0.0]
        scored.sort(key=lambda result: (-result["score"], result["chunk_id"]))
        return scored[:top_k]


def _searchable_text(chunk: dict[str, Any]) -> str:
    """Texte indexé : titres hiérarchiques puis contenu officiel."""

    headings = chunk["metadata"].get("heading_path", [])
    return " ".join([*headings, chunk["text"]])


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Charge un JSONL en signalant précisément une ligne invalide."""

    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            values.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON invalide dans {path}, ligne {line_number}: {exc}") from exc
    return values


def load_active_chunks(path: Path, schema_path: Path) -> list[dict[str, Any]]:
    chunks = load_jsonl(path)
    for chunk in chunks:
        validate_with_schema(chunk, schema_path)
    active = [chunk for chunk in chunks if chunk["metadata"]["validity_status"] == "active"]
    if not active:
        raise ValueError("Aucun chunk actif n'est disponible.")
    return active


def load_evaluation_cases(path: Path, schema_path: Path) -> list[dict[str, Any]]:
    cases = load_jsonl(path)
    for case in cases:
        validate_with_schema(case, schema_path)
    case_ids = [case["case_id"] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Les case_id du jeu d'évaluation doivent être uniques.")
    return cases


def evaluate_case(
    index: BM25Index,
    case: dict[str, Any],
    *,
    top_k: int,
) -> dict[str, Any]:
    ranking = index.search(
        case["question"],
        audience=case["audience"],
        distributor_id=case["distributor_id"],
        service_zone_id=case["service_zone_id"],
        top_k=top_k,
    )
    relevant = set(case["relevant_section_ids"])
    retrieved = [result["section_id"] for result in ranking]
    relevant_ranks = [rank for rank, section_id in enumerate(retrieved, start=1) if section_id in relevant]

    recalls = {
        str(k): (
            len(relevant.intersection(retrieved[:k])) / len(relevant)
            if relevant
            else None
        )
        for k in (1, 3, 5)
    }
    binary_recalls = {
        str(k): (
            float(bool(relevant.intersection(retrieved[:k])))
            if relevant
            else None
        )
        for k in (1, 3, 5)
    }

    return {
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
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def ranking_metrics(records: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    """Agrège les métriques seulement sur les questions ayant une réponse."""

    positives = [record for record in records if record["relevant_section_ids"]]
    return {
        "positive_case_count": len(positives),
        "recall_at_1": _mean(record["recall_at_k"]["1"] for record in positives),
        "recall_at_3": _mean(record["recall_at_k"]["3"] for record in positives),
        "recall_at_5": _mean(record["recall_at_k"]["5"] for record in positives),
        "binary_recall_at_1": _mean(record["binary_recall_at_k"]["1"] for record in positives),
        "binary_recall_at_3": _mean(record["binary_recall_at_k"]["3"] for record in positives),
        "binary_recall_at_5": _mean(record["binary_recall_at_k"]["5"] for record in positives),
        "mrr": _mean(record["reciprocal_rank"] for record in positives),
    }


def _threshold_candidates(records: Sequence[dict[str, Any]]) -> list[float]:
    scores = sorted({float(record["top_score"]) for record in records})
    if not scores:
        return [0.0]
    candidates = {0.0, scores[-1] + 1e-9}
    candidates.update((left + right) / 2.0 for left, right in zip(scores, scores[1:]))
    return sorted(candidates)


def _predict_has_answer(record: dict[str, Any], threshold: float) -> bool:
    return bool(record["has_candidates"] and record["top_score"] >= threshold)


def abstention_metrics(records: Sequence[dict[str, Any]], threshold: float) -> dict[str, float | int]:
    """Évalue la décision binaire répondre/s'abstenir."""

    true_positive = false_positive = true_negative = false_negative = 0
    for record in records:
        expected_has_answer = not record["should_abstain"]
        predicted_has_answer = _predict_has_answer(record, threshold)
        if expected_has_answer and predicted_has_answer:
            true_positive += 1
        elif not expected_has_answer and predicted_has_answer:
            false_positive += 1
        elif not expected_has_answer and not predicted_has_answer:
            true_negative += 1
        else:
            false_negative += 1

    positive_count = true_positive + false_negative
    negative_count = true_negative + false_positive
    answer_recall = true_positive / positive_count if positive_count else 0.0
    abstention_recall = true_negative / negative_count if negative_count else 0.0
    accuracy = (true_positive + true_negative) / len(records) if records else 0.0
    balanced_accuracy = (answer_recall + abstention_recall) / 2.0
    return {
        "case_count": len(records),
        "threshold": threshold,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "answer_recall": answer_recall,
        "abstention_recall": abstention_recall,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
    }


def calibrate_threshold(records: Sequence[dict[str, Any]]) -> tuple[float, dict[str, float | int]]:
    """Choisit sur calibration le seuil maximisant la balanced accuracy.

    En cas d'égalité, le seuil le plus élevé est retenu : QraLia préfère alors
    l'abstention prudente à une réponse faiblement étayée.
    """

    if not records:
        raise ValueError("Aucun cas de calibration n'est disponible.")
    scored = [
        (abstention_metrics(records, threshold), threshold)
        for threshold in _threshold_candidates(records)
    ]
    metrics, threshold = max(
        scored,
        key=lambda item: (
            item[0]["balanced_accuracy"],
            item[0]["accuracy"],
            item[1],
        ),
    )
    return threshold, metrics


def grouped_ranking_metrics(records: Sequence[dict[str, Any]], field: str) -> dict[str, dict[str, float | int]]:
    groups = sorted({str(record[field]) for record in records})
    return {
        group: ranking_metrics([record for record in records if record[field] == group])
        for group in groups
    }


def build_report(
    index: BM25Index,
    cases: Sequence[dict[str, Any]],
    *,
    top_k: int,
) -> dict[str, Any]:
    records = [
        evaluate_case(index, case, top_k=top_k)
        for case in cases
    ]
    calibration = [record for record in records if record["split"] == "calibration"]
    test = [record for record in records if record["split"] == "test"]
    threshold, calibration_abstention = calibrate_threshold(calibration)

    for record in records:
        record["predicted_abstain"] = not _predict_has_answer(record, threshold)

    return {
        "schema_version": "0.1.0",
        "configuration": {
            "algorithm": "BM25Okapi",
            "k1": index.k1,
            "b": index.b,
            "normalization": index.normalization,
            "routes": sorted(
                {
                    f"{case['distributor_id']}::{case['service_zone_id']}"
                    for case in cases
                }
            ),
            "top_k": top_k,
            "indexed_chunk_count": len(index.chunks),
            "indexed_fields": ["metadata.heading_path", "text"],
            "tokenization": (
                "Unicode alphanumeric tokens with casefold"
                if index.normalization == "basic"
                else "Unicode tokens, casefold, accent folding and curated Darija-Latin aliases"
            ),
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
        },
        "cases": records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Évalue une baseline BM25 sur les chunks actifs QraLia.")
    parser.add_argument(
        "--chunks",
        required=True,
        action="append",
        type=Path,
        help="Fichier JSONL de chunks ; l'option peut être répétée.",
    )
    parser.add_argument("--evaluation", required=True, type=Path, help="Jeu JSONL d'évaluation de la récupération.")
    parser.add_argument("--output", required=True, type=Path, help="Rapport JSON à produire.")
    parser.add_argument("--schemas-dir", type=Path, default=Path("schemas"), help="Dossier des schémas JSON.")
    parser.add_argument("--top-k", type=int, default=5, help="Nombre maximal de résultats conservés.")
    parser.add_argument(
        "--distributor-id",
        help="Optionnel : ne conserver que les cas de ce distributeur.",
    )
    parser.add_argument(
        "--service-zone-id",
        help="Optionnel : ne conserver que les cas de cette zone.",
    )
    parser.add_argument("--k1", type=float, default=1.5, help="Saturation de fréquence BM25.")
    parser.add_argument("--b", type=float, default=0.75, help="Normalisation par longueur BM25.")
    parser.add_argument(
        "--normalization",
        choices=NORMALIZATION_MODES,
        default="basic",
        help="Normalisation lexicale appliquée à l'index et aux questions.",
    )
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
        raise ValueError("Les chunk_id doivent être uniques dans l'index BM25.")
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
        raise ValueError(f"Sections attendues absentes des chunks actifs : {unknown}")

    index = BM25Index(
        chunks,
        k1=args.k1,
        b=args.b,
        normalization=args.normalization,
    )
    report = build_report(
        index,
        cases,
        top_k=args.top_k,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    test_metrics = report["test"]["ranking_metrics"]
    abstention = report["test"]["abstention_metrics"]
    print(f"Rapport : {args.output}")
    print(f"Recall@5 : {test_metrics['recall_at_5']:.3f}")
    print(f"Binary Recall@5 : {test_metrics['binary_recall_at_5']:.3f}")
    print(f"MRR : {test_metrics['mrr']:.3f}")
    print(f"Seuil d'abstention : {report['calibration']['selected_threshold']:.6f}")
    print(f"Balanced accuracy d'abstention : {abstention['balanced_accuracy']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
