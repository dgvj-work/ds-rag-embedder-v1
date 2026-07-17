"""Retrieval evaluation metrics for DS RAG embedder benchmarks."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import numpy as np

from ds_rag_embedder.config import BENCHMARK_PATH, CORPUS_PATH
from ds_rag_embedder.model import DSRAGEmbedder


@dataclass
class RetrievalMetrics:
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    num_queries: int
    latency_ms_per_query: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkReport:
    overall: RetrievalMetrics
    by_category: dict[str, RetrievalMetrics] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall.to_dict(),
            "by_category": {k: v.to_dict() for k, v in self.by_category.items()},
        }


def _dcg(relevances: list[float]) -> float:
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))


def _ndcg_at_k(relevances: list[float], k: int) -> float:
    ideal = sorted(relevances, reverse=True)[:k]
    dcg = _dcg(relevances[:k])
    idcg = _dcg(ideal)
    return float(dcg / idcg) if idcg > 0 else 0.0


def _metrics_from_rankings(
    qrels: list[set[int]],
    scores: np.ndarray,
    measure_latency: bool = False,
    latency_ms: float | None = None,
) -> RetrievalMetrics:
    recalls = {1: [], 3: [], 5: [], 10: []}
    mrrs: list[float] = []
    ndcgs: list[float] = []

    for i, rel in enumerate(qrels):
        if not rel:
            continue
        ranked = np.argsort(scores[i])[::-1]
        for k in recalls:
            top_k = set(ranked[:k].tolist())
            recalls[k].append(1.0 if top_k & rel else 0.0)
        rr = 0.0
        for rank, idx in enumerate(ranked, start=1):
            if idx in rel:
                rr = 1.0 / rank
                break
        mrrs.append(rr)
        rel_scores = [1.0 if idx in rel else 0.0 for idx in ranked[:10]]
        ndcgs.append(_ndcg_at_k(rel_scores, 10))

    n = len(mrrs)
    return RetrievalMetrics(
        recall_at_1=float(np.mean(recalls[1])) if n else 0.0,
        recall_at_3=float(np.mean(recalls[3])) if n else 0.0,
        recall_at_5=float(np.mean(recalls[5])) if n else 0.0,
        recall_at_10=float(np.mean(recalls[10])) if n else 0.0,
        mrr=float(np.mean(mrrs)) if n else 0.0,
        ndcg_at_10=float(np.mean(ndcgs)) if n else 0.0,
        num_queries=n,
        latency_ms_per_query=latency_ms,
    )


def _load_benchmark_data(
    benchmark_path: Path,
    corpus_path: Path,
) -> tuple[list[str], list[set[int]], list[str | None], list[str], dict[str, int]]:
    corpus_rows = []
    with corpus_path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                corpus_rows.append(json.loads(line))
    corpus_ids = [r["id"] for r in corpus_rows]
    corpus_texts = [r["text"] for r in corpus_rows]
    id_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}

    queries: list[str] = []
    qrels: list[set[int]] = []
    categories: list[str | None] = []
    with benchmark_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            queries.append(row["query"])
            categories.append(row.get("category"))
            rel_idxs = {id_to_idx[rid] for rid in row["relevant_ids"] if rid in id_to_idx}
            qrels.append(rel_idxs)
    return queries, qrels, categories, corpus_texts, id_to_idx


def evaluate_retrieval(
    embedder: DSRAGEmbedder,
    benchmark_path: Path | None = None,
    corpus_path: Path | None = None,
    measure_latency: bool = False,
) -> RetrievalMetrics:
    """Run Recall@k, MRR, and nDCG@10 on the DS RAG benchmark."""
    benchmark_path = benchmark_path or BENCHMARK_PATH
    corpus_path = corpus_path or CORPUS_PATH

    queries, qrels, _, corpus_texts, _ = _load_benchmark_data(benchmark_path, corpus_path)

    t0 = time.perf_counter()
    doc_emb = embedder.encode_documents(corpus_texts, show_progress_bar=False)
    q_emb = embedder.encode_queries(queries, show_progress_bar=False)
    scores = embedder.similarity(q_emb, doc_emb)
    latency = ((time.perf_counter() - t0) / max(len(queries), 1)) * 1000 if measure_latency else None

    return _metrics_from_rankings(qrels, scores, latency_ms=latency)


def evaluate_by_category(
    embedder: DSRAGEmbedder,
    benchmark_path: Path | None = None,
    corpus_path: Path | None = None,
    measure_latency: bool = True,
) -> BenchmarkReport:
    """Overall + per-category retrieval metrics."""
    benchmark_path = benchmark_path or BENCHMARK_PATH
    corpus_path = corpus_path or CORPUS_PATH

    queries, qrels, categories, corpus_texts, _ = _load_benchmark_data(benchmark_path, corpus_path)

    t0 = time.perf_counter()
    doc_emb = embedder.encode_documents(corpus_texts, show_progress_bar=False)
    q_emb = embedder.encode_queries(queries, show_progress_bar=False)
    scores = embedder.similarity(q_emb, doc_emb)
    overall_latency = ((time.perf_counter() - t0) / max(len(queries), 1)) * 1000

    overall = _metrics_from_rankings(qrels, scores, latency_ms=overall_latency if measure_latency else None)

    by_cat: dict[str, RetrievalMetrics] = {}
    unique_cats = sorted({c for c in categories if c})
    for cat in unique_cats:
        idxs = [i for i, c in enumerate(categories) if c == cat]
        sub_qrels = [qrels[i] for i in idxs]
        sub_scores = scores[idxs]
        by_cat[cat] = _metrics_from_rankings(sub_qrels, sub_scores)

    return BenchmarkReport(overall=overall, by_category=by_cat)


def compare_models(
    model_paths: dict[str, str | Path],
    benchmark_path: Path | None = None,
    include_categories: bool = False,
) -> dict[str, dict[str, Any]]:
    """Compare multiple embedders on the same benchmark."""
    results: dict[str, dict[str, Any]] = {}
    for name, path in model_paths.items():
        embedder = DSRAGEmbedder(model_name_or_path=path)
        if include_categories:
            report = evaluate_by_category(embedder, benchmark_path=benchmark_path)
            results[name] = report.to_dict()
        else:
            metrics = evaluate_retrieval(embedder, benchmark_path=benchmark_path, measure_latency=True)
            results[name] = metrics.to_dict()
    return results
