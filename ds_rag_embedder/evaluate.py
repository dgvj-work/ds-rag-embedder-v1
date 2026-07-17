"""Retrieval evaluation metrics for DS RAG embedder benchmarks."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dcg(relevances: list[float]) -> float:
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))


def _ndcg_at_k(relevances: list[float], k: int) -> float:
    ideal = sorted(relevances, reverse=True)[:k]
    dcg = _dcg(relevances[:k])
    idcg = _dcg(ideal)
    return float(dcg / idcg) if idcg > 0 else 0.0


def evaluate_retrieval(
    embedder: DSRAGEmbedder,
    benchmark_path: Path | None = None,
    corpus_path: Path | None = None,
) -> RetrievalMetrics:
    """Run Recall@k, MRR, and nDCG@10 on the DS RAG benchmark."""
    benchmark_path = benchmark_path or BENCHMARK_PATH
    corpus_path = corpus_path or CORPUS_PATH

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
    with benchmark_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            queries.append(row["query"])
            rel_idxs = {id_to_idx[rid] for rid in row["relevant_ids"] if rid in id_to_idx}
            qrels.append(rel_idxs)

    doc_emb = embedder.encode_documents(corpus_texts, show_progress_bar=False)
    q_emb = embedder.encode_queries(queries, show_progress_bar=False)
    scores = embedder.similarity(q_emb, doc_emb)

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
        # MRR
        rr = 0.0
        for rank, idx in enumerate(ranked, start=1):
            if idx in rel:
                rr = 1.0 / rank
                break
        mrrs.append(rr)
        # nDCG@10
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
    )


def compare_models(
    model_paths: dict[str, str | Path],
    benchmark_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Compare multiple embedders on the same benchmark."""
    results = {}
    for name, path in model_paths.items():
        embedder = DSRAGEmbedder(model_name_or_path=path)
        metrics = evaluate_retrieval(embedder, benchmark_path=benchmark_path)
        results[name] = metrics.to_dict()
    return results
