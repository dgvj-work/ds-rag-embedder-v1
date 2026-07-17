"""Hybrid BM25 + dense retrieval for production RAG stacks."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ds_rag_embedder.model import DSRAGEmbedder
from ds_rag_embedder.rag.retriever import Document, RetrievalResult


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class BM25Index:
    """Lightweight Okapi BM25 index (no extra dependencies)."""

    documents: list[str]
    doc_tokens: list[list[str]] = field(default_factory=list)
    df: Counter = field(default_factory=Counter)
    avgdl: float = 0.0
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        if not self.doc_tokens:
            self.doc_tokens = [_tokenize(d) for d in self.documents]
            self.df = Counter()
            for tokens in self.doc_tokens:
                self.df.update(set(tokens))
            self.avgdl = sum(len(t) for t in self.doc_tokens) / max(len(self.doc_tokens), 1)

    @classmethod
    def from_documents(cls, documents: list[str]) -> "BM25Index":
        return cls(documents=documents)

    def _idf(self, term: str) -> float:
        n = len(self.documents)
        df = self.df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def score_query(self, query: str) -> np.ndarray:
        q_terms = _tokenize(query)
        scores = np.zeros(len(self.documents), dtype=np.float64)
        for i, tokens in enumerate(self.doc_tokens):
            tf = Counter(tokens)
            dl = len(tokens)
            for term in q_terms:
                if term not in tf:
                    continue
                freq = tf[term]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += self._idf(term) * (freq * (self.k1 + 1)) / denom
        return scores


@dataclass
class HybridRetriever:
    """
    Combine lexical BM25 with dense embeddings.

    BM25 catches exact DS tokens (SMOTE, PSI, AUC); dense models capture paraphrases.
    """

    embedder: DSRAGEmbedder
    documents: list[str]
    metadata: list[dict] | None = None
    alpha: float = 0.65
    _doc_embeddings: np.ndarray | None = None
    _bm25: BM25Index | None = None
    _doc_objects: list[Document] = field(default_factory=list)

    def build_index(self) -> None:
        self._doc_objects = [
            Document(
                doc_id=str(i),
                text=text,
                metadata=(self.metadata[i] if self.metadata and i < len(self.metadata) else {}),
            )
            for i, text in enumerate(self.documents)
        ]
        texts = [d.text for d in self._doc_objects]
        self._doc_embeddings = self.embedder.encode_documents(texts, show_progress_bar=False)
        self._bm25 = BM25Index.from_documents(texts)

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        lo, hi = float(scores.min()), float(scores.max())
        if hi - lo < 1e-9:
            return np.zeros_like(scores)
        return (scores - lo) / (hi - lo)

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        if self._doc_embeddings is None or self._bm25 is None:
            self.build_index()
        assert self._doc_embeddings is not None and self._bm25 is not None

        q_emb = self.embedder.encode_queries([query])
        dense_scores = self.embedder.similarity(q_emb, self._doc_embeddings)[0]
        bm25_scores = self._bm25.score_query(query)
        combined = self.alpha * self._normalize(dense_scores) + (1 - self.alpha) * self._normalize(
            bm25_scores
        )
        ranked = np.argsort(combined)[::-1][:top_k]
        hits: list[dict[str, Any]] = []
        for rank, idx in enumerate(ranked, start=1):
            doc = self._doc_objects[int(idx)]
            hits.append(
                {
                    "rank": rank,
                    "score": float(combined[idx]),
                    "dense_score": float(dense_scores[idx]),
                    "bm25_score": float(bm25_scores[idx]),
                    "doc_id": doc.doc_id,
                    "text": doc.text,
                    "metadata": doc.metadata,
                    "index": int(idx),
                }
            )
        return RetrievalResult(query=query, hits=hits)
