"""Vector retriever built on DS RAG Embedder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ds_rag_embedder.model import DSRAGEmbedder
from ds_rag_embedder.rag.chunker import chunk_text


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    query: str
    hits: list[dict[str, Any]]


class DSRAGRetriever:
    """In-memory retriever for DS/ML docs. Swap index backend for production scale."""

    def __init__(self, embedder: DSRAGEmbedder | None = None) -> None:
        self.embedder = embedder or DSRAGEmbedder()
        self.documents: list[Document] = []
        self._embeddings: np.ndarray | None = None

    def add_text(
        self,
        doc_id: str,
        text: str,
        metadata: dict | None = None,
        chunk: bool = True,
        chunk_size: int = 400,
    ) -> int:
        """Add document(s) to the index. Returns number of chunks indexed."""
        metadata = metadata or {}
        if chunk:
            chunks = chunk_text(text, chunk_size=chunk_size, metadata=metadata)
            for i, c in enumerate(chunks):
                self.documents.append(
                    Document(
                        doc_id=f"{doc_id}::{i}",
                        text=c.text,
                        metadata={**c.metadata, "parent_id": doc_id},
                    )
                )
        else:
            self.documents.append(Document(doc_id=doc_id, text=text, metadata=metadata))
        self._embeddings = None
        return len(self.documents)

    def build_index(self, show_progress: bool = False) -> None:
        texts = [d.text for d in self.documents]
        self._embeddings = self.embedder.encode_documents(texts, show_progress_bar=show_progress)

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        if not self.documents:
            return RetrievalResult(query=query, hits=[])
        if self._embeddings is None:
            self.build_index()
        q_emb = self.embedder.encode_queries([query])
        scores = self.embedder.similarity(q_emb, self._embeddings)[0]
        ranked = np.argsort(scores)[::-1][:top_k]
        hits = []
        for rank, idx in enumerate(ranked, start=1):
            doc = self.documents[int(idx)]
            hits.append(
                {
                    "rank": rank,
                    "score": float(scores[idx]),
                    "doc_id": doc.doc_id,
                    "text": doc.text,
                    "metadata": doc.metadata,
                }
            )
        return RetrievalResult(query=query, hits=hits)

    def load_corpus_jsonl(self, path: str) -> int:
        """Load corpus from project jsonl format."""
        import json
        from pathlib import Path

        count = 0
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                self.add_text(
                    doc_id=row["id"],
                    text=row["text"],
                    metadata={
                        "category": row.get("category"),
                        "title": row.get("title"),
                        "source": row.get("source", "corpus"),
                    },
                    chunk=False,
                )
                count += 1
        return count
