"""Core embedding model wrapper with HF Hub integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ds_rag_embedder.config import DEFAULT_CONFIG, EmbedderConfig


class DSRAGEmbedder:
    """
    Domain-specific embedder for Data Science & ML documentation retrieval.

    Optimized for RAG pipelines over notebooks, model cards, experiment logs,
    metrics guides, and ML engineering docs.
    """

    def __init__(
        self,
        model_name_or_path: str | Path | None = None,
        config: EmbedderConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.model_name_or_path = str(
            model_name_or_path or self.config.hub_model_id or self.config.output_dir
        )
        self._model = None
        self._device = device

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name_or_path,
                device=self._device,
            )
            self._model.max_seq_length = self.config.max_seq_length
        return self._model

    def encode_queries(
        self,
        queries: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        prefixed = [f"{self.config.query_prefix}{q}" for q in queries]
        return self.model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
        )

    def encode_documents(
        self,
        documents: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        prefixed = [f"{self.config.doc_prefix}{d}" for d in documents]
        return self.model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
        )

    def encode(
        self,
        texts: list[str],
        *,
        is_query: bool = False,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        if is_query:
            return self.encode_queries(texts, batch_size, show_progress_bar)
        return self.encode_documents(texts, batch_size, show_progress_bar)

    @staticmethod
    def similarity(query_embeddings: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """Cosine similarity matrix (queries x documents). Assumes normalized vectors."""
        return np.matmul(query_embeddings, doc_embeddings.T)

    def search(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return top-k documents ranked by cosine similarity."""
        q_emb = self.encode_queries([query])
        d_emb = self.encode_documents(documents)
        scores = self.similarity(q_emb, d_emb)[0]
        ranked = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "rank": i + 1,
                "score": float(scores[idx]),
                "document": documents[idx],
                "index": int(idx),
            }
            for i, idx in enumerate(ranked)
        ]

    def save_local(self, output_dir: str | Path | None = None) -> Path:
        out = Path(output_dir or self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save(str(out))
        meta = {
            "model_id": self.config.hub_model_id,
            "base_model": self.config.base_model,
            "query_prefix": self.config.query_prefix,
            "doc_prefix": self.config.doc_prefix,
            "max_seq_length": self.config.max_seq_length,
            "embedding_dim": self.config.embedding_dim,
            "normalize_embeddings": self.config.normalize_embeddings,
            "domain": "data-science-ml-rag",
        }
        (out / "ds_rag_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return out

    def push_to_hub(
        self,
        repo_id: str | None = None,
        commit_message: str = "Upload DS RAG Embedder v1",
        private: bool = False,
    ) -> str:
        repo = repo_id or self.config.hub_model_id
        self.model.push_to_hub(repo, commit_message=commit_message, private=private)
        return f"https://huggingface.co/{repo}"

    @classmethod
    def from_local(cls, path: str | Path, config: EmbedderConfig | None = None) -> "DSRAGEmbedder":
        return cls(model_name_or_path=path, config=config)
