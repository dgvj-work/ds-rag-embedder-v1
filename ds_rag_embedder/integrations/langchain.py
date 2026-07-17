"""LangChain-compatible embeddings wrapper."""

from __future__ import annotations

from typing import List

from ds_rag_embedder.model import DSRAGEmbedder


class DSRAGLangChainEmbeddings:
    """
    Drop-in style wrapper for LangChain vector stores.

    Usage:
        from ds_rag_embedder.integrations.langchain import DSRAGLangChainEmbeddings
        embeddings = DSRAGLangChainEmbeddings()
        vec = embeddings.embed_query("How to detect data leakage?")
    """

    def __init__(self, model_name_or_path: str | None = None) -> None:
        self._embedder = DSRAGEmbedder(model_name_or_path=model_name_or_path)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        arr = self._embedder.encode_documents(texts)
        return arr.tolist()

    def embed_query(self, text: str) -> List[float]:
        arr = self._embedder.encode_queries([text])
        return arr[0].tolist()
