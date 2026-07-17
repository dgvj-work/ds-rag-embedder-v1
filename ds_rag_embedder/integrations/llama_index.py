"""LlamaIndex-compatible embeddings wrapper."""

from __future__ import annotations

from typing import List

from ds_rag_embedder.model import DSRAGEmbedder


class DSRAGLlamaIndexEmbedding:
    """
    Minimal LlamaIndex embedding adapter.

    Usage with LlamaIndex:
        from llama_index.core import Settings
        from ds_rag_embedder.integrations.llama_index import DSRAGLlamaIndexEmbedding
        Settings.embed_model = DSRAGLlamaIndexEmbedding()
    """

    def __init__(self, model_name_or_path: str | None = None) -> None:
        self._embedder = DSRAGEmbedder(model_name_or_path=model_name_or_path)

    @property
    def embed_batch_size(self) -> int:
        return 32

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embedder.encode_queries([query])[0].tolist()

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embedder.encode_documents([text])[0].tolist()

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embedder.encode_documents(texts).tolist()

    def get_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    def get_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)
