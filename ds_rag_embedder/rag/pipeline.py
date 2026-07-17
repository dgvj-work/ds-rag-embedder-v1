"""End-to-end RAG pipeline for Data Science documentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ds_rag_embedder.config import CORPUS_PATH
from ds_rag_embedder.model import DSRAGEmbedder
from ds_rag_embedder.rag.retriever import DSRAGRetriever


@dataclass
class RAGResponse:
    query: str
    contexts: list[str]
    hits: list[dict[str, Any]]
    prompt: str


class DSRAGPipeline:
    """
    Lightweight RAG pipeline: retrieve DS/ML docs → format context for LLM.

    Works with any LLM via the generated prompt; no vendor lock-in.
    """

    def __init__(
        self,
        embedder: DSRAGEmbedder | None = None,
        corpus_path: str | None = None,
        top_k: int = 4,
    ) -> None:
        self.embedder = embedder or DSRAGEmbedder()
        self.retriever = DSRAGRetriever(self.embedder)
        self.top_k = top_k
        path = corpus_path or str(CORPUS_PATH)
        self.retriever.load_corpus_jsonl(path)
        self.retriever.build_index()

    def retrieve(self, query: str, top_k: int | None = None) -> RAGResponse:
        k = top_k or self.top_k
        result = self.retriever.retrieve(query, top_k=k)
        contexts = [h["text"] for h in result.hits]
        prompt = self.format_prompt(query, contexts)
        return RAGResponse(
            query=query,
            contexts=contexts,
            hits=result.hits,
            prompt=prompt,
        )

    @staticmethod
    def format_prompt(query: str, contexts: list[str]) -> str:
        blocks = "\n\n".join(f"[Context {i+1}]\n{c}" for i, c in enumerate(contexts))
        return (
            "You are a senior data scientist assistant. Answer using ONLY the retrieved "
            "documentation contexts below. If insufficient, say what is missing.\n\n"
            f"{blocks}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

    def answer_with_llm(self, query: str, llm_client: Any, model: str | None = None) -> str:
        """Optional helper when an OpenAI-compatible or HF Inference client is available."""
        rag = self.retrieve(query)
        if hasattr(llm_client, "chat_completion"):
            resp = llm_client.chat_completion(
                model=model,
                messages=[{"role": "user", "content": rag.prompt}],
            )
            return resp.choices[0].message.content
        if hasattr(llm_client, "chat"):
            resp = llm_client.chat.completions.create(
                model=model or "gpt-4o-mini",
                messages=[{"role": "user", "content": rag.prompt}],
            )
            return resp.choices[0].message.content
        raise TypeError("Unsupported llm_client; use retrieve() and your own LLM call")
