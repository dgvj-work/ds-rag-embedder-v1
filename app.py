"""
DS RAG Embedder — Hugging Face Space demo.

Interactive retrieval over Data Science & ML documentation using
waghelad/ds-rag-embedder-v1 (or local checkpoint fallback).
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
CORPUS_PATH = ROOT / "data" / "corpus" / "ds_ml_corpus.jsonl"
MODEL_ID = os.environ.get("DS_RAG_MODEL", "waghelad/ds-rag-embedder-v1")
FALLBACK_MODEL = "BAAI/bge-small-en-v1.5"

EXAMPLE_QUERIES = [
    "How do I handle class imbalance?",
    "What is nested cross validation?",
    "Best chunk size for RAG on ML docs?",
    "How to detect data drift in production?",
    "When do I need feature scaling?",
    "How to evaluate RAG pipeline quality?",
    "Prevent target encoding leakage",
    "Time series cross validation best practice",
]


def _load_embedder():
    from ds_rag_embedder.model import DSRAGEmbedder

    for model_id in (MODEL_ID, str(ROOT / "models" / "ds-rag-embedder-v1"), FALLBACK_MODEL):
        try:
            if model_id.startswith("waghelad/") or Path(model_id).exists() or "/" in model_id:
                return DSRAGEmbedder(model_name_or_path=model_id), model_id
        except Exception:
            continue
    return DSRAGEmbedder(model_name_or_path=FALLBACK_MODEL), FALLBACK_MODEL


def _load_retriever():
    from ds_rag_embedder.rag.retriever import DSRAGRetriever

    embedder, model_name = _load_embedder()
    retriever = DSRAGRetriever(embedder)
    if CORPUS_PATH.exists():
        retriever.load_corpus_jsonl(str(CORPUS_PATH))
    retriever.build_index(show_progress=False)
    return retriever, model_name


_RETRIEVER = None
_MODEL_NAME = ""


def get_retriever():
    global _RETRIEVER, _MODEL_NAME
    if _RETRIEVER is None:
        _RETRIEVER, _MODEL_NAME = _load_retriever()
    return _RETRIEVER, _MODEL_NAME


def search(query: str, top_k: int = 5) -> tuple[str, str]:
    if not query.strip():
        return "Enter a Data Science question.", ""
    retriever, model_name = get_retriever()
    result = retriever.retrieve(query.strip(), top_k=int(top_k))
    lines = [f"**Model:** `{model_name}` · **Query:** {result.query}\n"]
    for hit in result.hits:
        meta = hit.get("metadata") or {}
        cat = meta.get("category", "general")
        lines.append(
            f"### #{hit['rank']} · score {hit['score']:.4f} · `{cat}`\n{hit['text']}\n"
        )
    prompt_lines = [
        "You are a senior data scientist. Answer using the contexts below.\n",
    ]
    for i, hit in enumerate(result.hits, 1):
        prompt_lines.append(f"[Context {i}]\n{hit['text']}\n")
    prompt_lines.append(f"Question: {query}\nAnswer:")
    return "\n".join(lines), "\n".join(prompt_lines)


def compare_query(query: str) -> str:
    """Side-by-side generic vs DS embedder (when both available)."""
    from ds_rag_embedder.model import DSRAGEmbedder

    if not query.strip() or not CORPUS_PATH.exists():
        return "Enter a query to compare embedders."
    import json

    docs = []
    with CORPUS_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                docs.append(json.loads(line))
    texts = [d["text"] for d in docs[:200]]
    lines = [f"**Query:** {query}\n"]
    for name, mid in [
        ("Generic MiniLM", "sentence-transformers/all-MiniLM-L6-v2"),
        ("BGE small", "BAAI/bge-small-en-v1.5"),
    ]:
        try:
            emb = DSRAGEmbedder(model_name_or_path=mid)
            top = emb.search(query, texts, top_k=1)[0]
            lines.append(f"**{name}** → score {top['score']:.4f}\n> {top['document'][:220]}…\n")
        except Exception as exc:
            lines.append(f"**{name}** — unavailable ({exc})\n")
    try:
        emb = DSRAGEmbedder(model_name_or_path=MODEL_ID)
        top = emb.search(query, texts, top_k=1)[0]
        lines.append(f"**DS RAG Embedder** → score {top['score']:.4f}\n> {top['document'][:220]}…\n")
    except Exception:
        lines.append("**DS RAG Embedder** — upload model to HF or train locally first.\n")
    return "\n".join(lines)


with gr.Blocks(title="DS RAG Embedder Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
# 🔬 DS RAG Embedder v1
**Domain-specific embeddings for Data Science & ML documentation retrieval**

Retrieve notebook guidance, metrics explainers, MLOps runbooks, and RAG best practices.
Built for daily RAG pipelines — LangChain · LlamaIndex · Chroma · FAISS compatible.

**Model:** [`waghelad/ds-rag-embedder-v1`](https://huggingface.co/waghelad/ds-rag-embedder-v1)
"""
    )
    with gr.Tab("Retrieve"):
        with gr.Row():
            query_in = gr.Textbox(
                label="Data Science question",
                placeholder="e.g. How do I detect data leakage in feature engineering?",
                lines=2,
            )
            top_k = gr.Slider(1, 10, value=5, step=1, label="Top-K passages")
        search_btn = gr.Button("Retrieve", variant="primary")
        results_md = gr.Markdown(label="Retrieved contexts")
        prompt_out = gr.Textbox(label="LLM-ready RAG prompt", lines=12)
        gr.Examples(EXAMPLE_QUERIES, inputs=query_in)
        search_btn.click(search, [query_in, top_k], [results_md, prompt_out])
        query_in.submit(search, [query_in, top_k], [results_md, prompt_out])

    with gr.Tab("Compare embedders"):
        cmp_query = gr.Textbox(label="Query", lines=2)
        cmp_btn = gr.Button("Compare")
        cmp_out = gr.Markdown()
        cmp_btn.click(compare_query, cmp_query, cmp_out)

    with gr.Tab("Quick start"):
        gr.Markdown(
            """
```python
from ds_rag_embedder import DSRAGEmbedder

embedder = DSRAGEmbedder("waghelad/ds-rag-embedder-v1")
hits = embedder.search(
    "How to handle class imbalance?",
    documents=["Use SMOTE...", "Accuracy is misleading..."],
    top_k=3,
)
for h in hits:
    print(h["score"], h["document"][:80])
```

**Install:** `pip install sentence-transformers` then clone this repo or load from Hub.

Author: **Digvijay Waghela** · Apache-2.0
"""
        )

if __name__ == "__main__":
    demo.launch()
