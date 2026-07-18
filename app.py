"""
DS RAG Embedder — Hugging Face Space demo.

Production-style retrieval demo for Data Science, ML, and AI documentation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
CORPUS_PATH = ROOT / "data" / "corpus" / "ds_ml_corpus.jsonl"
MODEL_ID = os.environ.get("DS_RAG_MODEL", "waghelad/ds-rag-embedder-v1")
FALLBACK_MODEL = "BAAI/bge-small-en-v1.5"

MODEL_URL = "https://huggingface.co/waghelad/ds-rag-embedder-v1"
DATASET_URL = "https://huggingface.co/datasets/waghelad/ds-rag-eval-v1"
GITHUB_URL = "https://github.com/dgvj-work/ds-rag-embedder-v1"

EXAMPLE_QUERIES = [
    ["How do I detect target encoding leakage before train/test split?"],
    ["What metric should I use for imbalanced classification?"],
    ["Explain nested cross-validation for hyperparameter tuning"],
    ["How do I monitor data drift with PSI in production?"],
    ["Best chunking strategy for RAG over ML runbooks"],
    ["When is SMOTE appropriate and how do I avoid leakage?"],
    ["How to evaluate RAG retrieval beyond BLEU scores?"],
    ["Time series cross-validation without shuffle leakage"],
]

WELCOME_RESULTS = """
### Welcome — try a retrieval query above

This demo searches **600 curated DS/ML passages** (metrics, validation, MLOps, RAG, deployment)
using the fine-tuned embedder **`waghelad/ds-rag-embedder-v1`**.

**Benchmark (87 queries):** Recall@1 **0.851** · Recall@5 **1.000** vs generic MiniLM **0.621**

Pick an example query or type your own — results and an LLM-ready prompt appear below.
"""

CUSTOM_CSS = """
#hero-title { margin-bottom: 0.25rem; }
#hero-sub { color: #475569; font-size: 1.05rem; margin-top: 0; }
.stat-row { display: flex; flex-wrap: wrap; gap: 0.75rem; margin: 1rem 0 1.25rem 0; }
.stat-pill {
  background: linear-gradient(135deg, #eff6ff 0%, #ecfdf5 100%);
  border: 1px solid #dbeafe;
  border-radius: 999px;
  padding: 0.35rem 0.85rem;
  font-size: 0.82rem;
  font-weight: 600;
  color: #1e3a5f;
}
.link-row a { margin-right: 1rem; font-weight: 600; }
.section-label { font-weight: 700; color: #0f172a; margin-bottom: 0.35rem; }
.hint { color: #64748b; font-size: 0.9rem; margin-top: 0; }
footer { margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #e2e8f0; color: #64748b; font-size: 0.85rem; }
"""


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


def _format_hit(hit: dict) -> str:
    meta = hit.get("metadata") or {}
    cat = meta.get("category", "general").replace("_", " ")
    title = meta.get("title") or meta.get("id") or f"Passage {hit['rank']}"
    return (
        f"#### {hit['rank']}. {title}\n"
        f"<span class='stat-pill'>{cat}</span> "
        f"**Relevance score:** `{hit['score']:.3f}`\n\n"
        f"{hit['text']}\n"
    )


def search(query: str, top_k: int = 5) -> tuple[str, str, str]:
    if not query.strip():
        return (
            WELCOME_RESULTS,
            "",
            "Type a retrieval query to search the DS/ML documentation index.",
        )
    retriever, model_name = get_retriever()
    result = retriever.retrieve(query.strip(), top_k=int(top_k))
    header = (
        f"**{len(result.hits)} passages** retrieved · "
        f"model [`{model_name}`]({MODEL_URL}) · "
        f"corpus: 600 docs\n\n---\n\n"
    )
    body = "".join(_format_hit(h) for h in result.hits)
    results_md = header + body if result.hits else header + "_No matches above threshold — try rephrasing._"

    prompt_lines = [
        "You are a senior data scientist assistant. Answer using ONLY the retrieved contexts.\n",
    ]
    for i, hit in enumerate(result.hits, 1):
        prompt_lines.append(f"[Context {i}]\n{hit['text']}\n")
    prompt_lines.append(f"Question: {query.strip()}\nAnswer:")
    prompt = "\n".join(prompt_lines)

    status = f"Loaded `{model_name}` · indexed {len(retriever.documents)} passages"
    return results_md, prompt, status


def hybrid_search(query: str, top_k: int = 5, alpha: float = 0.65) -> str:
    if not query.strip():
        return "_Enter a query to run hybrid BM25 + dense retrieval._"
    from ds_rag_embedder.rag import HybridRetriever

    embedder, model_name = _load_embedder()
    rows = []
    with CORPUS_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    texts = [r["text"] for r in rows]
    meta = [{"category": r.get("category"), "title": r.get("title")} for r in rows]
    retriever = HybridRetriever(embedder=embedder, documents=texts, metadata=meta, alpha=alpha)
    result = retriever.retrieve(query.strip(), top_k=int(top_k))
    lines = [
        f"**Hybrid retrieval** · dense weight `{alpha:.2f}` · model `{model_name}`\n\n",
    ]
    for hit in result.hits:
        cat = (hit.get("metadata") or {}).get("category", "general").replace("_", " ")
        lines.append(
            f"#### {hit['rank']}. combined `{hit['score']:.3f}` "
            f"(dense `{hit['dense_score']:.3f}` · BM25 `{hit['bm25_score']:.3f}`) · {cat}\n\n"
            f"{hit['text']}\n\n"
        )
    return "".join(lines)


def compare_query(query: str) -> str:
    from ds_rag_embedder.model import DSRAGEmbedder

    if not query.strip() or not CORPUS_PATH.exists():
        return "_Enter a query to compare base vs domain-specific retrieval._"

    docs = []
    with CORPUS_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                docs.append(json.loads(line))
    texts = [d["text"] for d in docs[:200]]
    lines = [
        f"**Query:** _{query.strip()}_\n\n",
        "| Embedder | Top-1 score | Snippet |\n",
        "|----------|-------------|--------|\n",
    ]
    for name, mid in [
        ("BGE small (generic base)", "BAAI/bge-small-en-v1.5"),
        ("DS RAG Embedder v1 (domain)", MODEL_ID),
    ]:
        try:
            emb = DSRAGEmbedder(model_name_or_path=mid)
            top = emb.search(query, texts, top_k=1)[0]
            snippet = top["document"][:120].replace("|", "/") + "…"
            lines.append(f"| **{name}** | `{top['score']:.3f}` | {snippet} |\n")
        except Exception as exc:
            lines.append(f"| {name} | error | {exc} |\n")
    lines.append(
        "\n\n_Benchmark on 87 DS queries: DS RAG Embedder Recall@1 **0.851** vs BGE **0.506**._"
    )
    return "".join(lines)


with gr.Blocks(
    title="DS RAG Embedder — ML Documentation Retrieval",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="emerald"),
    css=CUSTOM_CSS,
) as demo:
    gr.HTML(
        """
<div id="hero-title">
  <h1 style="margin:0;font-size:1.85rem;">DS RAG Embedder v1</h1>
</div>
<p id="hero-sub">
  Domain-specific embeddings for retrieval over <strong>Data Science, ML, and AI documentation</strong>.
  Built for production RAG stacks — LangChain, LlamaIndex, Chroma, FAISS, Hugging Face TEI.
</p>
<div class="stat-row">
  <span class="stat-pill">Recall@1: 0.851</span>
  <span class="stat-pill">Recall@5: 1.000</span>
  <span class="stat-pill">600 doc passages</span>
  <span class="stat-pill">87 eval queries</span>
  <span class="stat-pill">Apache-2.0</span>
</div>
<div class="link-row">
  <a href="https://huggingface.co/waghelad/ds-rag-embedder-v1" target="_blank">Model card</a>
  <a href="https://huggingface.co/datasets/waghelad/ds-rag-eval-v1" target="_blank">Eval dataset</a>
  <a href="https://github.com/dgvj-work/ds-rag-embedder-v1" target="_blank">GitHub</a>
</div>
"""
    )

    model_status = gr.Markdown(
        value="_First search loads the embedder (~30s on CPU). Subsequent queries are fast._"
    )

    with gr.Tabs():
        with gr.Tab("Semantic search"):
            gr.Markdown(
                """
**What this tab does:** semantic search over a DS/ML knowledge base —
metrics, validation, feature engineering, MLOps, RAG, deployment, and experiment design.

Use a **natural-language retrieval query** (the question you would ask your doc search or RAG copilot).
"""
            )
            with gr.Row():
                with gr.Column(scale=3):
                    query_in = gr.Textbox(
                        label="Retrieval query",
                        placeholder=(
                            "e.g. How do I prevent target encoding leakage when building sklearn pipelines?"
                        ),
                        lines=3,
                        info="Ask what you need from ML docs, runbooks, or internal playbooks — not a chat message.",
                    )
                    with gr.Row():
                        search_btn = gr.Button("Search documentation", variant="primary", scale=2)
                        clear_btn = gr.Button("Clear", scale=1)
                    gr.Examples(
                        examples=EXAMPLE_QUERIES,
                        inputs=query_in,
                        label="Example queries (click to try)",
                    )
                with gr.Column(scale=1):
                    top_k = gr.Slider(
                        1,
                        10,
                        value=5,
                        step=1,
                        label="Results to return (Top-K)",
                        info="Number of passages passed to your RAG context window.",
                    )

            gr.Markdown("### Retrieved passages", elem_classes=["section-label"])
            results_md = gr.Markdown(value=WELCOME_RESULTS)

            with gr.Accordion("LLM-ready RAG prompt (copy into your copilot)", open=False):
                prompt_out = gr.Textbox(
                    label="Prompt template",
                    lines=14,
                    show_copy_button=True,
                    placeholder="Run a search to generate a grounded prompt with retrieved contexts.",
                )

            def _clear():
                return "", WELCOME_RESULTS, "", "_Ready._"

            search_btn.click(search, [query_in, top_k], [results_md, prompt_out, model_status])
            query_in.submit(search, [query_in, top_k], [results_md, prompt_out, model_status])
            clear_btn.click(_clear, outputs=[query_in, results_md, prompt_out, model_status])

        with gr.Tab("Hybrid search (BM25 + dense)"):
            gr.Markdown(
                """
Combine **lexical (BM25)** and **semantic (embedding)** search — recommended for production when
queries contain exact terms like _SMOTE_, _PSI_, _SHAP_, or _nested CV_.
"""
            )
            h_query = gr.Textbox(
                label="Retrieval query",
                placeholder="e.g. SMOTE class imbalance without leakage",
                lines=2,
            )
            with gr.Row():
                h_alpha = gr.Slider(
                    0.0,
                    1.0,
                    value=0.65,
                    step=0.05,
                    label="Semantic weight (alpha)",
                    info="Higher = trust embeddings more; lower = trust keyword match more.",
                )
                h_top_k = gr.Slider(1, 10, value=5, step=1, label="Top-K results")
            h_btn = gr.Button("Run hybrid search", variant="primary")
            h_out = gr.Markdown()
            h_btn.click(hybrid_search, [h_query, h_top_k, h_alpha], h_out)

        with gr.Tab("Embedder comparison"):
            gr.Markdown(
                """
Side-by-side **generic BGE** vs **DS RAG Embedder v1** on the same query.
Shows why domain fine-tuning matters for data-team documentation retrieval.
"""
            )
            cmp_query = gr.Textbox(
                label="Retrieval query",
                placeholder="e.g. nested cross validation hyperparameter tuning",
                lines=2,
            )
            cmp_btn = gr.Button("Compare embedders", variant="primary")
            cmp_out = gr.Markdown()
            cmp_btn.click(compare_query, cmp_query, cmp_out)

        with gr.Tab("Integrate in your stack"):
            gr.Markdown(
                f"""
### Python (Sentence Transformers)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("{MODEL_ID}")
query = (
    "Represent this Data Science question for retrieving relevant documentation: "
    "How do I detect data leakage in feature engineering?"
)
docs = ["Target encoding before split leaks label information...", "..."]
q_emb = model.encode([query], normalize_embeddings=True)
d_emb = model.encode(docs, normalize_embeddings=True)
scores = q_emb @ d_emb.T
```

### Package API

```python
from ds_rag_embedder import DSRAGEmbedder

embedder = DSRAGEmbedder("{MODEL_ID}")
hits = embedder.search("nested cross validation", documents, top_k=5)
```

### Resources

- [Model card]({MODEL_URL}) · [Eval dataset]({DATASET_URL}) · [GitHub]({GITHUB_URL})
- Integrations: LangChain, LlamaIndex, Chroma, FAISS (see repo `examples/`)

**Author:** Digvijay Waghela · **License:** Apache-2.0
"""
            )

    gr.HTML(
        """
<footer>
  DS RAG Embedder v1 — domain embeddings for Data Science documentation retrieval.
  Cite: <a href="https://huggingface.co/waghelad/ds-rag-embedder-v1">waghelad/ds-rag-embedder-v1</a>
</footer>
"""
    )

if __name__ == "__main__":
    demo.queue().launch()
