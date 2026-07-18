# Hugging Face Community launch post

Copy for manual posting at [huggingface.co](https://huggingface.co) if the script is unavailable.
Used by `scripts/publish_hf_community.py` as the discussion body.

---

## TL;DR

**DS RAG Embedder v1** is a domain-specific embedding model for retrieval over Data Science, ML, and AI documentation. Fine-tuned from BGE-small on 600+ passages with a public eval benchmark.

**Benchmark (87 queries):** Recall@1 **0.851** · Recall@5 **1.000** (vs MiniLM 0.621 / BGE 0.506)

## Links

| Resource | URL |
|----------|-----|
| Model | https://huggingface.co/waghelad/ds-rag-embedder-v1 |
| Eval dataset | https://huggingface.co/datasets/waghelad/ds-rag-eval-v1 |
| Live demo Space | https://huggingface.co/spaces/waghelad/ds-rag-embedder-demo |
| GitHub | https://github.com/dgvj-work/ds-rag-embedder-v1 |
| PyPI | https://pypi.org/project/ds-rag-embedder/ |
| Kaggle notebook | https://www.kaggle.com/code/waghelad/ds-rag-embedder-v1-train-benchmark |

## Why this model?

Generic embedders miss DS/ML task intent: class imbalance, nested CV, target leakage, PSI drift, RAG eval metrics, SMOTE, experiment tracking, and MLOps runbooks.

This model uses a **BGE-style query prefix** for asymmetric retrieval and ships with:

- Hybrid BM25 + dense retriever
- LangChain / LlamaIndex adapters
- Full train → eval → export pipeline
- Verified benchmark artifacts on GitHub

## Quick start

```python
pip install ds-rag-embedder sentence-transformers

from ds_rag_embedder import DSRAGEmbedder

embedder = DSRAGEmbedder("waghelad/ds-rag-embedder-v1")
hits = embedder.search(
    "How do I prevent target encoding leakage?",
    documents=["Target encoding before split leaks label information...", "..."],
    top_k=5,
)
for h in hits:
    print(h["score"], h["document"][:100])
```

## Try the demo

Open the **Gradio Space** and run a retrieval query against 600 curated DS/ML passages. The demo returns ranked passages plus an LLM-ready RAG prompt.

## Reproduce benchmarks

```bash
git clone https://github.com/dgvj-work/ds-rag-embedder-v1
cd ds-rag-embedder-v1
pip install -e ".[dev]"
python scripts/benchmark_report.py --model waghelad/ds-rag-embedder-v1
```

Results are saved to `outputs/eval_results.json`.

## Feedback welcome

If you use this in a RAG stack, experiment tracker, or internal doc search, please share:

- Your domain (metrics, MLOps, notebooks, etc.)
- Recall@k before/after vs your baseline embedder
- Feature requests for v2 (multilingual, code-aware chunks, larger corpus)

Apache-2.0 · Digvijay Waghela
