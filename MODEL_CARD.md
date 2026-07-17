---
language: en
license: apache-2.0
library_name: sentence-transformers
pipeline_tag: feature-extraction
tags:
  - sentence-transformers
  - feature-extraction
  - sentence-similarity
  - rag
  - retrieval
  - data-science
  - machine-learning
  - embeddings
  - domain-specific
  - text-embeddings-inference
  - hybrid-search
datasets:
  - waghelad/ds-rag-eval-v1
metrics:
  - cosine similarity
  - recall at k
  - mrr
  - ndcg
base_model: BAAI/bge-small-en-v1.5
---

# DS RAG Embedder v1

Domain-specific embedding model for **Data Science and ML documentation retrieval** in RAG pipelines.

Fine-tuned from [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) on 600+ passages covering metrics, leakage, cross-validation, imbalance, MLOps, RAG, deep learning, deployment, and experiment design.

## Quick start

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("waghelad/ds-rag-embedder-v1")

query = (
    "Represent this Data Science question for retrieving relevant documentation: "
    "How do I detect data leakage in feature engineering?"
)
docs = [
    "Target encoding before split leaks label information. Fit encoders inside CV folds only.",
    "Accuracy is misleading when classes are imbalanced; use PR-AUC and per-class F1.",
]

q_emb = model.encode([query], normalize_embeddings=True)
d_emb = model.encode(docs, normalize_embeddings=True)
scores = q_emb @ d_emb.T
print(scores)
```

### Python package

```python
from ds_rag_embedder import DSRAGEmbedder

embedder = DSRAGEmbedder("waghelad/ds-rag-embedder-v1")
for hit in embedder.search("nested cross validation", docs, top_k=3):
    print(hit["score"], hit["document"][:100])
```

## Query prefix (required)

| Input | Prefix |
|-------|--------|
| Queries | `Represent this Data Science question for retrieving relevant documentation: ` |
| Documents | None (encode as-is) |

`DSRAGEmbedder.encode_queries()` applies this automatically.

## Benchmark (DS RAG Eval v1)

Run locally after training:

```bash
git clone https://github.com/dgvj-work/ds-rag-embedder-v1
cd ds-rag-embedder-v1
python scripts/benchmark_report.py --model models/ds-rag-embedder-v1
```

| Model | Recall@1 | Recall@5 | MRR | nDCG@10 |
|-------|----------|----------|-----|---------|
| all-MiniLM-L6-v2 | run report | run report | run report | run report |
| bge-small-en-v1.5 | run report | run report | run report | run report |
| **ds-rag-embedder-v1** | run report | run report | run report | run report |

Paste verified numbers from `outputs/benchmark_report.md` after training.

## Hybrid retrieval (BM25 + dense)

Production teams often combine lexical and dense search:

```python
from ds_rag_embedder.rag import HybridRetriever

hybrid = HybridRetriever(embedder=embedder, documents=docs, alpha=0.65)
hits = hybrid.retrieve("SMOTE leakage cross validation", top_k=5).hits
```

## Integrations

- LangChain: `DSRAGLangChainEmbeddings`
- LlamaIndex: `DSRAGLlamaIndexEmbedding`
- Chroma, FAISS: see `examples/`
- Gradio Space demo included in repo

## Model details

| Property | Value |
|----------|-------|
| Base model | BAAI/bge-small-en-v1.5 |
| Embedding dim | 384 |
| Max seq length | 512 |
| Normalization | L2 cosine |
| Training loss | MultipleNegativesRankingLoss |
| Eval dataset | waghelad/ds-rag-eval-v1 |
| Language | English |

## Intended use

Good for RAG over DS/ML docs, notebook search, experiment runbooks, and data-team copilots.

Not for general web search, legal/medical use without evaluation, or fully automated high-stakes decisions.

## Links

- GitHub: https://github.com/dgvj-work/ds-rag-embedder-v1
- Dataset: https://huggingface.co/datasets/waghelad/ds-rag-eval-v1
- Kaggle notebook: `notebooks/kaggle_ds_rag_embedder.ipynb`

## Citation

```bibtex
@misc{waghela2026dsrag,
  author = {Digvijay Waghela},
  title = {DS RAG Embedder v1: Domain Embeddings for Data Science Documentation Retrieval},
  year = {2026},
  howpublished = {\\url{https://huggingface.co/waghelad/ds-rag-embedder-v1}}
}
```

## Author

Digvijay Waghela · digvijay.vaghela@yahoo.com · Apache-2.0
