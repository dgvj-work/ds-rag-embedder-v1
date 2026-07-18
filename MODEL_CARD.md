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
| all-MiniLM-L6-v2 | 0.621 | 0.828 | 0.708 | 0.740 |
| bge-small-en-v1.5 | 0.506 | 0.609 | 0.558 | 0.567 |
| **ds-rag-embedder-v1** | **0.851** | **1.000** | **0.921** | **0.942** |

Full JSON: [`outputs/eval_results.json`](outputs/eval_results.json) on GitHub.

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
- Demo Space: https://huggingface.co/spaces/waghelad/ds-rag-embedder-demo
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


## Latest evaluation

```json
{
  "generated_at": "2026-07-17T23:48:05.329581+00:00",
  "models": {
    "all-MiniLM-L6-v2": {
      "recall_at_1": 0.6206896551724138,
      "recall_at_3": 0.7931034482758621,
      "recall_at_5": 0.8275862068965517,
      "recall_at_10": 0.8390804597701149,
      "mrr": 0.7079458024662366,
      "ndcg_at_10": 0.7396208395585102,
      "num_queries": 87,
      "latency_ms_per_query": 76.45090660963464
    },
    "bge-small-en-v1.5": {
      "recall_at_1": 0.5057471264367817,
      "recall_at_3": 0.6091954022988506,
      "recall_at_5": 0.6091954022988506,
      "recall_at_10": 0.6091954022988506,
      "mrr": 0.5579086972039102,
      "ndcg_at_10": 0.5665009025451581,
      "num_queries": 87,
      "latency_ms_per_query": 27.16016379350946
    },
    "ds-rag-embedder-v1": {
      "recall_at_1": 0.8505747126436781,
      "recall_at_3": 1.0,
      "recall_at_5": 1.0,
      "recall_at_10": 1.0,
      "mrr": 0.9214559386973179,
      "ndcg_at_10": 0.9418416929802992,
      "num_queries": 87,
      "latency_ms_per_query": 8.395674322656859
    }
  },
  "category_breakdown": {
    "ds-rag-embedder-v1": {
      "overall": {
        "recall_at_1": 0.8505747126436781,
        "recall_at_3": 1.0,
        "recall_at_5": 1.0,
        "recall_at_10": 1.0,
        "mrr": 0.9214559386973179,
        "ndcg_at_10": 0.9418416929802992,
        "num_queries": 87,
        "latency_ms_per_query": 8.590979401246997
      },
      "by_category": {
        "data_leakage": {
          "recall_at_1": 1.0,
          "recall_at_3": 1.0,
          "recall_at_5": 1.0,
          "recall_at_10": 1.0,
          "mrr": 1.0,
          "ndcg_at_10": 1.0,
          "num_queries": 3,
          "latency_ms_per_query": null
        },
        "deep_learning": {
          "recall_at_1": 0.8888888888888888,
          "recall_at_3": 1.0,
          "recall_at_5": 1.0,
          "recall_at_10": 1.0,
          "mrr": 0.9444444444444444,
          "ndcg_at_10": 0.9589921948412731,
          "num_queries": 9,
          "latency_ms_per_query": null
        },
        "deployment": {
          "recall_at_1": 0.0,
          "recall_at_3": 1.0,
          "recall_at_5": 1.0,
          "recall_at_10": 1.0,
          "mrr": 0.4444444444444444,
          "ndcg_at_10": 0.5872865023809717,
          "num_queries": 3,
          "latency_ms_per_query": null
        },
        "experiment_design": {
          "recall_at_1": 1.0,
          "recall_at_3": 1.0,
          "recall_at_5": 1.0,
          "recall_at_10": 1.0,
          "mrr": 1.0,
          "ndcg_at_10": 1.0,
          "num_queries": 3,
          "latency_ms_per_query": null
        },
        "feature_engineering": {
          "recall_at_1": 0.8333333333333334,
          "recall_at_3": 1.0,
          "recall_at_5": 1.0,
          "recall_at_10": 1.0,
          "mrr": 0.888888888888889,
          "ndcg_at_10": 0.9166666666666666,
          "num_queries": 6,
          "latency_ms_per_query": null
        },
        "hyperparameter_tuning": {
          "recall_at_1": 0.6666666666666666,
          "recall_at_3": 1.0,
          "recall_at_5": 1.0,
          "recall_at_10": 1.0,
          "mrr": 0.8333333333333334,
          "ndcg_at_10": 0.8769765845238192,
          "num_queries": 3,
          "latency_ms_per_query": null
        },
        "imbalanced_data": {
          "recall_at_1": 0.6666666666666666,
          "recall_at_3": 1.0,
          "recall_at_5": 1.0,
          "recall_at_10": 1.0,
          "mrr": 0.8333333333333334,
          "ndcg_at_10": 0.8769765845238192,
          "num_queries": 3,
          "latency_ms_per_query": null
        },
        "interpretability": {
          "recall_at_1": 1.0,
          "recall_at_3": 1.0,
          "recall_at_5": 1.0,
          "recall_at_10": 1.0,
          "mrr": 1.0,
          "ndcg_at_10": 1.0,
          "num_queries": 3,
          "latency_ms_per_query": null
        },
        "metrics": {
          "recall_at_1": 0.8888888888888888,
          "recall_at_3": 1.0,
          "recall_at_5": 1.0,
          "recall_at_10": 1.0,
          "mrr": 0.9444444444444444,
          "ndcg_at_10": 0.9589921948412731,
          "num_queries": 9,
          "latency_ms_per_query": null
        },
        "mlops": {
          "recall_at_1": 1.0,
          "recall_at_3": 1.0,
          "recall_at_5": 1.0,
          "recall_at_10": 1.0,
          "mrr": 1.0,
          "ndcg_at_10": 1.0,
          "num_queries": 9,
          "latency_ms_per_query": null
        },
        "model_selection": {
          "recall_at_1": 1.0,
          "recall_at_3": 1.0,
          "recall_at_5": 1.0,
          "recall_at_10": 1.0,
          "mrr": 1.0,
          "ndcg_at_10": 1.0,
          "num_queries": 6,
          "latency_ms_per_query": null
        },
        "nlp": {
          "recall_at_1": 1.0,
          "recall_at_3": 1.0,
          "recall_at_5": 1.0,
          "recall_at_10": 1.0,
          "mrr": 1.0,
          "ndcg_at_10": 1.0,
          "num_queries": 3,
          "latency_ms_per_query": null
        },
        "pandas_sklearn": {
          "recall_at_1": 1.0,
          "recall_at_3": 1.0,
          "recall_at_5": 1.0,
          "recall_at_10": 1.0,
          "mrr": 1.0,
          "ndcg_at_10": 1.0,
          "num_queries": 3,
          "latency_ms_per_query": null
        },
        "rag_llm": {
          "recall_at_1": 0.7333333333333333,
          "recall_at_3": 1.0,
          "recall_at_5": 1.0,
          "recall_at_10": 1.0,
          "mrr": 0.8666666666666667,
          "ndcg_at_10": 0.9015812676190554,
          "num_queries": 15,
          "latency_ms_per_query": null
        },
        "statistics": {
          "recall_at_1": 1.0,
          "recall_at_3": 1.0,
          "recall_at_5": 1.0,
          "recall_at_10": 1.0,
          "mrr": 1.0,
          "ndcg_at_10": 1.0,
          "num_queries": 3,
          "latency_ms_per_query": null
        },
        "time_series": {
          "recall_at_1": 1.0,
          "recall_at_3": 1.0,
          "recall_at_5": 1.0,
          "recall_at_10": 1.0,
          "mrr": 1.0,
          "ndcg_at_10": 1.0,
          "num_queries": 3,
          "latency_ms_per_query": null
        },
        "validation": {
          "recall_at_1": 0.6666666666666666,
          "recall_at_3": 1.0,
          "recall_at_5": 1.0,
          "recall_at_10": 1.0,
          "mrr": 0.8333333333333334,
          "ndcg_at_10": 0.8769765845238192,
          "num_queries": 3,
          "latency_ms_per_query": null
        }
      }
    }
  }
}
```
