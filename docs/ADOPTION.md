# Adoption Guide: Swap Generic Embeddings in 5 Minutes

## Why teams switch

Generic embedders miss DS terminology and task phrasing. **DS RAG Embedder v1** improves retrieval for notebook docs, runbooks, and experiment logs.

## Drop-in replacement

### Before (MiniLM)

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
q_emb = model.encode(["How to handle class imbalance?"], normalize_embeddings=True)
```

### After (DS RAG Embedder)

```python
from ds_rag_embedder import DSRAGEmbedder
embedder = DSRAGEmbedder("waghelad/ds-rag-embedder-v1")
q_emb = embedder.encode_queries(["How to handle class imbalance?"])
d_emb = embedder.encode_documents(documents)
```

## LangChain

```python
from ds_rag_embedder.integrations.langchain import DSRAGLangChainEmbeddings
embeddings = DSRAGLangChainEmbeddings("waghelad/ds-rag-embedder-v1")
```

## Production pattern: hybrid retrieval

Use dense + BM25 when exact tokens matter (SMOTE, PSI, AUC):

```python
from ds_rag_embedder.rag import HybridRetriever

hybrid = HybridRetriever(embedder=embedder, documents=docs, alpha=0.65)
hits = hybrid.retrieve(user_query, top_k=5).hits
```

See `examples/hybrid_retrieval_example.py`.

## Checklist after swap

1. Run `python scripts/benchmark_report.py` on your corpus sample
2. Log Recall@5 on live queries for 2 weeks
3. Pin HF model + dataset + Space on your profile
4. Add model card link to internal runbooks

## Launch channels

- Hugging Face Community post
- Kaggle notebook (included in repo)
- LinkedIn post with before/after retrieval example
- Cross-link GitHub + HF + Kaggle in one pinned Collection
