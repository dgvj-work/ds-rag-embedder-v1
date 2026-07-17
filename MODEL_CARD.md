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

**Domain-specific embedding model for retrieving Data Science & ML documentation in RAG pipelines.**

Fine-tuned from [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) on 600+ curated DS/ML passages.

## Quick start

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("waghelad/ds-rag-embedder-v1")
query = "Represent this Data Science question for retrieving relevant documentation: How do I detect data leakage?"
q_emb = model.encode([query], normalize_embeddings=True)
```

## Query prefix

Use: `Represent this Data Science question for retrieving relevant documentation: `

Documents: no prefix.

## Integrations

LangChain · LlamaIndex · Chroma · FAISS · Hugging Face TEI

## Author

Digvijay Waghela · Apache-2.0
