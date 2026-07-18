---
title: DS RAG Embedder — ML Doc Retrieval Demo
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.12.0"
python_version: "3.11"
app_file: app.py
pinned: true
license: apache-2.0
short_description: Domain-tuned embeddings for DS/ML doc retrieval (R@1 0.85)
tags:
  - rag
  - retrieval
  - feature-extraction
  - sentence-similarity
  - data-science
  - machine-learning
  - embeddings
  - gradio
models:
  - waghelad/ds-rag-embedder-v1
datasets:
  - waghelad/ds-rag-eval-v1
suggested_hardware: cpu-basic
---

# DS RAG Embedder — Documentation Retrieval Demo

Interactive semantic search over **600 Data Science & ML passages** using
[`waghelad/ds-rag-embedder-v1`](https://huggingface.co/waghelad/ds-rag-embedder-v1).

**Benchmark:** Recall@1 **0.851** · Recall@5 **1.000** on 87 held-out queries (vs MiniLM 0.621).

Try example queries for metrics, leakage, MLOps, RAG evaluation, and deployment — or paste your own retrieval query.
