---
title: DS RAG Embedder Demo
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.12.0"
python_version: "3.11"
app_file: app.py
pinned: true
license: apache-2.0
short_description: DS/ML documentation RAG retrieval demo
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

# DS RAG Embedder Demo

Try domain-specific retrieval for **Data Science & ML** documentation.

1. Enter a DS/ML question
2. Click **Retrieve** → top passages + LLM-ready prompt
3. Use **Compare embedders** to see generic vs domain retrieval

Model: [`waghelad/ds-rag-embedder-v1`](https://huggingface.co/waghelad/ds-rag-embedder-v1)
