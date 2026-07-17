# DS RAG Embedder v1

[![Model](https://img.shields.io/badge/🤗%20Model-ds--rag--embedder--v1-blue)](https://huggingface.co/waghelad/ds-rag-embedder-v1)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-ds--rag--eval--v1-blue)](https://huggingface.co/datasets/waghelad/ds-rag-eval-v1)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)

**Domain-specific RAG embedding model for Data Science & ML documentation retrieval.**

Fine-tuned from `BAAI/bge-small-en-v1.5` for metrics, leakage, CV, imbalance, MLOps, RAG, time series, deployment, and experiment design queries.

| Use case | Benefit |
|----------|---------|
| RAG over DS/ML docs | Better Recall@k vs generic MiniLM/BGE |
| Notebook / runbook search | Understands AUC, SMOTE, SHAP, PSI, nested CV |
| Daily pipelines | 384-dim, fast, LangChain/LlamaIndex/Chroma/FAISS ready |

## Quick start

```python
from ds_rag_embedder import DSRAGEmbedder

embedder = DSRAGEmbedder("waghelad/ds-rag-embedder-v1")
hits = embedder.search("How to detect data leakage?", documents, top_k=3)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[demo,dev]"
python scripts/build_corpus.py
./scripts/run_local.sh
```

## Publish to Hugging Face

```bash
hf auth login
./scripts/publish_hf.sh
```

See `docs/HF_UPLOAD.md`, `docs/TRAINING.md`, `docs/KAGGLE.md`.

Author: Digvijay Waghela · Apache-2.0
