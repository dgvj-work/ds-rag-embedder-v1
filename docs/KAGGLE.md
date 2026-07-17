# Kaggle Notebook Guide

## File

`notebooks/kaggle_ds_rag_embedder.ipynb`

## How to publish

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → New Notebook
2. Upload the notebook or copy cells
3. Enable **GPU** accelerator (Settings)
4. Add dataset input: optional external ML docs
5. Set title: **"DS RAG Embedder v1 — Train & Evaluate Domain Embeddings"**
6. Tags: `rag`, `nlp`, `embeddings`, `data-science`, `huggingface`

## Suggested sections

1. Install dependencies
2. Build / load corpus
3. Fine-tune embedder
4. Benchmark vs MiniLM / BGE
5. Interactive retrieval demo
6. Push to Hugging Face Hub

## Link back to HF

Include in notebook conclusion:
- Model: https://huggingface.co/waghelad/ds-rag-embedder-v1
- Dataset: https://huggingface.co/datasets/waghelad/ds-rag-eval-v1

Kaggle → HF cross-links drive discovery for both platforms.
