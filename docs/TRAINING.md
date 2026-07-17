# Training Guide

## Overview

DS RAG Embedder v1 fine-tunes `BAAI/bge-small-en-v1.5` with **MultipleNegativesRankingLoss** on query-passage pairs derived from the DS/ML corpus.

## Prerequisites

- Python 3.10+
- GPU recommended (T4/V100/A10); CPU works for smoke tests with fewer epochs
- ~2 GB disk for base model cache

## Steps

```bash
source .venv/bin/activate
pip install -e ".[dev,demo]"

# 1. Build corpus (600 passages + eval pairs)
python scripts/build_corpus.py --corpus-size 600

# 2. Train
python scripts/train.py --epochs 4 --batch-size 32

# 3. Evaluate
python scripts/evaluate.py --model models/ds-rag-embedder-v1 --compare
```

## Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `base_model` | BAAI/bge-small-en-v1.5 | Strong baseline for retrieval |
| `epochs` | 4 | Increase to 6 for larger corpus |
| `batch_size` | 32 | Lower if OOM |
| `learning_rate` | 2e-5 | Standard for ST fine-tuning |
| `max_seq_length` | 512 | Sufficient for doc chunks |

## Output

Training writes to `models/ds-rag-embedder-v1/` including:
- Sentence-Transformers model files
- `training_meta.json` with run stats

## Kaggle / Colab

Use `notebooks/kaggle_ds_rag_embedder.ipynb` for GPU training in the cloud.
