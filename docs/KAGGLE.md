# Kaggle Notebook Guide

## File

`notebooks/kaggle_ds_rag_embedder.ipynb` (30 cells, comprehensive workflow)

Regenerate after edits:

```bash
python scripts/generate_kaggle_notebook.py
```

## Sections covered

1. Setup and clone from GitHub
2. Corpus build and EDA (category charts, text length stats)
3. Fine-tune DS RAG Embedder on GPU
4. Benchmark vs MiniLM and BGE (table + bar chart)
5. Category-level Recall@5 analysis
6. Latency profiling (ms/query)
7. Hybrid BM25 + dense retrieval demo
8. Error analysis (missed queries)
9. Interactive retrieval examples
10. Save eval JSON + optional HF upload

## How to publish on Kaggle

**Automated (API):**

```bash
pip install kaggle
# Place token at ~/.kaggle/kaggle.json (Kaggle → Settings → API)
chmod +x scripts/publish_kaggle.sh
./scripts/publish_kaggle.sh
```

Default kernel URL: https://www.kaggle.com/code/waghelad/ds-rag-embedder-v1-train-benchmark

Override slug: `KAGGLE_KERNEL_ID=youruser/your-slug ./scripts/publish_kaggle.sh`

**Manual:**

1. Go to [kaggle.com/code](https://www.kaggle.com/code) and create a new notebook
2. Upload the notebook or paste cells
3. Enable **GPU** accelerator (Settings)
4. Title: **DS RAG Embedder v1: Train, Benchmark, and Deploy Domain Embeddings**
5. Tags: `rag`, `nlp`, `embeddings`, `data-science`, `huggingface`, `retrieval`

## Recommended Kaggle Dataset

Publish a snapshot of this repo as a Kaggle Dataset so users can skip manual upload:

- Dataset name: `ds-rag-embedder-v1`
- Include: `ds_rag_embedder/`, `scripts/`, `data/`, `pyproject.toml`

## Link back to Hugging Face

Include in the notebook conclusion:

- Model: https://huggingface.co/waghelad/ds-rag-embedder-v1
- Dataset: https://huggingface.co/datasets/waghelad/ds-rag-eval-v1
- GitHub: https://github.com/dgvj-work/ds-rag-embedder-v1

Cross-linking Kaggle, GitHub, and HF improves discovery and download counts.
