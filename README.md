# DS RAG Embedder v1

[![GitHub](https://img.shields.io/badge/GitHub-dgvj--work%2Fds--rag--embedder--v1-blue)](https://github.com/dgvj-work/ds-rag-embedder-v1)
[![Model](https://img.shields.io/badge/🤗%20Model-ds--rag--embedder--v1-blue)](https://huggingface.co/waghelad/ds-rag-embedder-v1)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-ds--rag--eval--v1-blue)](https://huggingface.co/datasets/waghelad/ds-rag-eval-v1)
[![Space](https://img.shields.io/badge/🤗%20Space-ds--rag--embedder--demo-blue)](https://huggingface.co/spaces/waghelad/ds-rag-embedder-demo)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue)](https://github.com/dgvj-work/ds-rag-embedder-v1/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)

**Domain-specific embedding model for RAG over Data Science & ML documentation.**

Fine-tuned from [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) to improve retrieval for the questions practitioners actually ask: class imbalance, data leakage, cross-validation, drift monitoring, experiment tracking, RAG evaluation, feature engineering, and MLOps runbooks.

Built for daily use in **LangChain**, **LlamaIndex**, **Chroma**, **FAISS**, and Hugging Face TEI pipelines.

**Published on Hugging Face:** [Model](https://huggingface.co/waghelad/ds-rag-embedder-v1) · [Dataset](https://huggingface.co/datasets/waghelad/ds-rag-eval-v1) · [Demo Space](https://huggingface.co/spaces/waghelad/ds-rag-embedder-demo)

---

## Why this model?

Generic embedders (MiniLM, base BGE) work broadly but miss **DS/ML task intent** and terminology. This model is optimized for retrieval over:

- Notebook and experiment documentation
- Model cards and ML runbooks
- Metrics and validation guides
- MLOps / monitoring playbooks
- Internal DS knowledge bases

| Scenario | What you get |
|----------|--------------|
| RAG copilot for data teams | Better Recall@k on DS-specific queries |
| Semantic search over ML docs | Understands AUC, SMOTE, SHAP, PSI, nested CV, etc. |
| Production vector pipelines | 384-dim, L2-normalized, fast on CPU/GPU |

---

## Quick start

### Option A: Python package (recommended)

```python
from ds_rag_embedder import DSRAGEmbedder

embedder = DSRAGEmbedder("waghelad/ds-rag-embedder-v1")

documents = [
    "Target encoding on the full dataset before train/test split causes label leakage.",
    "Use nested cross-validation when tuning hyperparameters to avoid optimistic bias.",
    "Population Stability Index above 0.25 indicates significant feature drift.",
]

for hit in embedder.search("How do I prevent data leakage?", documents, top_k=3):
    print(f"{hit['score']:.4f}  {hit['document'][:80]}…")
```

### Option B: Sentence Transformers

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("waghelad/ds-rag-embedder-v1")

query = (
    "Represent this Data Science question for retrieving relevant documentation: "
    "How do I handle class imbalance?"
)
q_emb = model.encode([query], normalize_embeddings=True)
d_emb = model.encode(documents, normalize_embeddings=True)

scores = q_emb @ d_emb.T
print(scores)
```

### Option C: Full RAG pipeline (retrieve → LLM prompt)

```python
from ds_rag_embedder.rag import DSRAGPipeline

pipe = DSRAGPipeline()  # loads bundled DS/ML corpus for demo
response = pipe.retrieve("Best metric for imbalanced classification?")

print(response.contexts[0][:200])
print(response.prompt)  # pass to your LLM
```

---

## Query prefix (important)

This model uses **asymmetric retrieval** (BGE-style). For best results:

| Input | Prefix |
|-------|--------|
| **Queries** | `Represent this Data Science question for retrieving relevant documentation: ` |
| **Documents** | None: encode passage text as-is |

The `DSRAGEmbedder.encode_queries()` helper applies the prefix automatically.

---

## Installation

```bash
git clone https://github.com/dgvj-work/ds-rag-embedder-v1.git
cd ds-rag-embedder-v1

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[demo,dev]"
```

Optional integration extras:

```bash
pip install -e ".[integrations]"   # LangChain, LlamaIndex, Chroma, FAISS examples
```

Verify locally:

```bash
python scripts/build_corpus.py
chmod +x scripts/run_local.sh
./scripts/run_local.sh
```

---

## Features

- **Hybrid retrieval:** BM25 + dense for production-grade search (SMOTE, PSI, AUC tokens)
- **Benchmark report:** HTML/Markdown/JSON via `scripts/benchmark_report.py`
- **Category eval:** Per-topic Recall@k breakdown (metrics, MLOps, RAG, etc.)
- **Fine-tuned embedder:** domain adaptation on 600+ DS/ML passages (29 curated + expansion)
- **Eval benchmark:** Recall@k, MRR, nDCG on curated DS retrieval queries
- **RAG toolkit:** chunker, retriever, pipeline, LLM-ready prompts
- **Framework adapters:** LangChain, LlamaIndex wrappers
- **CLI:** `ds-rag-embed encode`, `search`, `train`, `eval`
- **Gradio Space demo:** interactive retrieval UI
- **HF publish scripts:** one-command model + dataset upload
- **Kaggle notebook:** train and evaluate on GPU

---

## Benchmark (DS RAG Eval v1)

Reproduce on your machine:

```bash
python scripts/train.py --epochs 4
python scripts/evaluate.py --compare
```

| Model | Recall@1 | Recall@5 | MRR | nDCG@10 |
|-------|----------|----------|-----|---------|
| all-MiniLM-L6-v2 | 0.621 | 0.828 | 0.708 | 0.740 |
| bge-small-en-v1.5 | 0.506 | 0.609 | 0.558 | 0.567 |
| **ds-rag-embedder-v1** | **0.851** | **1.000** | **0.921** | **0.942** |

*Verified on DS RAG Eval v1 (87 queries). Full results: [`outputs/eval_results.json`](outputs/eval_results.json) · generated 2026-07-17.*

Dataset: [`waghelad/ds-rag-eval-v1`](https://huggingface.co/datasets/waghelad/ds-rag-eval-v1) (87 benchmark queries, 658 eval pairs)

Generate a verified report after training:

```bash
python scripts/benchmark_report.py --model models/ds-rag-embedder-v1
open outputs/benchmark_report.html
```

See [`docs/BENCHMARK.md`](docs/BENCHMARK.md) for methodology.

---

## Hybrid retrieval (BM25 + dense)

Exact DS tokens (SMOTE, PSI, AUC) plus semantic paraphrases:

```python
from ds_rag_embedder.rag import HybridRetriever

hybrid = HybridRetriever(embedder=embedder, documents=documents, alpha=0.65)
hits = hybrid.retrieve("SMOTE leakage cross validation", top_k=5).hits
```

Example: [`examples/hybrid_retrieval_example.py`](examples/hybrid_retrieval_example.py)

---

## Adoption guide

Swap MiniLM/BGE in existing RAG stacks: [`docs/ADOPTION.md`](docs/ADOPTION.md)

---

## Training

```bash
# Build corpus + eval sets (600 passages, 658 eval pairs, 87 benchmark queries)
python scripts/build_corpus.py --corpus-size 600

# Fine-tune (GPU recommended)
python scripts/train.py --epochs 4 --batch-size 32

# Export for Hugging Face
python scripts/export_hf.py
```

See [`docs/TRAINING.md`](docs/TRAINING.md) for hyperparameters and cloud GPU notes.

---

## Publish to Hugging Face

```bash
pip install -U "huggingface_hub[cli]" sentence-transformers datasets
hf auth login

chmod +x scripts/publish_hf.sh
./scripts/publish_hf.sh
```

This publishes:

| Asset | Repo |
|-------|------|
| Model | `waghelad/ds-rag-embedder-v1` |
| Dataset | `waghelad/ds-rag-eval-v1` |
| Space demo | `waghelad/ds-rag-embedder-demo` |

Full guide: [`docs/HF_UPLOAD.md`](docs/HF_UPLOAD.md)

---

## Live demo (Gradio Space)

```bash
python app.py
```

Or open the [Hugging Face Space demo](https://huggingface.co/spaces/waghelad/ds-rag-embedder-demo).

Tabs: **Retrieve** · **Compare embedders** · **Quick start**

---

## Integrations

| Framework | Example |
|-----------|---------|
| LangChain | [`examples/langchain_example.py`](examples/langchain_example.py) |
| LlamaIndex | [`examples/llama_index_example.py`](examples/llama_index_example.py) |
| ChromaDB | [`examples/chromadb_example.py`](examples/chromadb_example.py) |
| FAISS | [`examples/faiss_example.py`](examples/faiss_example.py) |
| Hybrid BM25+dense | [`examples/hybrid_retrieval_example.py`](examples/hybrid_retrieval_example.py) |

Guide: [`docs/RAG_INTEGRATION.md`](docs/RAG_INTEGRATION.md)

---

## CLI

```bash
ds-rag-embed search "nested cross validation" \
  --docs "Use nested CV when tuning hyperparameters." "Accuracy is misleading when imbalanced."

ds-rag-embed train
ds-rag-embed eval
ds-rag-embed publish --repo waghelad/ds-rag-embedder-v1
```

---

## Project structure

```
ds-rag-embedder-v1/
├── ds_rag_embedder/          Core package
│   ├── model.py              DSRAGEmbedder (encode, search, push_to_hub)
│   ├── train.py              Fine-tuning pipeline
│   ├── evaluate.py           Recall@k, MRR, nDCG
│   ├── rag/                  Chunker, retriever, RAG pipeline
│   └── integrations/         LangChain, LlamaIndex adapters
├── data/                     Corpus, eval pairs, benchmark (generated)
├── scripts/                  build_corpus, train, evaluate, publish_hf
├── app.py                    Gradio Space demo
├── notebooks/                Kaggle notebook
├── docs/                     Training, eval, HF upload, Kaggle guides
├── examples/                 Framework integration scripts
└── tests/
```

---

## Model details

| Property | Value |
|----------|-------|
| Base model | `BAAI/bge-small-en-v1.5` |
| Embedding dimension | 384 |
| Max sequence length | 512 |
| Normalization | L2 (cosine similarity) |
| Training loss | MultipleNegativesRankingLoss |
| Corpus size | 600 passages (configurable) |
| Language | English |

---

## Documentation

| Guide | Description |
|-------|-------------|
| [TRAINING.md](docs/TRAINING.md) | Fine-tuning workflow & hyperparameters |
| [EVALUATION.md](docs/EVALUATION.md) | Benchmark metrics and interpretation |
| [HF_UPLOAD.md](docs/HF_UPLOAD.md) | Hugging Face model/dataset/Space upload |
| [KAGGLE.md](docs/KAGGLE.md) | Kaggle notebook publishing |
| [RAG_INTEGRATION.md](docs/RAG_INTEGRATION.md) | Production RAG integration patterns |
| [ADOPTION.md](docs/ADOPTION.md) | Swap generic embedders in 5 minutes |
| [BENCHMARK.md](docs/BENCHMARK.md) | Benchmark methodology and reproduction |
| [MODEL_CARD.md](MODEL_CARD.md) | Hugging Face model card |

---

## Kaggle

Train and evaluate on Kaggle GPU:

- Notebook: [`notebooks/kaggle_ds_rag_embedder.ipynb`](notebooks/kaggle_ds_rag_embedder.ipynb)
- Guide: [`docs/KAGGLE.md`](docs/KAGGLE.md)

---

## Intended use

**Good for:**
- RAG over DS/ML documentation and runbooks
- Semantic search in experiment tracking / knowledge bases
- Retrieval layer in data-team copilots

**Not intended for:**
- General open-web search
- Legal / medical domains without evaluation
- High-stakes automated decisions without human review

---

## Limitations

- English-only at v1
- Optimized for technical DS/ML prose (not raw code-only snippets)
- Benchmark is curated; validate on your own corpus before production
- For large scale, pair with a vector database (Chroma, Pinecone, Weaviate, etc.)

---

## Citation

```bibtex
@misc{waghela2026dsrag,
  author = {Digvijay Waghela},
  title = {DS RAG Embedder v1: Domain Embeddings for Data Science Documentation Retrieval},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/waghelad/ds-rag-embedder-v1}}
}
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `401` / model download fails | Run `hf auth logout` then retry, or `hf auth login --force` with a valid token |
| Training fails on `accelerate` | `pip install 'accelerate>=1.1.0'` (included in requirements.txt) |
| Gradio theme warning | Set `theme=` on `gr.Blocks(...)`; Space uses Gradio SDK 5.12 via `README_HF_SPACE.md` |
| `./scripts/run_local.sh` encode skip | Usually invalid HF token; logout fixes public model downloads |

---

**Digvijay Waghela** · digvijay.vaghela@yahoo.com · [GitHub](https://github.com/dgvj-work) · Apache-2.0
