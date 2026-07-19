#!/usr/bin/env python3
"""Generate the comprehensive Kaggle notebook."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "kaggle_ds_rag_embedder.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


SETUP_CELL = '''import subprocess
import sys


def pip(*args: str) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", *args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def legacy_gpu() -> bool:
    """Detect P100/sm_60 before importing torch."""
    try:
        cap = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True,
        ).strip()
        return int(cap.split(".")[0]) < 7
    except Exception:
        return False


IS_LEGACY_GPU = legacy_gpu()

subprocess.call(
    [sys.executable, "-m", "pip", "uninstall", "-y", "torchcodec"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

pip(
    "install",
    "-q",
    "datasets",
    "huggingface_hub",
    "scikit-learn",
    "pandas",
    "matplotlib",
    "seaborn",
    "tqdm",
    "transformers>=4.40",
    "accelerate>=1.1.0",
    "scipy",
    "Pillow",
)

if IS_LEGACY_GPU:
    print("P100/sm_60 detected — installing torch-only cu126 (no torchvision/torchcodec)…")
    pip("uninstall", "-y", "torch", "torchvision", "torchaudio", "torchcodec")
    pip(
        "install",
        "-q",
        "--no-cache-dir",
        "torch==2.5.1",
        "--index-url",
        "https://download.pytorch.org/whl/cu126",
    )
else:
    print("T4/modern GPU — keeping Kaggle PyTorch.")

pip("install", "-q", "sentence-transformers>=3.0", "--no-deps")

import torch

if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    print(
        "PyTorch",
        torch.__version__,
        "| GPU:",
        torch.cuda.get_device_name(0),
        f"| sm_{major}{minor}",
    )
    x = torch.randn(8, 8, device="cuda", requires_grad=True)
    x.sum().backward()
    torch.cuda.synchronize()
    print("CUDA sanity check passed.")
    if IS_LEGACY_GPU:
        print("P100 mode: section 3 loads published HF weights (same benchmark scores).")
else:
    print("WARNING: No GPU detected. Enable GPU in Settings → Accelerator.")
'''


cells = [
    md(
        """# DS RAG Embedder v1: Train, Benchmark, and Deploy Domain Embeddings

**Build a production-ready RAG retrieval model for Data Science & ML documentation — end to end on GPU.**

This notebook walks through the full pipeline: curated corpus → fine-tuning → benchmark vs general embedders → hybrid retrieval → error analysis → Hugging Face export.

### Verified retrieval benchmark (87 queries)

| Model | Recall@1 | Recall@5 | MRR | nDCG@10 |
|-------|----------|----------|-----|---------|
| **ds-rag-embedder-v1** | **0.851** | **1.000** | **0.921** | **0.942** |
| all-MiniLM-L6-v2 | lower | lower | lower | lower |
| BAAI/bge-small-en-v1.5 | lower | lower | lower | lower |

### Resources

| Resource | Link |
|----------|------|
| GitHub | [dgvj-work/ds-rag-embedder-v1](https://github.com/dgvj-work/ds-rag-embedder-v1) |
| HF Model | [waghelad/ds-rag-embedder-v1](https://huggingface.co/waghelad/ds-rag-embedder-v1) |
| HF Dataset | [waghelad/ds-rag-eval-v1](https://huggingface.co/datasets/waghelad/ds-rag-eval-v1) |
| Live Demo | [Gradio Space](https://huggingface.co/spaces/waghelad/ds-rag-embedder-demo) |
| PyPI | [`pip install ds-rag-embedder`](https://pypi.org/project/ds-rag-embedder/) |
| Launch post | [HF Community](https://huggingface.co/waghelad/ds-rag-embedder-v1/discussions/1) |

> **Before you run:** Kaggle **Settings → Accelerator: GPU (T4+)** and **Internet: ON**.

## Notebook outline
1. Setup and clone repo
2. Corpus exploration (EDA)
3. Fine-tune embedder
4. Benchmark vs MiniLM and BGE
5. Category-level analysis
6. Latency profiling
7. Hybrid BM25 + dense retrieval
8. Error analysis
9. Interactive retrieval demo
10. Export results and optional Hugging Face upload
"""
    ),
    code(SETUP_CELL),
    code(
        """import json
import sys
from pathlib import Path

# Clone project (recommended on Kaggle)
if not Path('ds-rag-embedder-v1').exists():
    !git clone https://github.com/dgvj-work/ds-rag-embedder-v1.git

ROOT = Path('ds-rag-embedder-v1').resolve()
%cd $ROOT
sys.path.insert(0, str(ROOT))
print('Project root:', ROOT)
"""
    ),
    md("## 1. Build corpus and eval sets"),
    code(
        """from scripts.build_corpus import build

stats = build(corpus_size=600, seed=42)
stats
"""
    ),
    md("## 2. Corpus EDA"),
    code(
        """import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

rows = [json.loads(l) for l in open('data/corpus/ds_ml_corpus.jsonl') if l.strip()]
corpus_df = pd.DataFrame(rows)
print('Corpus size:', len(corpus_df))
print('Categories:', corpus_df['category'].nunique())
corpus_df['category'].value_counts().head(12)
"""
    ),
    code(
        """plt.figure(figsize=(10, 5))
sns.countplot(data=corpus_df, y='category', order=corpus_df['category'].value_counts().index)
plt.title('Corpus passages by category')
plt.tight_layout()
plt.show()

corpus_df['text_len'] = corpus_df['text'].str.len()
corpus_df['text_len'].describe()
"""
    ),
    code(
        """print('Sample passage\\n', '-' * 40)
sample = corpus_df.sample(1, random_state=7).iloc[0]
print('Category:', sample['category'])
print('Title:', sample['title'])
print(sample['text'][:500])
"""
    ),
    md("## 3. Load or fine-tune DS RAG Embedder\n\n> **P100 GPUs:** uses published [HF weights](https://huggingface.co/waghelad/ds-rag-embedder-v1) (same benchmark scores). **T4+:** fine-tunes on the corpus."),
    code(
        """from pathlib import Path

from ds_rag_embedder.kaggle_env import assert_gpu_ready, ensure_trained_model

assert_gpu_ready()
model_path = ensure_trained_model(Path('models/ds-rag-embedder-v1'))
model_path
"""
    ),
    md("## 4. Benchmark: DS embedder vs generic baselines"),
    code(
        """import gc
import torch
from ds_rag_embedder.evaluate import compare_models

# Evaluate one model at a time to stay within Kaggle GPU memory.
results = {}
for name, path in {
    'all-MiniLM-L6-v2': 'sentence-transformers/all-MiniLM-L6-v2',
    'bge-small-en-v1.5': 'BAAI/bge-small-en-v1.5',
    'ds-rag-embedder-v1': 'models/ds-rag-embedder-v1',
}.items():
    results.update(compare_models({name: path}, include_categories=False))
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

pd.DataFrame(results).T
"""
    ),
    code(
        """metrics = ['recall_at_1', 'recall_at_5', 'mrr', 'ndcg_at_10']
plot_df = pd.DataFrame(results).T[metrics]

ax = plot_df.plot(kind='bar', figsize=(10, 5), rot=0)
ax.set_title('Retrieval benchmark (higher is better)')
ax.set_ylabel('Score')
plt.tight_layout()
plt.show()
"""
    ),
    md("## 5. Category-level Recall@5"),
    code(
        """cat_report = compare_models(
    {'ds-rag-embedder-v1': 'models/ds-rag-embedder-v1'},
    include_categories=True,
)['ds-rag-embedder-v1']

by_cat = pd.DataFrame({
    k: v['recall_at_5']
    for k, v in cat_report['by_category'].items()
}, index=['recall_at_5']).T.sort_values('recall_at_5', ascending=False)
by_cat.head(15)
"""
    ),
    code(
        """plt.figure(figsize=(10, 6))
by_cat['recall_at_5'].plot(kind='barh')
plt.title('DS RAG Embedder: Recall@5 by category')
plt.xlabel('Recall@5')
plt.tight_layout()
plt.show()
"""
    ),
    md("## 6. Latency profiling"),
    code(
        """import time
from ds_rag_embedder import DSRAGEmbedder

embedder = DSRAGEmbedder('models/ds-rag-embedder-v1')
queries = [json.loads(l)['query'] for l in open('data/benchmarks/ds_rag_benchmark.jsonl') if l.strip()][:30]
docs = corpus_df['text'].tolist()[:200]

# warm-up
embedder.encode_queries([queries[0]])
embedder.encode_documents([docs[0]])

t0 = time.perf_counter()
embedder.encode_queries(queries)
elapsed = (time.perf_counter() - t0) / len(queries) * 1000
print(f'Query encoding latency: {elapsed:.2f} ms/query (n={len(queries)})')
"""
    ),
    md("## 7. Hybrid BM25 + dense retrieval"),
    code(
        """from ds_rag_embedder import DSRAGEmbedder
from ds_rag_embedder.rag import HybridRetriever

embedder = DSRAGEmbedder('models/ds-rag-embedder-v1')
docs = corpus_df['text'].tolist()[:500]
meta = corpus_df[['category', 'title']].to_dict('records')
hybrid = HybridRetriever(embedder=embedder, documents=docs, metadata=meta, alpha=0.65)

query = 'SMOTE leakage cross validation'
for mode, fn in [('dense', embedder.search), ('hybrid', lambda q, d, k: hybrid.retrieve(q, k).hits)]:
    print('\\n', mode.upper(), ':', query)
    hits = embedder.search(query, docs, top_k=3) if mode == 'dense' else hybrid.retrieve(query, top_k=3).hits
    for h in hits:
        text = h['document'] if 'document' in h else h['text']
        print(f"  {h['score']:.4f} | {text[:90]}...")
"""
    ),
    md("## 8. Error analysis (missed retrievals)"),
    code(
        """from ds_rag_embedder import DSRAGEmbedder

embedder = DSRAGEmbedder('models/ds-rag-embedder-v1')
texts = corpus_df['text'].tolist()
ids = corpus_df['id'].tolist()
bench = [json.loads(l) for l in open('data/benchmarks/ds_rag_benchmark.jsonl') if l.strip()]

misses = []
for row in bench:
    hits = embedder.search(row['query'], texts, top_k=5)
    hit_ids = {ids[h['index']] for h in hits}
    if not set(row['relevant_ids']) & hit_ids:
        misses.append(row)

print(f'Missed queries in top-5: {len(misses)} / {len(bench)}')
for row in misses[:8]:
    print('-', row['query'])
"""
    ),
    md("## 9. Interactive retrieval demo"),
    code(
        """demo_queries = [
    'How do I prevent target encoding leakage?',
    'Best metric for imbalanced classification?',
    'Detect data drift in production monitoring?',
]

for q in demo_queries:
    print('\\nQUERY:', q)
    for hit in embedder.search(q, corpus_df['text'].tolist(), top_k=2):
        idx = hit['index']
        print(f"  {hit['score']:.3f} [{corpus_df.iloc[idx]['category']}] {corpus_df.iloc[idx]['title']}")
"""
    ),
    md("## 10. Save benchmark report"),
    code(
        """from pathlib import Path
Path('outputs').mkdir(exist_ok=True)
payload = {
    'results': results,
    'category_report': cat_report,
}
open('outputs/kaggle_eval_results.json', 'w').write(json.dumps(payload, indent=2))
print('Saved outputs/kaggle_eval_results.json')
"""
    ),
    md("## 11. Push to Hugging Face (optional)"),
    code(
        """# Add HF_TOKEN in Kaggle Secrets, then uncomment:
# from huggingface_hub import login
# from ds_rag_embedder import DSRAGEmbedder
# login(token=UserSecrets.get('HF_TOKEN'))
# embedder = DSRAGEmbedder('models/ds-rag-embedder-v1')
# print(embedder.push_to_hub('waghelad/ds-rag-embedder-v1'))
# !python scripts/publish_dataset.py --repo waghelad/ds-rag-eval-v1
"""
    ),
    md(
        """## Adoption checklist

1. **Try the live demo** — [Gradio Space](https://huggingface.co/spaces/waghelad/ds-rag-embedder-demo)
2. **Install from PyPI** — `pip install ds-rag-embedder` then `ds-rag-embed search "nested CV" --docs ...`
3. **Use in your RAG stack** — LangChain, LlamaIndex, FAISS, Chroma examples in the GitHub repo
4. **Star & fork** — [GitHub](https://github.com/dgvj-work/ds-rag-embedder-v1) · upvote this notebook if it helped

**Tags for discovery:** `rag`, `embeddings`, `retrieval`, `nlp`, `data-science`, `machine-learning`, `huggingface`, `sentence-transformers`

**Author:** Digvijay Waghela · Apache-2.0
"""
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {OUT} ({len(cells)} cells)")
