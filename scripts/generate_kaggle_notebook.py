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


cells = [
    md(
        """# DS RAG Embedder v1: Train, Benchmark, and Deploy

**Domain-specific embeddings for Data Science and ML documentation retrieval**

| Resource | Link |
|----------|------|
| GitHub | [dgvj-work/ds-rag-embedder-v1](https://github.com/dgvj-work/ds-rag-embedder-v1) |
| HF Model | [waghelad/ds-rag-embedder-v1](https://huggingface.co/waghelad/ds-rag-embedder-v1) |
| HF Dataset | [waghelad/ds-rag-eval-v1](https://huggingface.co/datasets/waghelad/ds-rag-eval-v1) |

> **Settings:** enable **GPU** (T4 or better) for training.

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
10. Export and push to Hugging Face
"""
    ),
    code(
        """!pip install -q sentence-transformers datasets huggingface_hub scikit-learn pandas matplotlib seaborn tqdm
"""
    ),
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
    md("## 3. Fine-tune DS RAG Embedder"),
    code(
        """from ds_rag_embedder.train import train
from ds_rag_embedder.config import EmbedderConfig

cfg = EmbedderConfig(
    epochs=4,
    batch_size=32,
    output_dir=Path('models/ds-rag-embedder-v1'),
)
model_path = train(config=cfg)
model_path
"""
    ),
    md("## 4. Benchmark: DS embedder vs generic baselines"),
    code(
        """from ds_rag_embedder.evaluate import compare_models

results = compare_models({
    'all-MiniLM-L6-v2': 'sentence-transformers/all-MiniLM-L6-v2',
    'bge-small-en-v1.5': 'BAAI/bge-small-en-v1.5',
    'ds-rag-embedder-v1': 'models/ds-rag-embedder-v1',
}, include_categories=False)

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
        """## Next steps

1. Pin the HF model + dataset + Space on your profile
2. Publish this notebook to Kaggle with tags: `rag`, `embeddings`, `nlp`, `data-science`
3. Link GitHub, HF, and Kaggle in a single launch post

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
