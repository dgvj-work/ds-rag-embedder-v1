# Benchmark Methodology

## Dataset

**DS RAG Eval v1** contains three splits:

| Split | Purpose |
|-------|---------|
| `corpus` | Document passages indexed for retrieval |
| `eval_pairs` | Query-positive pairs for fine-tuning |
| `benchmark` | Held-out style queries with relevance labels |

Curated benchmark queries (87) are anchored to high-quality passages covering metrics, leakage, CV, MLOps, RAG, deep learning, and deployment.

## Metrics

| Metric | Definition |
|--------|------------|
| Recall@k | Share of queries where a relevant doc appears in top-k |
| MRR | Mean reciprocal rank of first relevant document |
| nDCG@10 | Ranking quality with binary relevance in top 10 |

## Reproduce

```bash
python scripts/train.py
python scripts/benchmark_report.py --model models/ds-rag-embedder-v1
open outputs/benchmark_report.html
```

## Category breakdown

```python
from ds_rag_embedder.evaluate import evaluate_by_category
from ds_rag_embedder import DSRAGEmbedder

report = evaluate_by_category(DSRAGEmbedder("models/ds-rag-embedder-v1"))
print(report.by_category.keys())
```

## Limitations

- Benchmark is English and DS/ML focused
- Synthetic expansion augments corpus volume; validate on your own docs
- Report approximate baseline numbers in README until you run `benchmark_report.py` locally
