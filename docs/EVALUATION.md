# Evaluation Guide

## Metrics

| Metric | Meaning |
|--------|---------|
| **Recall@k** | Fraction of queries with a relevant doc in top-k |
| **MRR** | Mean reciprocal rank of first relevant doc |
| **nDCG@10** | Ranking quality with graded relevance |

## Run benchmark

```bash
python scripts/evaluate.py --model models/ds-rag-embedder-v1
python scripts/evaluate.py --compare   # vs MiniLM + BGE base
```

Results saved to `outputs/eval_results.json`.

## Benchmark data

- `data/benchmarks/ds_rag_benchmark.jsonl` — curated DS queries with labeled relevant passage IDs
- `data/eval/ds_retrieval_eval.jsonl` — training/eval query-passage pairs

## Interpret results

Improvements on DS-specific queries indicate the model is suitable for RAG over ML docs. Always validate on **your** internal corpus before production.
