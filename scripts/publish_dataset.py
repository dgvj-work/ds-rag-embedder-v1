#!/usr/bin/env python3
"""Publish DS RAG eval corpus + benchmark to Hugging Face Hub."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus" / "ds_ml_corpus.jsonl"
EVAL = ROOT / "data" / "eval" / "ds_retrieval_eval.jsonl"
BENCH = ROOT / "data" / "benchmarks" / "ds_rag_benchmark.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="waghelad/ds-rag-eval-v1")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    if not CORPUS.exists():
        from scripts.build_corpus import build

        build()

    try:
        from datasets import Dataset
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("pip install datasets huggingface_hub") from exc

    corpus = _load_jsonl(CORPUS)
    eval_rows = _load_jsonl(EVAL)
    bench = _load_jsonl(BENCH)

    api = HfApi()
    try:
        api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    except Exception:
        pass

    # Each split has a different schema; publish as separate dataset configs.
    corpus_ds = Dataset.from_list(
        [
            {
                "id": r["id"],
                "text": r["text"],
                "category": r.get("category") or "",
                "title": r.get("title") or "",
            }
            for r in corpus
        ]
    )
    eval_ds = Dataset.from_list(eval_rows)
    bench_ds = Dataset.from_list(
        [
            {
                "query": r["query"],
                "relevant_ids": r["relevant_ids"],
                "category": r.get("category") or "",
            }
            for r in bench
        ]
    )

    print(f"Uploading corpus split ({len(corpus_ds)} rows)…")
    corpus_ds.push_to_hub(args.repo, config_name="corpus", private=args.private)
    print(f"Uploading eval_pairs split ({len(eval_ds)} rows)…")
    eval_ds.push_to_hub(args.repo, config_name="eval_pairs", private=args.private)
    print(f"Uploading benchmark split ({len(bench_ds)} rows)…")
    bench_ds.push_to_hub(args.repo, config_name="benchmark", private=args.private)

    # Also upload raw JSONL for users who prefer files over Arrow.
    for path, dest in [
        (CORPUS, "data/ds_ml_corpus.jsonl"),
        (EVAL, "data/ds_retrieval_eval.jsonl"),
        (BENCH, "data/ds_rag_benchmark.jsonl"),
    ]:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=dest,
            repo_id=args.repo,
            repo_type="dataset",
        )

    readme = f"""---
license: apache-2.0
task_categories:
  - sentence-similarity
  - feature-extraction
language:
  - en
tags:
  - rag
  - retrieval
  - data-science
  - machine-learning
  - embeddings
  - evaluation
  - benchmark
size_categories:
  - n<1K
pretty_name: DS RAG Eval v1
configs:
  - config_name: corpus
    data_files:
      - split: corpus
        path: data/ds_ml_corpus.jsonl
  - config_name: eval_pairs
    data_files:
      - split: eval_pairs
        path: data/ds_retrieval_eval.jsonl
  - config_name: benchmark
    data_files:
      - split: benchmark
        path: data/ds_rag_benchmark.jsonl
---

# DS RAG Eval v1

Evaluation dataset for **Data Science & ML documentation retrieval**, companion to
[`waghelad/ds-rag-embedder-v1`](https://huggingface.co/waghelad/ds-rag-embedder-v1).

## Splits (configs)

| Config | Rows | Purpose |
|--------|------|---------|
| `corpus` | {len(corpus)} | Document passages for indexing |
| `eval_pairs` | {len(eval_rows)} | Query to positive passage pairs for training/eval |
| `benchmark` | {len(bench)} | Retrieval benchmark with relevance labels |

## Usage

```python
from datasets import load_dataset

corpus = load_dataset("{args.repo}", "corpus", split="corpus")
eval_pairs = load_dataset("{args.repo}", "eval_pairs", split="eval_pairs")
benchmark = load_dataset("{args.repo}", "benchmark", split="benchmark")
print(benchmark[0])
```

Raw JSONL files are also available under `data/`.

## Citation

Digvijay Waghela (2026). DS RAG Eval v1. Hugging Face.
"""
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="dataset",
    )
    print(f"Published https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
