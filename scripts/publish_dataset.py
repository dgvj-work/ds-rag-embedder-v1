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
        from datasets import Dataset, DatasetDict
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("pip install datasets huggingface_hub") from exc

    corpus = _load_jsonl(CORPUS)
    eval_rows = _load_jsonl(EVAL)
    bench = _load_jsonl(BENCH)

    ds = DatasetDict(
        {
            "corpus": Dataset.from_list(
                [{"id": r["id"], "text": r["text"], "category": r.get("category"), "title": r.get("title")} for r in corpus]
            ),
            "eval_pairs": Dataset.from_list(eval_rows),
            "benchmark": Dataset.from_list(bench),
        }
    )
    ds.push_to_hub(args.repo, private=args.private)

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
---

# DS RAG Eval v1

Evaluation dataset for **Data Science & ML documentation retrieval** — companion to
[`waghelad/ds-rag-embedder-v1`](https://huggingface.co/waghelad/ds-rag-embedder-v1).

## Splits

| Split | Rows | Purpose |
|-------|------|---------|
| `corpus` | {len(corpus)} | Document passages for indexing |
| `eval_pairs` | {len(eval_rows)} | Query → positive passage pairs for training/eval |
| `benchmark` | {len(bench)} | Retrieval benchmark with relevance labels |

## Usage

```python
from datasets import load_dataset
ds = load_dataset("{args.repo}")
print(ds["benchmark"][0])
```

## Citation

Digvijay Waghela (2026). DS RAG Eval v1. Hugging Face.
"""
    api = HfApi()
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="dataset",
    )
    print(f"Published https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
