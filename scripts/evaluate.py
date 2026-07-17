#!/usr/bin/env python3
"""Evaluate retrieval quality and compare against baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ds_rag_embedder.evaluate import compare_models, evaluate_retrieval
from ds_rag_embedder.model import DSRAGEmbedder


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DS RAG embedder")
    parser.add_argument("--model", default="models/ds-rag-embedder-v1")
    parser.add_argument("--compare", action="store_true", help="Compare with generic baselines")
    parser.add_argument("--categories", action="store_true")
    parser.add_argument("--output", default="outputs/eval_results.json")
    args = parser.parse_args()

    if args.compare:
        models = {
            "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
            "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
        }
        if Path(args.model).exists():
            models["ds-rag-embedder-v1"] = args.model
        results = compare_models(models, include_categories=args.categories)
    else:
        embedder = DSRAGEmbedder(
            model_name_or_path=str(args.model)
            if Path(args.model).exists()
            else "BAAI/bge-small-en-v1.5"
        )
        if args.categories:
            from ds_rag_embedder.evaluate import evaluate_by_category

            results = evaluate_by_category(embedder).to_dict()
        else:
            results = evaluate_retrieval(embedder, measure_latency=True).to_dict()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
