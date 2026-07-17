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
    parser.add_argument("--output", default="outputs/eval_results.json")
    args = parser.parse_args()

    results = {}
    model_path = Path(args.model)
    if model_path.exists():
        embedder = DSRAGEmbedder(model_name_or_path=str(model_path))
        results["ds-rag-embedder-v1"] = evaluate_retrieval(embedder).to_dict()
    else:
        print(f"Warning: {model_path} not found — evaluating base model for smoke test")
        embedder = DSRAGEmbedder(model_name_or_path="BAAI/bge-small-en-v1.5")
        results["bge-small-base"] = evaluate_retrieval(embedder).to_dict()

    if args.compare:
        baselines = compare_models(
            {
                "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
                "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
            }
        )
        if model_path.exists():
            baselines["ds-rag-embedder-v1"] = results["ds-rag-embedder-v1"]
        results = baselines

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
