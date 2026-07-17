#!/usr/bin/env python3
"""Export trained model to Hugging Face Hub format with model card."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ds_rag_embedder.config import DEFAULT_CONFIG
from ds_rag_embedder.model import DSRAGEmbedder


def _write_model_card(dest: Path, metrics: dict | None = None) -> None:
    card_src = ROOT / "MODEL_CARD.md"
    content = card_src.read_text(encoding="utf-8")
    if metrics and "## Latest evaluation" not in content:
        content += "\n\n## Latest evaluation\n\n```json\n"
        content += json.dumps(metrics, indent=2)
        content += "\n```\n"
    (dest / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model for HF Hub upload")
    parser.add_argument("--model-dir", default="models/ds-rag-embedder-v1")
    parser.add_argument("--export-dir", default="exports/hf-model")
    parser.add_argument("--eval-results", default="outputs/eval_results.json")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    export_dir = Path(args.export_dir)
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)

    embedder = DSRAGEmbedder(model_name_or_path=str(model_dir))
    embedder.save_local(export_dir)

    metrics = None
    eval_path = Path(args.eval_results)
    if eval_path.exists():
        metrics = json.loads(eval_path.read_text(encoding="utf-8"))

    _write_model_card(export_dir, metrics)
    config_meta = {
        "model_type": "ds_rag_embedder",
        "hub_id": DEFAULT_CONFIG.hub_model_id,
        "sentence_transformers": True,
        "pooling_mode": "mean",
        "embedding_dimension": DEFAULT_CONFIG.embedding_dim,
        "prompts": {
            "query": DEFAULT_CONFIG.query_prefix.rstrip(),
            "document": "",
        },
        "similarity_fn_name": "cosine",
        "domain": "data-science-ml-rag",
    }
    (export_dir / "config_sentence_transformers.json").write_text(
        json.dumps(config_meta, indent=2), encoding="utf-8"
    )
    print(f"Export ready → {export_dir}")


if __name__ == "__main__":
    main()
