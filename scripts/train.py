#!/usr/bin/env python3
"""Fine-tune DS RAG embedder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ds_rag_embedder.config import EmbedderConfig
from ds_rag_embedder.train import train
from scripts.build_corpus import build


def main() -> None:
    parser = argparse.ArgumentParser(description="Train waghelad/ds-rag-embedder-v1")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--base-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--output", default="models/ds-rag-embedder-v1")
    parser.add_argument("--corpus-size", type=int, default=600)
    parser.add_argument("--skip-corpus", action="store_true")
    args = parser.parse_args()

    if not args.skip_corpus:
        print("Building corpus…")
        build(corpus_size=args.corpus_size)

    cfg = EmbedderConfig(
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=Path(args.output),
    )
    out = train(config=cfg, output_dir=Path(args.output))
    print(f"Model saved → {out}")


if __name__ == "__main__":
    main()
