"""Command-line interface for DS RAG Embedder."""

from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ds-rag-embed",
        description="DS RAG Embedder — encode, search, train, evaluate, publish",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    enc = sub.add_parser("encode", help="Encode queries or documents")
    enc.add_argument("texts", nargs="+", help="Texts to encode")
    enc.add_argument("--query", action="store_true", help="Treat as queries (adds retrieval prefix)")
    enc.add_argument("--model", default=None, help="Model path or HF repo id")

    sr = sub.add_parser("search", help="Search documents for a query")
    sr.add_argument("query", help="Search query")
    sr.add_argument("--docs", nargs="+", required=True, help="Candidate documents")
    sr.add_argument("--top-k", type=int, default=3)
    sr.add_argument("--model", default=None)

    sub.add_parser("train", help="Fine-tune embedder on DS/ML corpus")
    sub.add_parser("eval", help="Run retrieval benchmark")
    sub.add_parser("build-corpus", help="Generate training corpus and eval sets")

    pub = sub.add_parser("publish", help="Push model to Hugging Face Hub")
    pub.add_argument("--repo", default="waghelad/ds-rag-embedder-v1")
    pub.add_argument("--private", action="store_true")

    args = parser.parse_args()

    if args.command == "build-corpus":
        from scripts.build_corpus import main as build_main

        build_main()
        return

    if args.command == "train":
        from ds_rag_embedder.train import train

        out = train()
        print(f"Training complete → {out}")
        return

    if args.command == "eval":
        from ds_rag_embedder.evaluate import evaluate_retrieval
        from ds_rag_embedder.model import DSRAGEmbedder

        embedder = DSRAGEmbedder(model_name_or_path=getattr(args, "model", None))
        metrics = evaluate_retrieval(embedder)
        print(json.dumps(metrics.to_dict(), indent=2))
        return

    if args.command == "publish":
        from ds_rag_embedder.model import DSRAGEmbedder

        embedder = DSRAGEmbedder(model_name_or_path="models/ds-rag-embedder-v1")
        url = embedder.push_to_hub(repo_id=args.repo, private=args.private)
        print(f"Published → {url}")
        return

    from ds_rag_embedder.model import DSRAGEmbedder

    embedder = DSRAGEmbedder(model_name_or_path=getattr(args, "model", None))

    if args.command == "encode":
        fn = embedder.encode_queries if args.query else embedder.encode_documents
        vecs = fn(args.texts)
        for text, vec in zip(args.texts, vecs):
            preview = vec[:8].tolist()
            print(json.dumps({"text": text, "dim": len(vec), "preview": preview}))
        return

    if args.command == "search":
        results = embedder.search(args.query, args.docs, top_k=args.top_k)
        print(json.dumps(results, indent=2))
        return

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
