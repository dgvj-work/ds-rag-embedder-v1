#!/usr/bin/env python3
"""Publish Hugging Face model repo launch discussion (Community tab)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAUNCH_MD = ROOT / "docs" / "LAUNCH_POST.md"


def _body() -> str:
    if not LAUNCH_MD.exists():
        raise FileNotFoundError(LAUNCH_MD)
    text = LAUNCH_MD.read_text(encoding="utf-8")
    marker = "## TL;DR"
    if marker in text:
        return text[text.index(marker) :].strip()
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create HF model launch discussion")
    parser.add_argument("--repo", default="waghelad/ds-rag-embedder-v1")
    parser.add_argument(
        "--title",
        default="Launch: DS RAG Embedder v1 — domain embeddings for DS/ML documentation RAG",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    body = _body()
    if args.dry_run:
        print(f"Title: {args.title}\n\n{body[:500]}...")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    discussion = api.create_discussion(
        repo_id=args.repo,
        title=args.title,
        description=body,
        repo_type="model",
        pull_request=False,
    )
    url = getattr(discussion, "url", None) or discussion
    print(f"Created discussion: {url}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
