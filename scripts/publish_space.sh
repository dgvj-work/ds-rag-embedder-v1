#!/usr/bin/env bash
# Publish Gradio demo Space to Hugging Face Hub (synced with GitHub app.py + README_HF_SPACE.md)
set -euo pipefail

SPACE_ID="${1:-waghelad/ds-rag-embedder-demo}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v hf >/dev/null 2>&1; then
  echo "ERROR: install HF CLI: pip install -U 'huggingface_hub[cli]'"
  exit 1
fi

if ! hf auth whoami >/dev/null 2>&1; then
  echo "ERROR: run: hf auth login"
  exit 1
fi

echo "Publishing Space → $SPACE_ID"

hf repo create "$SPACE_ID" --repo-type space --space_sdk gradio --private false --exist-ok 2>/dev/null || true

# App code, package, corpus for demo, and dependencies
hf upload "$SPACE_ID" . --repo-type=space \
  --include "app.py" \
  --include "ds_rag_embedder/**" \
  --include "data/corpus/**"

# Space requirements (no editable "." install — HF Docker build cannot pip install ".")
hf upload "$SPACE_ID" requirements-space.txt requirements.txt --repo-type=space

# Space README with YAML frontmatter
hf upload "$SPACE_ID" README_HF_SPACE.md README.md --repo-type=space

echo "Done: https://huggingface.co/spaces/$SPACE_ID"
