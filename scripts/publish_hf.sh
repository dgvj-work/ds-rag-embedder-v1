#!/usr/bin/env bash
# Publish DS RAG Embedder model + eval dataset + optional Space to Hugging Face Hub
#
# Prerequisites:
#   pip install -U "huggingface_hub[cli]" sentence-transformers datasets
#   hf auth login
#
# Usage:
#   ./scripts/publish_hf.sh
#   ./scripts/publish_hf.sh waghelad/ds-rag-embedder-v1 waghelad/ds-rag-eval-v1
set -euo pipefail

MODEL_ID="${1:-waghelad/ds-rag-embedder-v1}"
DATASET_ID="${2:-waghelad/ds-rag-eval-v1}"
SPACE_ID="${3:-waghelad/ds-rag-embedder-demo}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "Publishing DS RAG Embedder to Hugging Face"
echo "  Model:   $MODEL_ID"
echo "  Dataset: $DATASET_ID"
echo "  Space:   $SPACE_ID"
echo ""

if ! command -v hf >/dev/null 2>&1; then
  echo "ERROR: install HF CLI: pip install -U 'huggingface_hub[cli]'"
  exit 1
fi

if ! hf auth whoami >/dev/null 2>&1; then
  echo "ERROR: run: hf auth login"
  exit 1
fi

echo "Step 1/5 — Build corpus if missing…"
python scripts/build_corpus.py --corpus-size 600

echo "Step 2/5 — Train model (skip if already trained)…"
if [[ ! -d "models/ds-rag-embedder-v1" ]]; then
  python scripts/train.py --epochs 4 --batch-size 32
else
  echo "  Using existing models/ds-rag-embedder-v1"
fi

echo "Step 3/5 — Evaluate and generate report…"
python scripts/benchmark_report.py --model models/ds-rag-embedder-v1 || python scripts/evaluate.py --compare || true

echo "Step 4/5 — Export + upload model…"
python scripts/export_hf.py
hf upload "$MODEL_ID" exports/hf-model . --repo-type=model

echo "Step 5/5 — Publish eval dataset…"
python scripts/publish_dataset.py --repo "$DATASET_ID"

echo ""
echo "Optional Space upload (Gradio demo):"
echo "  hf upload $SPACE_ID . --repo-type=space"
echo ""
echo "Done."
echo "  Model:   https://huggingface.co/$MODEL_ID"
echo "  Dataset: https://huggingface.co/datasets/$DATASET_ID"
