#!/usr/bin/env bash
# Publish notebook to Kaggle Kernels
#
# Prerequisites:
#   pip install kaggle
#   Place API token at ~/.kaggle/kaggle.json (Kaggle → Settings → API → Create New Token)
#
# Usage:
#   ./scripts/publish_kaggle.sh
#   KAGGLE_KERNEL_ID=youruser/ds-rag-embedder-v1-train-benchmark ./scripts/publish_kaggle.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NOTEBOOKS="$ROOT/notebooks"
KERNEL_ID="${KAGGLE_KERNEL_ID:-waghelad/ds-rag-embedder-v1-train-benchmark}"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "ERROR: pip install kaggle"
  exit 1
fi

if [[ ! -f "$HOME/.kaggle/kaggle.json" ]]; then
  echo "ERROR: missing ~/.kaggle/kaggle.json"
  echo "  1. https://www.kaggle.com/settings → API → Create New Token"
  echo "  2. mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json"
  echo "  3. chmod 600 ~/.kaggle/kaggle.json"
  exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cp "$NOTEBOOKS/kaggle_ds_rag_embedder.ipynb" "$TMP/"
cat > "$TMP/kernel-metadata.json" <<EOF
{
  "id": "$KERNEL_ID",
  "title": "DS RAG Embedder v1: Train, Benchmark, and Deploy Domain Embeddings",
  "code_file": "kaggle_ds_rag_embedder.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": false,
  "enable_gpu": true,
  "enable_internet": true,
  "enable_tpu": false,
  "keywords": ["rag", "nlp", "embeddings", "data-science", "machine-learning", "huggingface", "retrieval"],
  "dataset_sources": [],
  "kernel_sources": [],
  "competition_sources": []
}
EOF

echo "Pushing Kaggle kernel → $KERNEL_ID"
kaggle kernels push -p "$TMP"

echo ""
echo "Done: https://www.kaggle.com/code/${KERNEL_ID}"
