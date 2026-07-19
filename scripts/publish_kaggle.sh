#!/usr/bin/env bash
# Publish notebook to Kaggle Kernels
#
# Prerequisites:
#   pip install kaggle
#   export KAGGLE_API_TOKEN=KGAT_...   # Kaggle → Settings → API
#   # Legacy: export KAGGLE_USERNAME=... KAGGLE_KEY=...
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

if [[ -z "${KAGGLE_API_TOKEN:-}" ]]; then
  if [[ -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ]]; then
    if [[ ! -f "$HOME/.kaggle/access_token" && ! -f "$HOME/.kaggle/kaggle.json" ]]; then
      echo "ERROR: Kaggle credentials not found."
      echo "  Option A — export KAGGLE_API_TOKEN=KGAT_..."
      echo "  Option B — export KAGGLE_USERNAME and KAGGLE_KEY (legacy)"
      echo "  Option C — ~/.kaggle/access_token or ~/.kaggle/kaggle.json"
      exit 1
    fi
  fi
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cp "$NOTEBOOKS/kaggle_ds_rag_embedder.ipynb" "$TMP/"
cat > "$TMP/kernel-metadata.json" <<EOF
{
  "id": "$KERNEL_ID",
  "title": "DS RAG Embedder v1 Train Benchmark",
  "code_file": "kaggle_ds_rag_embedder.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": false,
  "enable_gpu": true,
  "enable_internet": true,
  "enable_tpu": false,
  "keywords": ["nlp"],
  "dataset_sources": [],
  "kernel_sources": [],
  "competition_sources": []
}
EOF

echo "Pushing Kaggle kernel → $KERNEL_ID"
kaggle kernels push -p "$TMP"

echo ""
echo "Done: https://www.kaggle.com/code/${KERNEL_ID}"
