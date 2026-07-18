#!/usr/bin/env bash
# Publish adoption assets: PyPI, Kaggle kernel, HF Community launch post
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== DS RAG Embedder — Adoption publish ==="
echo ""

STEP="${1:-all}"

run_pypi() {
  chmod +x scripts/publish_pypi.sh
  ./scripts/publish_pypi.sh
}

run_kaggle() {
  chmod +x scripts/publish_kaggle.sh
  ./scripts/publish_kaggle.sh
}

run_hf_community() {
  python scripts/publish_hf_community.py
}

case "$STEP" in
  pypi) run_pypi ;;
  kaggle) run_kaggle ;;
  community|hf) run_hf_community ;;
  all)
    run_hf_community || echo "WARN: HF community post skipped/failed"
    run_kaggle || echo "WARN: Kaggle publish skipped/failed"
    run_pypi || echo "WARN: PyPI publish skipped/failed"
    ;;
  *)
    echo "Usage: $0 [pypi|kaggle|community|all]"
    exit 1
    ;;
esac

echo ""
echo "Update published/ADOPTION_LINKS.md with live URLs after successful uploads."
