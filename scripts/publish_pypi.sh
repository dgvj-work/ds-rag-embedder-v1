#!/usr/bin/env bash
# Build and upload ds-rag-embedder to PyPI
#
# Prerequisites:
#   pip install build twine
#   export TWINE_USERNAME=__token__
#   export TWINE_PASSWORD=pypi-...   # PyPI → Account settings → API tokens
#
# Usage:
#   ./scripts/publish_pypi.sh
#   ./scripts/publish_pypi.sh --test   # upload to TestPyPI first
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Optional local credentials file (gitignored): TWINE_USERNAME= / TWINE_PASSWORD=
if [[ -f "$ROOT/.pypi_env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.pypi_env"
  set +a
fi

TEST=0
if [[ "${1:-}" == "--test" ]]; then
  TEST=1
fi

if ! command -v twine >/dev/null 2>&1; then
  echo "ERROR: pip install twine build"
  exit 1
fi

if [[ -z "${TWINE_PASSWORD:-}" ]] && [[ ! -f "$HOME/.pypirc" ]]; then
  echo "ERROR: PyPI credentials not found."
  echo "  Option A — create $ROOT/.pypi_env with:"
  echo "    TWINE_USERNAME=__token__"
  echo "    TWINE_PASSWORD=pypi-..."
  echo "  Option B — create ~/.pypirc (see https://packaging.python.org/en/latest/specifications/pypirc/)"
  exit 1
fi

export TWINE_USERNAME="${TWINE_USERNAME:-__token__}"

echo "Building sdist + wheel…"
python -m build

twine check dist/*

if [[ "$TEST" == "1" ]]; then
  echo "Uploading to TestPyPI…"
  twine upload --repository testpypi dist/*
  echo "Install test: pip install -i https://test.pypi.org/simple/ ds-rag-embedder"
else
  echo "Uploading to PyPI…"
  twine upload dist/*
  echo "Install: pip install ds-rag-embedder"
fi

echo "Done: https://pypi.org/project/ds-rag-embedder/"
