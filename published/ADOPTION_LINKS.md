# Adoption publish links

Track public distribution URLs. Updated after publish runs.

| Channel | URL | Status |
|---------|-----|--------|
| HF Community launch | https://huggingface.co/waghelad/ds-rag-embedder-v1/discussions/1 | **Live** |
| HF Model | https://huggingface.co/waghelad/ds-rag-embedder-v1 | Live |
| HF Dataset | https://huggingface.co/datasets/waghelad/ds-rag-eval-v1 | Live |
| HF Space | https://huggingface.co/spaces/waghelad/ds-rag-embedder-demo | Live |
| GitHub | https://github.com/dgvj-work/ds-rag-embedder-v1 | Live |
| PyPI | https://pypi.org/project/ds-rag-embedder/1.0.0/ | **Live** (v1.0.0) |
| Kaggle notebook | https://www.kaggle.com/code/waghelad/ds-rag-embedder-v1-train-benchmark | **Needs API token** (see below) |

## Publish commands

```bash
# HF Community (done — re-run to update discussion body)
python scripts/publish_hf_community.py

# Kaggle
pip install kaggle
# Kaggle → Settings → API → Create New Token → ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
./scripts/publish_kaggle.sh

# PyPI
pip install build twine
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-...   # PyPI account → API tokens
./scripts/publish_pypi.sh
```

All three: `./scripts/publish_adoption.sh all`
