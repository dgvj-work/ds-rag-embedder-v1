# Kaggle Notebook Guide

## File

`notebooks/kaggle_ds_rag_embedder.ipynb` (30 cells, comprehensive workflow)

Regenerate after edits:

```bash
python scripts/generate_kaggle_notebook.py
```

## Runtime matrix (pre-analysed)

| Kaggle GPU | PyTorch | Training in notebook | Model source |
|------------|---------|-------------------|--------------|
| **T4** (sm_75+) | Kaggle cu128 | Yes | Fine-tuned locally |
| **P100** (sm_60) | **torch==2.10.0+cu126** (auto-installed) | No | Published HF weights |
| **P100 install fail** | CPU fallback | No | Published HF weights |

Kaggle's default **PyTorch 2.10+cu128** does **not** support P100 (minimum sm_70). The notebook setup cell replaces it with **cu126** before importing torch. If that install fails, it sets `KAGGLE_DEVICE=cpu` so the notebook still completes.

## Sections covered

1. Setup (GPU/PyTorch compatibility)
2. Clone from GitHub + corpus build + EDA
3. Load or fine-tune embedder
4. Benchmark vs MiniLM and BGE
5. Category-level Recall@5
6. Latency profiling
7. Hybrid BM25 + dense retrieval
8. Error analysis
9. Interactive retrieval demo
10. Export results

## How to publish on Kaggle

**Automated (API):**

```bash
pip install kaggle
# Kaggle → Settings → API → export KAGGLE_API_TOKEN=KGAT_...
chmod +x scripts/publish_kaggle.sh
./scripts/publish_kaggle.sh
```

Default kernel URL: https://www.kaggle.com/code/waghelad/ds-rag-embedder-v1-train-benchmark

## Troubleshooting

| Error | Cause | Fix in notebook |
|-------|-------|-----------------|
| `AcceleratorError` on P100 | cu128 PyTorch lacks sm_60 kernels | Auto-install cu126 torch |
| `importlib.reload` / `ImportError` | Reloading torch after pip | Install cu126 **before** first import |
| `torchcodec` / `libavutil` | torchvision pulls FFmpeg deps | torch-only install, no torchvision |
| `pip install torch==2.5.1` fails | Wrong version for py3.12 | Uses **torch==2.10.0** cu126 |
| cu126 install fails | Network/index edge case | CPU fallback via `KAGGLE_DEVICE=cpu` |

**Manual:**

1. Go to [kaggle.com/code](https://www.kaggle.com/code) and create a new notebook
2. Upload the notebook or paste cells
3. Enable **GPU** accelerator and **Internet**
4. Title: **DS RAG Embedder v1 Train Benchmark**

## Link back to Hugging Face

- Model: https://huggingface.co/waghelad/ds-rag-embedder-v1
- Dataset: https://huggingface.co/datasets/waghelad/ds-rag-eval-v1
- GitHub: https://github.com/dgvj-work/ds-rag-embedder-v1
- PyPI: https://pypi.org/project/ds-rag-embedder/
