# Hugging Face publish record

This folder documents the published Hugging Face artifacts and keeps GitHub in sync with the Hub.

## Live assets

| Asset | URL | Status |
|-------|-----|--------|
| Model | https://huggingface.co/waghelad/ds-rag-embedder-v1 | Published |
| Dataset | https://huggingface.co/datasets/waghelad/ds-rag-eval-v1 | Published |
| Space demo | https://huggingface.co/spaces/waghelad/ds-rag-embedder-demo | Gradio SDK 5.12 — sync via `scripts/publish_space.sh` |

## Model publish commit (HF)

- Weights commit: `fda3162a1fb08d7d96eb3eaf952061e1303cd98f`
- README sync commit: `6631524c09e5635fce5b3831e9fbaed8025fa929`
- Size: ~133 MB (`model.safetensors`)
- Base model: `BAAI/bge-small-en-v1.5`

## Training run (local, used for publish)

See [`training_meta.json`](training_meta.json):

- Corpus size: 600 passages
- Train pairs: 675
- Epochs: 4
- Eval benchmark queries: 87

## Benchmark results (published eval)

Canonical file on GitHub: [`../outputs/eval_results.json`](../outputs/eval_results.json)

| Model | Recall@1 | Recall@5 | MRR | nDCG@10 |
|-------|----------|----------|-----|---------|
| all-MiniLM-L6-v2 | 0.621 | 0.828 | 0.708 | 0.740 |
| bge-small-en-v1.5 | 0.506 | 0.609 | 0.558 | 0.567 |
| ds-rag-embedder-v1 | 0.851 | 1.000 | 0.921 | 0.942 |

## Sync checklist

GitHub files that must match Hugging Face:

| GitHub file | HF location |
|-------------|-------------|
| `MODEL_CARD.md` | Model repo `README.md` |
| `outputs/eval_results.json` | Embedded in model card under "Latest evaluation" |
| `data/*` corpus/eval/benchmark | Dataset repo configs + `data/*.jsonl` |
| `scripts/publish_dataset.py` | Dataset upload logic |
| `scripts/publish_space.sh` | Space upload (app.py, package, corpus, README) |
| `requirements-space.txt` | Space `requirements.txt` (runtime deps only, no editable install) |

To re-export model card from repo source:

```bash
python scripts/export_hf.py --model-dir models/ds-rag-embedder-v1
hf upload waghelad/ds-rag-embedder-v1 exports/hf-model . --repo-type=model
```
