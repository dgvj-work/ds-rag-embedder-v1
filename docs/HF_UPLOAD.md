# Hugging Face Upload Guide

## Repos to create

| Asset | Repo ID | Type |
|-------|---------|------|
| Model | `waghelad/ds-rag-embedder-v1` | model |
| Dataset | `waghelad/ds-rag-eval-v1` | dataset |
| Demo | `waghelad/ds-rag-embedder-demo` | space |

Create empty repos on [huggingface.co/new](https://huggingface.co/new) or let the CLI create them on first upload.

## One-command publish

```bash
pip install -U "huggingface_hub[cli]" sentence-transformers datasets
hf auth login

chmod +x scripts/publish_hf.sh
./scripts/publish_hf.sh
```

## Manual steps

```bash
python scripts/build_corpus.py
python scripts/train.py
python scripts/evaluate.py --compare
python scripts/export_hf.py
hf upload waghelad/ds-rag-embedder-v1 exports/hf-model . --repo-type=model
python scripts/publish_dataset.py --repo waghelad/ds-rag-eval-v1
```

## Space upload

1. Create Space (Gradio SDK)
2. Copy `README_HF_SPACE.md` → Space README
3. Upload project files or connect Git repo
4. Set `DS_RAG_MODEL=waghelad/ds-rag-embedder-v1` if needed

```bash
hf upload waghelad/ds-rag-embedder-demo . --repo-type=space
```

## Checklist before publish

- [ ] Model is **public**
- [ ] Dataset is **public** and linked in model card
- [ ] Model card has query prefix documented
- [ ] Space builds and retrieves example queries
- [ ] Pin Space on profile
- [ ] HF Community post announcing launch

## Download optimization tips

- Use `pipeline_tag: feature-extraction`
- Tag with `rag`, `data-science`, `sentence-transformers`
- Add 5+ copy-paste examples in model card
- Cross-link dataset + Space + GitHub
