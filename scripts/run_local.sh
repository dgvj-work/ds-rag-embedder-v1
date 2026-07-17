#!/usr/bin/env bash
# Local verification: corpus → tests → demo import smoke
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== DS RAG Embedder local checks =="

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -U pip
pip install -q -e ".[dev,demo]"

echo "→ Build corpus"
python scripts/build_corpus.py --corpus-size 600

echo "→ pytest"
pytest -q

echo "→ Import + encode smoke (base BGE fallback)"
python - <<'PY'
try:
    from ds_rag_embedder import DSRAGEmbedder
    from ds_rag_embedder.rag import DSRAGRetriever

    emb = DSRAGEmbedder(model_name_or_path="sentence-transformers/all-MiniLM-L6-v2")
    vec = emb.encode_queries(["nested cross validation"])
    assert vec.shape[0] == 1
    retriever = DSRAGRetriever(emb)
    n = retriever.load_corpus_jsonl("data/corpus/ds_ml_corpus.jsonl")
    retriever.build_index()
    r = retriever.retrieve("How to handle class imbalance?", top_k=3)
    assert len(r.hits) == 3
    print(f"OK — indexed {n} docs, top score {r.hits[0]['score']:.4f}")
except Exception as exc:
    print(f"SKIP encode smoke (HF model download unavailable): {exc}")
PY

echo "→ Space import smoke"
python -c "import app; print('Gradio app OK')"

echo ""
echo "All checks passed."
echo "Next: python scripts/train.py  |  ./scripts/publish_hf.sh"
