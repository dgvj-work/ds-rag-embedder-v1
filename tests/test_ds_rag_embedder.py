"""Tests for DS RAG Embedder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_build_corpus():
    from scripts.build_corpus import build

    stats = build(corpus_size=120, seed=0)
    assert stats["corpus_size"] == 120
    assert stats["eval_pairs"] > 20
    assert Path(stats["corpus_path"]).exists()


def test_chunker():
    from ds_rag_embedder.rag.chunker import chunk_text

    text = "word " * 500
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(c.text for c in chunks)


def _mini_embedder():
    pytest.importorskip("sentence_transformers")
    from ds_rag_embedder.model import DSRAGEmbedder

    try:
        embedder = DSRAGEmbedder(model_name_or_path="sentence-transformers/all-MiniLM-L6-v2")
        embedder.encode_documents(["warmup"])
        return embedder
    except Exception as exc:  # HF auth/network in CI or offline envs
        pytest.skip(f"Could not load sentence-transformers model: {exc}")


def test_embedder_search_smoke():
    embedder = _mini_embedder()
    docs = [
        "Use SMOTE on training data only for class imbalance.",
        "Learning rate warmup helps transformer training stability.",
    ]
    hits = embedder.search("class imbalance SMOTE", docs, top_k=1)
    assert hits[0]["score"] > 0.2


def test_retriever_loads_corpus():
    from scripts.build_corpus import build
    from ds_rag_embedder.model import DSRAGEmbedder
    from ds_rag_embedder.rag.retriever import DSRAGRetriever

    build(corpus_size=50, seed=1)
    emb = _mini_embedder()
    retriever = DSRAGRetriever(emb)
    n = retriever.load_corpus_jsonl(str(ROOT / "data" / "corpus" / "ds_ml_corpus.jsonl"))
    assert n == 50
    retriever.build_index()
    result = retriever.retrieve("data leakage target encoding", top_k=2)
    assert len(result.hits) == 2


def test_rag_pipeline_prompt():
    from scripts.build_corpus import build
    from ds_rag_embedder.rag.pipeline import DSRAGPipeline

    build(corpus_size=50, seed=2)
    pipe = DSRAGPipeline(embedder=_mini_embedder())
    out = pipe.retrieve("AUC vs accuracy")
    assert "Question:" in out.prompt
    assert len(out.contexts) >= 1


def test_hybrid_retriever():
    from ds_rag_embedder.rag import HybridRetriever

    embedder = _mini_embedder()
    docs = [
        "Apply SMOTE on training folds only to avoid leakage.",
        "Use nested cross-validation for hyperparameter tuning.",
        "PSI above 0.25 indicates drift.",
    ]
    hybrid = HybridRetriever(embedder=embedder, documents=docs, alpha=0.65)
    result = hybrid.retrieve("SMOTE leakage", top_k=2)
    assert len(result.hits) == 2
    assert result.hits[0]["score"] >= result.hits[1]["score"]


def test_evaluate_by_category():
    from scripts.build_corpus import build
    from ds_rag_embedder.evaluate import evaluate_by_category

    build(corpus_size=80, seed=4)
    report = evaluate_by_category(_mini_embedder())
    assert report.overall.num_queries > 0
    assert len(report.by_category) > 0


def test_training_examples():
    from scripts.build_corpus import build
    from ds_rag_embedder.train import build_training_examples

    build(corpus_size=80, seed=3)
    examples = build_training_examples()
    assert len(examples) > 30
    assert "query" in examples[0] and "positive" in examples[0]
