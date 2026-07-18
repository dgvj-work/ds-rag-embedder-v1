"""Tests for DS RAG Embedder."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """Isolated corpus/eval/benchmark files for tests (never touches data/)."""
    return tmp_path / "data"


def test_build_corpus(data_root: Path):
    from scripts.build_corpus import build

    stats = build(corpus_size=120, seed=0, output_root=data_root)
    assert stats["corpus_size"] == 120
    assert stats["eval_pairs"] > 20
    assert (data_root / "corpus" / "ds_ml_corpus.jsonl").exists()


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


def test_retriever_loads_corpus(data_root: Path):
    from scripts.build_corpus import build
    from ds_rag_embedder.rag.retriever import DSRAGRetriever

    build(corpus_size=50, seed=1, output_root=data_root)
    emb = _mini_embedder()
    retriever = DSRAGRetriever(emb)
    corpus_path = data_root / "corpus" / "ds_ml_corpus.jsonl"
    n = retriever.load_corpus_jsonl(str(corpus_path))
    assert n == 50
    retriever.build_index()
    result = retriever.retrieve("data leakage target encoding", top_k=2)
    assert len(result.hits) == 2


def test_rag_pipeline_prompt(data_root: Path):
    from scripts.build_corpus import build
    from ds_rag_embedder.rag.pipeline import DSRAGPipeline

    build(corpus_size=50, seed=2, output_root=data_root)
    pipe = DSRAGPipeline(
        embedder=_mini_embedder(),
        corpus_path=str(data_root / "corpus" / "ds_ml_corpus.jsonl"),
    )
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


def test_evaluate_by_category(data_root: Path):
    from scripts.build_corpus import build
    from ds_rag_embedder.evaluate import evaluate_by_category

    build(corpus_size=80, seed=4, output_root=data_root)
    report = evaluate_by_category(
        _mini_embedder(),
        benchmark_path=data_root / "benchmarks" / "ds_rag_benchmark.jsonl",
        corpus_path=data_root / "corpus" / "ds_ml_corpus.jsonl",
    )
    assert report.overall.num_queries > 0
    assert len(report.by_category) > 0


def test_training_examples(data_root: Path):
    from scripts.build_corpus import build
    from ds_rag_embedder.train import build_training_examples, load_corpus, load_eval_pairs

    build(corpus_size=80, seed=3, output_root=data_root)
    corpus = load_corpus(data_root / "corpus" / "ds_ml_corpus.jsonl")
    eval_pairs = load_eval_pairs(data_root / "eval" / "ds_retrieval_eval.jsonl")
    examples = build_training_examples(corpus, eval_pairs)
    assert len(examples) > 30
    assert "query" in examples[0] and "positive" in examples[0]
