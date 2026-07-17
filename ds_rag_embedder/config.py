"""Configuration for DS RAG Embedder training and inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CORPUS_PATH = DATA_DIR / "corpus" / "ds_ml_corpus.jsonl"
EVAL_PATH = DATA_DIR / "eval" / "ds_retrieval_eval.jsonl"
BENCHMARK_PATH = DATA_DIR / "benchmarks" / "ds_rag_benchmark.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "models" / "ds-rag-embedder-v1"

# BGE-style query instruction (improves asymmetric retrieval)
QUERY_PREFIX = "Represent this Data Science question for retrieving relevant documentation: "
DOC_PREFIX = ""


@dataclass
class EmbedderConfig:
    """Hyperparameters and paths for the embedder."""

    base_model: str = "BAAI/bge-small-en-v1.5"
    hub_model_id: str = "waghelad/ds-rag-embedder-v1"
    hub_dataset_id: str = "waghelad/ds-rag-eval-v1"
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)
    max_seq_length: int = 512
    embedding_dim: int = 384
    query_prefix: str = QUERY_PREFIX
    doc_prefix: str = DOC_PREFIX
    normalize_embeddings: bool = True
    # Training
    epochs: int = 4
    batch_size: int = 32
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    seed: int = 42


DEFAULT_CONFIG = EmbedderConfig()

# Category tags for corpus filtering / metadata boosts
CORPUS_CATEGORIES = [
    "metrics",
    "feature_engineering",
    "model_selection",
    "validation",
    "data_leakage",
    "imbalanced_data",
    "hyperparameter_tuning",
    "mlops",
    "deep_learning",
    "nlp",
    "time_series",
    "statistics",
    "rag_llm",
    "pandas_sklearn",
    "experiment_design",
    "interpretability",
    "deployment",
]
