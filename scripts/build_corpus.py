#!/usr/bin/env python3
"""
Build DS/ML RAG training corpus, eval pairs, and retrieval benchmark.

Generates professional-quality synthetic + curated passages covering the topics
data scientists search daily: metrics, leakage, CV, imbalance, MLOps, RAG, etc.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_OUT = ROOT / "data" / "corpus" / "ds_ml_corpus.jsonl"
EVAL_OUT = ROOT / "data" / "eval" / "ds_retrieval_eval.jsonl"
BENCH_OUT = ROOT / "data" / "benchmarks" / "ds_rag_benchmark.jsonl"

# ---------------------------------------------------------------------------
# Curated DS/ML knowledge passages (high-quality, retrieval-ready)
# ---------------------------------------------------------------------------
PASSAGES: list[dict] = [
    {
        "id": "metrics_auc_vs_accuracy",
        "category": "metrics",
        "title": "AUC-ROC vs Accuracy for imbalanced classification",
        "text": (
            "Accuracy is misleading when classes are imbalanced. AUC-ROC measures ranking "
            "quality across thresholds and is robust to class priors. For heavy imbalance, "
            "also track PR-AUC, balanced accuracy, and per-class F1. Always align the metric "
            "with the business cost of false positives vs false negatives."
        ),
        "queries": [
            "Which metric should I use for imbalanced classification?",
            "Is accuracy enough when classes are imbalanced?",
            "AUC vs accuracy for skewed labels",
        ],
    },
    {
        "id": "leakage_target_encoding",
        "category": "data_leakage",
        "title": "Target leakage via global target encoding",
        "text": (
            "Target encoding computed on the full dataset before train/test split leaks "
            "label information. Fit encoders inside cross-validation folds or use "
            "out-of-fold statistics only. Any feature using future or label-derived "
            "information at training time inflates offline metrics and fails in production."
        ),
        "queries": [
            "How does target encoding cause data leakage?",
            "Can I target-encode before splitting data?",
            "Prevent leakage with categorical encoding",
        ],
    },
    {
        "id": "cv_nested",
        "category": "validation",
        "title": "Nested cross-validation for unbiased model selection",
        "text": (
            "When tuning hyperparameters, use nested CV: an outer loop estimates generalization "
            "and an inner loop selects hyperparameters. Standard k-fold on the same data used "
            "for tuning yields optimistically biased scores. Report mean and std of outer-fold "
            "metrics for stakeholder-ready evaluation."
        ),
        "queries": [
            "What is nested cross validation?",
            "How to tune hyperparameters without overfitting evaluation?",
            "Unbiased model selection workflow",
        ],
    },
    {
        "id": "imbalance_smote",
        "category": "imbalanced_data",
        "title": "Handling class imbalance with SMOTE and class weights",
        "text": (
            "For rare positive classes, try class weights, threshold tuning, or oversampling "
            "like SMOTE — but apply SMOTE only on training folds to avoid leakage. "
            "Prefer PR-AUC over ROC-AUC when positives are rare. Consider cost-sensitive "
            "learning if false negatives are expensive."
        ),
        "queries": [
            "How to handle class imbalance in sklearn?",
            "When should I use SMOTE?",
            "Best metrics for rare positive class",
        ],
    },
    {
        "id": "feature_scaling_tree",
        "category": "feature_engineering",
        "title": "When feature scaling matters",
        "text": (
            "Tree-based models (RandomForest, XGBoost, LightGBM) are scale-invariant. "
            "Linear models, SVMs, k-NN, and neural nets require scaling — typically "
            "StandardScaler or RobustScaler for heavy tails. Fit scalers on training data "
            "only and persist them in inference pipelines."
        ),
        "queries": [
            "Do I need to scale features for XGBoost?",
            "Which models require feature scaling?",
            "StandardScaler in production pipelines",
        ],
    },
    {
        "id": "hp_random_search",
        "category": "hyperparameter_tuning",
        "title": "Random search vs grid search",
        "text": (
            "Random search often outperforms grid search in high-dimensional hyperparameter "
            "spaces because it explores more distinct values per budget. Use Bayesian "
            "optimization (Optuna, Hyperopt) when evaluations are expensive. Always cap "
            "trials with early stopping and log all runs for reproducibility."
        ),
        "queries": [
            "Random search or grid search for hyperparameters?",
            "Efficient hyperparameter tuning for ML models",
            "Use Optuna for model tuning",
        ],
    },
    {
        "id": "mlops_model_registry",
        "category": "mlops",
        "title": "Model registry and promotion workflow",
        "text": (
            "A model registry tracks versions, metrics, artifacts, and stage transitions "
            "(Staging → Production). Promotion gates should require passing offline benchmarks, "
            "data drift checks, and shadow/canary evaluation. Store training data hash, "
            "feature schema, and dependency versions with each registered model."
        ),
        "queries": [
            "What is a model registry in MLOps?",
            "How to promote models to production safely?",
            "MLflow model registry best practices",
        ],
    },
    {
        "id": "drift_psi",
        "category": "mlops",
        "title": "Detecting data drift with PSI",
        "text": (
            "Population Stability Index (PSI) compares reference vs production feature "
            "distributions. PSI < 0.1 is stable; 0.1–0.25 moderate shift; > 0.25 significant "
            "drift. Monitor PSI on key features and model scores. Trigger retraining or "
            "fallback policies when drift persists across windows."
        ),
        "queries": [
            "How to detect data drift in production?",
            "What PSI threshold indicates drift?",
            "Monitor feature drift in ML systems",
        ],
    },
    {
        "id": "rag_chunking",
        "category": "rag_llm",
        "title": "Chunking strategies for RAG over technical docs",
        "text": (
            "For DS/ML docs, use semantic or structure-aware chunking: split on headings, "
            "keep code blocks intact, and use 300–800 token chunks with 10–20% overlap. "
            "Store metadata (section, library, task) for hybrid search. Evaluate retrieval "
            "with Recall@k and nDCG on domain-specific query sets — generic benchmarks "
            "understate production quality."
        ),
        "queries": [
            "Best chunk size for RAG on technical documentation?",
            "How to chunk ML docs for vector search?",
            "Evaluate RAG retrieval quality",
        ],
    },
    {
        "id": "rag_embeddings_domain",
        "category": "rag_llm",
        "title": "Domain-specific embeddings for RAG",
        "text": (
            "General-purpose embedders miss domain terminology (AUC, SHAP, SMOTE, dbt). "
            "Fine-tune embedding models on in-domain query-passage pairs or use asymmetric "
            "query/document prefixes (BGE-style). For daily DS workflows, domain embedders "
            "improve Recall@5 on notebooks, runbooks, and experiment logs vs generic MiniLM."
        ),
        "queries": [
            "Should I fine-tune embeddings for RAG?",
            "Domain-specific embedding models for ML docs",
            "BGE query prefix for retrieval",
        ],
    },
    {
        "id": "dl_learning_rate",
        "category": "deep_learning",
        "title": "Learning rate warmup and schedulers",
        "text": (
            "Use learning rate warmup for transformers and large batch training to stabilize "
            "early updates. Cosine decay and OneCycle are common schedulers. Track train vs "
            "validation loss; divergence often indicates excessive LR or batch size. "
            "Mixed precision (fp16/bf16) speeds training with minimal accuracy impact."
        ),
        "queries": [
            "Learning rate warmup for neural networks",
            "Best LR scheduler for fine-tuning transformers",
            "Training diverges high learning rate",
        ],
    },
    {
        "id": "ts_cross_validation",
        "category": "time_series",
        "title": "Time series cross-validation",
        "text": (
            "Never shuffle time series for CV — use expanding or sliding window splits "
            "to respect temporal order. Purged cross-validation removes overlapping labels "
            "in finance-style problems. Report metrics per fold and on the most recent "
            "holdout window to mimic production forecasting."
        ),
        "queries": [
            "How to do cross validation for time series?",
            "Avoid leakage in time series ML",
            "Expanding window validation",
        ],
    },
    {
        "id": "stats_pvalue",
        "category": "statistics",
        "title": "p-values and practical significance",
        "text": (
            "Statistical significance (p < 0.05) does not imply practical impact. With large "
            "samples, tiny effects become significant. Report effect sizes, confidence "
            "intervals, and business metrics alongside p-values. For A/B tests on models, "
            "pre-register metrics and minimum detectable effect."
        ),
        "queries": [
            "p-value vs practical significance in experiments",
            "How to interpret A/B test results for ML?",
            "Effect size in model experiments",
        ],
    },
    {
        "id": "shap_interpret",
        "category": "interpretability",
        "title": "SHAP for model interpretability",
        "text": (
            "SHAP explains individual predictions via Shapley values. TreeExplainer is fast "
            "for tree models; KernelExplainer works broadly but is slower. Use SHAP summary "
            "plots for global importance and force plots for single predictions. "
            "Correlated features can make SHAP attributions unstable — check robustness."
        ),
        "queries": [
            "How to use SHAP for feature importance?",
            "Explain individual predictions with SHAP",
            "SHAP for XGBoost models",
        ],
    },
    {
        "id": "deploy_batch_vs_online",
        "category": "deployment",
        "title": "Batch vs online inference patterns",
        "text": (
            "Batch scoring suits periodic reports and backfills; online serving needs low "
            "latency, autoscaling, and feature freshness from streaming/batch feature stores. "
            "Validate that offline training features match online serving with contract tests. "
            "Log predictions, features, and model version for auditability."
        ),
        "queries": [
            "Batch vs real-time ML inference architecture",
            "Training serving skew prevention",
            "Feature store for online ML serving",
        ],
    },
    {
        "id": "pandas_merge_leak",
        "category": "pandas_sklearn",
        "title": "Merge hygiene in pandas feature pipelines",
        "text": (
            "When building features with pandas merges, verify join keys, cardinality, and "
            "timestamps — as-of merges prevent future data leakage. After merges, check for "
            "duplicate rows and null rate spikes. Persist merge logic in reproducible "
            "pipeline steps, not notebook-only cells."
        ),
        "queries": [
            "Prevent data leakage in pandas feature engineering",
            "As-of merge for time series features",
            "pandas merge best practices ML",
        ],
    },
    {
        "id": "experiment_tracking",
        "category": "experiment_design",
        "title": "Experiment tracking essentials",
        "text": (
            "Log parameters, metrics, artifacts, git SHA, and data version for every run. "
            "Tools like MLflow, W&B, or Neptune enable comparison and reproducibility. "
            "Define a single primary metric before experiments. Avoid cherry-picking best "
            "runs without correcting for multiple comparisons."
        ),
        "queries": [
            "ML experiment tracking best practices",
            "What to log in MLflow runs?",
            "Reproducible machine learning experiments",
        ],
    },
    {
        "id": "nlp_text_classification",
        "category": "nlp",
        "title": "Text classification baselines",
        "text": (
            "Start with TF-IDF + linear model or fine-tune a small transformer (DistilBERT) "
            "depending on data size. For imbalanced text labels, stratify splits and use "
            "macro-F1. Augmentation (back-translation, synonym noise) helps low-resource "
            "regimes. Evaluate calibration if thresholds drive actions."
        ),
        "queries": [
            "Baseline approach for text classification",
            "Fine-tune transformer for small text dataset",
            "Metrics for imbalanced text labels",
        ],
    },
    {
        "id": "llm_eval_rag",
        "category": "rag_llm",
        "title": "Evaluating RAG systems beyond BLEU",
        "text": (
            "RAG quality requires retrieval metrics (Recall@k, MRR) and generation metrics "
            "(faithfulness, answer relevance). Use LLM-as-judge carefully with human-labeled "
            "gold sets. Test failure modes: missing context, conflicting docs, stale docs, "
            "and out-of-domain queries. Production RAG needs continuous eval on live queries."
        ),
        "queries": [
            "How to evaluate RAG pipeline quality?",
            "RAG metrics recall faithfulness",
            "LLM as judge for RAG evaluation",
        ],
    },
    {
        "id": "model_selection_bias_variance",
        "category": "model_selection",
        "title": "Bias-variance and model complexity",
        "text": (
            "High bias (underfitting): increase model capacity or features. High variance "
            "(overfitting): regularization, more data, simpler models, or ensembling. "
            "Learning curves diagnose data vs algorithm limits. Prefer simpler models that "
            "meet business metrics — complexity increases maintenance and drift risk."
        ),
        "queries": [
            "Bias variance tradeoff in model selection",
            "Diagnose overfitting with learning curves",
            "When to use simpler ML models",
        ],
    },
]

# Template expansions for corpus volume
EXPANSION_TEMPLATES = [
    (
        "metrics",
        "When optimizing for {metric}, ensure the evaluation split reflects production "
        "class distribution. Report confidence intervals via bootstrap. Compare against "
        "a strong baseline before claiming improvement.",
        ["{metric} evaluation best practices", "How to evaluate {metric} reliably?"],
    ),
    (
        "feature_engineering",
        "Feature {feat} should be computed with point-in-time correctness. Document "
        "null handling and expected cardinality. Monitor distribution drift after deployment.",
        ["Engineering feature {feat}", "Feature store pattern for {feat}"],
    ),
    (
        "mlops",
        "Pipeline step '{step}' should be idempotent, versioned, and observable. Alert on "
        "failures and SLA breaches. Include data quality checks before model training.",
        ["MLOps pipeline {step}", "Production checklist for {step}"],
    ),
]

METRICS = ["F1", "ROC-AUC", "PR-AUC", "RMSE", "MAE", "R2", "log loss", "NDCG"]
FEATURES = ["recency", "frequency", "rolling mean", "embedding", "category count"]
STEPS = ["ingestion", "validation", "training", "batch scoring", "monitoring"]


def _expand_corpus(rng: random.Random, target_size: int) -> list[dict]:
    rows = [dict(p) for p in PASSAGES]
    idx = 0
    while len(rows) < target_size:
        cat, text_tpl, query_tpls = rng.choice(EXPANSION_TEMPLATES)
        if "{metric}" in text_tpl:
            metric = rng.choice(METRICS)
            text = text_tpl.format(metric=metric)
            queries = [q.format(metric=metric) for q in query_tpls]
            title = f"{metric} evaluation guide"
        elif "{feat}" in text_tpl:
            feat = rng.choice(FEATURES)
            text = text_tpl.format(feat=feat)
            queries = [q.format(feat=feat) for q in query_tpls]
            title = f"Feature engineering: {feat}"
        else:
            step = rng.choice(STEPS)
            text = text_tpl.format(step=step)
            queries = [q.format(step=step) for q in query_tpls]
            title = f"MLOps: {step}"
        doc_id = f"auto_{cat}_{idx}"
        rows.append(
            {
                "id": doc_id,
                "category": cat,
                "title": title,
                "text": text,
                "queries": queries,
                "source": "synthetic_expansion",
            }
        )
        idx += 1
    return rows[:target_size]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def build(seed: int = 42, corpus_size: int = 600) -> dict:
    rng = random.Random(seed)
    corpus = _expand_corpus(rng, corpus_size)

    eval_rows: list[dict] = []
    benchmark_rows: list[dict] = []
    for doc in PASSAGES:  # benchmark anchored on high-quality curated set
        for q in doc.get("queries", []):
            eval_rows.append(
                {
                    "query": q,
                    "positive_id": doc["id"],
                    "positive": doc["text"],
                    "category": doc["category"],
                }
            )
            benchmark_rows.append(
                {
                    "query": q,
                    "relevant_ids": [doc["id"]],
                    "category": doc["category"],
                }
            )

    # Additional eval from expanded corpus
    for doc in corpus:
        if doc["id"].startswith("auto_") and doc.get("queries"):
            q = doc["queries"][0]
            eval_rows.append(
                {
                    "query": q,
                    "positive_id": doc["id"],
                    "positive": doc["text"],
                    "category": doc["category"],
                }
            )

    _write_jsonl(CORPUS_OUT, corpus)
    _write_jsonl(EVAL_OUT, eval_rows)
    _write_jsonl(BENCH_OUT, benchmark_rows)

    stats = {
        "corpus_size": len(corpus),
        "eval_pairs": len(eval_rows),
        "benchmark_queries": len(benchmark_rows),
        "corpus_path": str(CORPUS_OUT),
        "eval_path": str(EVAL_OUT),
        "benchmark_path": str(BENCH_OUT),
    }
    (ROOT / "data" / "corpus_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DS/ML RAG corpus and eval sets")
    parser.add_argument("--corpus-size", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    stats = build(seed=args.seed, corpus_size=args.corpus_size)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
