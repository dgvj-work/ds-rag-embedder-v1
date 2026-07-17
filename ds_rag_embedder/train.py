"""Training utilities for fine-tuning the DS RAG embedder."""

from __future__ import annotations

import json
import random
from pathlib import Path

from ds_rag_embedder.config import DEFAULT_CONFIG, EmbedderConfig, CORPUS_PATH, EVAL_PATH


def load_corpus(path: Path | None = None) -> list[dict]:
    p = path or CORPUS_PATH
    rows: list[dict] = []
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_eval_pairs(path: Path | None = None) -> list[dict]:
    p = path or EVAL_PATH
    rows: list[dict] = []
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_training_examples(
    corpus: list[dict] | None = None,
    eval_pairs: list[dict] | None = None,
    seed: int = 42,
) -> list[dict]:
    """
    Build (anchor, positive) pairs for contrastive fine-tuning.

    Uses labeled eval pairs plus synthetic in-batch negatives (MNRL).
    """
    corpus = corpus or load_corpus()
    eval_pairs = eval_pairs or load_eval_pairs()
    rng = random.Random(seed)
    passages = {row["id"]: row["text"] for row in corpus}

    examples: list[dict] = []
    for pair in eval_pairs:
        pos_id = pair.get("positive_id") or pair.get("doc_id")
        text = pair.get("positive") or passages.get(pos_id)
        if text:
            examples.append({"query": pair["query"], "positive": text})

    # Augment with same-category pairs from corpus metadata
    by_cat: dict[str, list[str]] = {}
    for row in corpus:
        by_cat.setdefault(row.get("category", "general"), []).append(row["text"])

    for cat, texts in by_cat.items():
        if len(texts) < 2:
            continue
        for _ in range(min(3, len(texts))):
            a, b = rng.sample(texts, 2)
            q = f"Explain {cat.replace('_', ' ')} best practices in data science"
            examples.append({"query": q, "positive": b})

    rng.shuffle(examples)
    return examples


def train(
    config: EmbedderConfig | None = None,
    corpus_path: Path | None = None,
    eval_path: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Fine-tune base BGE model on DS/ML retrieval pairs."""
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from torch.utils.data import DataLoader

    cfg = config or DEFAULT_CONFIG
    out = Path(output_dir or cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    corpus = load_corpus(corpus_path)
    eval_pairs = load_eval_pairs(eval_path)
    train_examples = build_training_examples(corpus, eval_pairs, seed=cfg.seed)

    model = SentenceTransformer(cfg.base_model)
    model.max_seq_length = cfg.max_seq_length

    train_samples = [
        InputExample(
            texts=[f"{cfg.query_prefix}{ex['query']}", f"{cfg.doc_prefix}{ex['positive']}"]
        )
        for ex in train_examples
    ]
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=cfg.batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Build IR evaluator from eval pairs
    queries: dict[str, str] = {}
    corpus_dict: dict[str, str] = {row["id"]: row["text"] for row in corpus}
    relevant_docs: dict[str, set[str]] = {}
    for i, pair in enumerate(eval_pairs):
        qid = f"q{i}"
        queries[qid] = pair["query"]
        doc_id = pair.get("positive_id") or pair.get("doc_id")
        if doc_id and doc_id in corpus_dict:
            relevant_docs[qid] = {doc_id}
        elif pair.get("positive"):
            # inject positive into corpus for eval
            cid = f"eval_pos_{i}"
            corpus_dict[cid] = pair["positive"]
            relevant_docs[qid] = {cid}

    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus_dict,
        relevant_docs=relevant_docs,
        name="ds-rag-eval",
        show_progress_bar=True,
        query_prompt=cfg.query_prefix,
    )

    warmup_steps = int(len(train_dataloader) * cfg.epochs * cfg.warmup_ratio)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=cfg.epochs,
        warmup_steps=max(10, warmup_steps),
        optimizer_params={"lr": cfg.learning_rate},
        output_path=str(out),
        save_best_model=True,
        show_progress_bar=True,
    )

    meta = {
        "base_model": cfg.base_model,
        "train_pairs": len(train_examples),
        "corpus_size": len(corpus),
        "eval_queries": len(eval_pairs),
        "epochs": cfg.epochs,
    }
    (out / "training_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out
