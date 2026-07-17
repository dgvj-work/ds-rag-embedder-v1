"""Hybrid BM25 + dense retrieval example."""

from ds_rag_embedder import DSRAGEmbedder
from ds_rag_embedder.rag import HybridRetriever


def main() -> None:
    embedder = DSRAGEmbedder("waghelad/ds-rag-embedder-v1")
    documents = [
        "Apply SMOTE only on training folds to avoid leakage in imbalanced classification.",
        "Use nested cross-validation when tuning hyperparameters.",
        "Population Stability Index above 0.25 signals significant feature drift.",
        "Tree models like XGBoost do not require feature scaling.",
        "Fine-tune LoRA adapters when GPU memory is limited.",
    ]
    metadata = [{"category": "demo"} for _ in documents]

    hybrid = HybridRetriever(embedder=embedder, documents=documents, metadata=metadata, alpha=0.65)
    query = "SMOTE leakage cross validation"

    print("Dense only:")
    for hit in embedder.search(query, documents, top_k=3):
        print(f"  {hit['score']:.4f} {hit['document'][:70]}...")

    print("\nHybrid BM25 + dense:")
    for hit in hybrid.retrieve(query, top_k=3).hits:
        print(
            f"  {hit['score']:.4f} (dense={hit['dense_score']:.3f}, bm25={hit['bm25_score']:.3f}) "
            f"{hit['text'][:70]}..."
        )


if __name__ == "__main__":
    main()
