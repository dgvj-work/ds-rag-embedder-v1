"""Example: FAISS + DS RAG Embedder."""

import numpy as np


def main() -> None:
    try:
        import faiss
    except ImportError:
        print("Install: pip install faiss-cpu")
        return

    from ds_rag_embedder import DSRAGEmbedder

    embedder = DSRAGEmbedder("waghelad/ds-rag-embedder-v1")
    docs = [
        "Target encoding before train/test split causes label leakage.",
        "Learning rate warmup stabilizes transformer fine-tuning.",
        "PR-AUC is informative when positive class is rare.",
    ]
    d_emb = embedder.encode_documents(docs).astype("float32")
    index = faiss.IndexFlatIP(d_emb.shape[1])
    index.add(d_emb)
    q_emb = embedder.encode_queries(["avoid leakage in encoding"]).astype("float32")
    scores, idx = index.search(q_emb, k=2)
    for i, doc_idx in enumerate(idx[0]):
        print(scores[0][i], docs[doc_idx])


if __name__ == "__main__":
    main()
