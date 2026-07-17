"""Example: ChromaDB + DS RAG Embedder."""

def main() -> None:
    try:
        import chromadb
    except ImportError:
        print("Install: pip install chromadb")
        return

    from ds_rag_embedder import DSRAGEmbedder

    client = chromadb.Client()
    collection = client.create_collection("ds_docs")
    embedder = DSRAGEmbedder("waghelad/ds-rag-embedder-v1")
    docs = [
        "Nested CV reduces optimistic bias during hyperparameter tuning.",
        "Population Stability Index PSI detects feature drift in production.",
    ]
    embeddings = embedder.encode_documents(docs).tolist()
    collection.add(ids=["1", "2"], documents=docs, embeddings=embeddings)
    q_emb = embedder.encode_queries(["hyperparameter tuning without overfitting"]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=1)
    print(results["documents"])


if __name__ == "__main__":
    main()
