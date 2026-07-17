"""Example: LlamaIndex with DS RAG Embedder."""

def main() -> None:
    try:
        from llama_index.core import Document, VectorStoreIndex, Settings
    except ImportError:
        print("Install: pip install llama-index-core")
        return

    from ds_rag_embedder.integrations.llama_index import DSRAGLlamaIndexEmbedding

    Settings.embed_model = DSRAGLlamaIndexEmbedding("waghelad/ds-rag-embedder-v1")
    docs = [
        Document(text="AUC-ROC is preferred over accuracy for imbalanced labels."),
        Document(text="Use expanding window CV for time series forecasting."),
    ]
    index = VectorStoreIndex.from_documents(docs)
    engine = index.as_query_engine(similarity_top_k=2)
    response = engine.query("metric for imbalanced classification")
    print(response)


if __name__ == "__main__":
    main()
