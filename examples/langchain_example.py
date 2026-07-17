"""Example: LangChain vector store with DS RAG Embedder."""

from ds_rag_embedder.integrations.langchain import DSRAGLangChainEmbeddings

# pip install langchain-community chromadb

def main() -> None:
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError:
        print("Install: pip install langchain-community chromadb")
        return

    docs = [
        "Use nested cross-validation when tuning hyperparameters.",
        "SMOTE should be applied only on training folds to avoid leakage.",
        "Monitor PSI above 0.25 as significant data drift.",
    ]
    embeddings = DSRAGLangChainEmbeddings("waghelad/ds-rag-embedder-v1")
    store = Chroma.from_texts(docs, embedding=embeddings)
    results = store.similarity_search("prevent data leakage with SMOTE", k=2)
    for doc in results:
        print("-", doc.page_content)


if __name__ == "__main__":
    main()
