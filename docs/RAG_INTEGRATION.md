# RAG Integration Guide

## Query prefix (required for best results)

```
Represent this Data Science question for retrieving relevant documentation: 
```

Documents: encode as-is (no prefix).

## Python API

```python
from ds_rag_embedder import DSRAGEmbedder

embedder = DSRAGEmbedder("waghelad/ds-rag-embedder-v1")
q_emb = embedder.encode_queries(["nested cross validation"])
d_emb = embedder.encode_documents(["Use nested CV for hyperparameter tuning..."])
```

## Full RAG pipeline

```python
from ds_rag_embedder.rag import DSRAGPipeline

pipe = DSRAGPipeline()
out = pipe.retrieve("How to detect data drift?")
# pass out.prompt to your LLM
```

## Framework examples

| Framework | Example |
|-----------|---------|
| LangChain | `examples/langchain_example.py` |
| LlamaIndex | `examples/llama_index_example.py` |
| ChromaDB | `examples/chromadb_example.py` |
| FAISS | `examples/faiss_example.py` |

## Production tips

1. Chunk docs at 300–500 tokens with overlap
2. Store metadata (category, source, date)
3. Monitor Recall@k on live query logs
4. Combine with reranker for top precision
5. Refresh index when docs change

## Chunking

```python
from ds_rag_embedder.rag import chunk_text

chunks = chunk_text(long_markdown, chunk_size=400, chunk_overlap=60)
```
