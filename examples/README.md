# Examples

Runnable scripts showing how to use **DS RAG Embedder v1** with common vector-search stacks.

All examples default to the published Hugging Face model: `waghelad/ds-rag-embedder-v1`.

## Prerequisites

```bash
pip install -e ".[integrations]"
# or minimal: pip install -e .
```

## Scripts

| Example | Command | What it demonstrates |
|---------|---------|----------------------|
| LangChain + Chroma | `python examples/langchain_example.py` | `DSRAGLangChainEmbeddings` with Chroma |
| LlamaIndex | `python examples/llama_index_example.py` | Vector store index + query engine |
| Chroma (manual) | `python examples/chromadb_example.py` | Encode queries/docs and search |
| FAISS | `python examples/faiss_example.py` | In-memory FAISS index |
| Hybrid BM25+dense | `python examples/hybrid_retrieval_example.py` | Dense vs hybrid retrieval |

## Query prefix reminder

Always use `DSRAGEmbedder.encode_queries()` for questions, or apply the BGE-style prefix documented in the README.
