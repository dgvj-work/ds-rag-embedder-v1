"""DS RAG Embedder — domain-specific embeddings for Data Science & ML document retrieval."""

from ds_rag_embedder.model import DSRAGEmbedder
from ds_rag_embedder.config import EmbedderConfig, DEFAULT_CONFIG

__version__ = "1.0.0"
__all__ = ["DSRAGEmbedder", "EmbedderConfig", "DEFAULT_CONFIG", "__version__"]
