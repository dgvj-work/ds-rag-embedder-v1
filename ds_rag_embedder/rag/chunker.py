"""Text chunking utilities for DS/ML documentation RAG."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    index: int
    metadata: dict


def chunk_text(
    text: str,
    chunk_size: int = 400,
    chunk_overlap: int = 60,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Split markdown/plain docs into overlapping chunks for vector indexing."""
    metadata = metadata or {}
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if len(text) <= chunk_size:
        return [Chunk(text=text, index=0, metadata={**metadata, "chunk": 0})]

    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(text=piece, index=idx, metadata={**metadata, "chunk": idx}))
            idx += 1
        if end >= len(text):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def chunk_notebook_cells(cells: list[dict], chunk_size: int = 400) -> list[Chunk]:
    """Chunk Jupyter notebook markdown/code cells."""
    parts: list[str] = []
    for cell in cells:
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        cell_type = cell.get("cell_type", "markdown")
        parts.append(f"[{cell_type}]\n{src.strip()}")
    return chunk_text("\n\n".join(parts), chunk_size=chunk_size, metadata={"source": "notebook"})
