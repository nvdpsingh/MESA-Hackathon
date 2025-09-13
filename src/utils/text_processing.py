"""
Lightweight text processing utilities: chunking and normalization.
"""

from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> List[str]:
    """
    Split text into overlapping chunks suitable for embedding.

    Args:
        text: Raw input text
        chunk_size: Max characters per chunk
        chunk_overlap: Overlap between consecutive chunks

    Returns:
        List of chunk strings
    """
    if not text:
        return []

    normalized = " ".join(text.split())
    if len(normalized) <= chunk_size:
        return [normalized]

    chunks: List[str] = []
    start = 0
    while start < len(normalized):
        end = start + chunk_size
        chunk = normalized[start:end]
        chunks.append(chunk)
        if end >= len(normalized):
            break
        start = max(0, end - chunk_overlap)
    return chunks


