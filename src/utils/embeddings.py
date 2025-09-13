"""
Embeddings helper using sentence-transformers.
"""

from typing import List
import numpy as np

from sentence_transformers import SentenceTransformer


class Embeddings:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        # Small, fast, good quality. Replaceable later.
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        vectors = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        # Ensure float32 for FAISS
        return vectors.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])


