"""
Local FAISS vector store with metadata persistence.
"""

from __future__ import annotations

import os
import json
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import faiss  # type: ignore


class FaissVectorStore:
    def __init__(self, index_path: str, meta_path: str) -> None:
        self.index_path = index_path
        self.meta_path = meta_path
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: List[Dict[str, Any]] = []
        if os.path.exists(index_path) and os.path.exists(meta_path):
            self._load()

    def _save(self) -> None:
        assert self.index is not None
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def _load(self) -> None:
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def _ensure_index(self, dim: int) -> None:
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)

    def add(self, vectors: np.ndarray, metadatas: List[Dict[str, Any]]) -> None:
        if vectors.size == 0:
            return
        dim = vectors.shape[1]
        self._ensure_index(dim)
        # Normalize for cosine similarity if not already normalized
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.metadata.extend(metadatas)
        self._save()

    def search(self, query_vec: np.ndarray, k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        faiss.normalize_L2(query_vec)
        D, I = self.index.search(query_vec, k)
        results: List[Tuple[float, Dict[str, Any]]] = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            results.append((float(score), meta))
        return results


