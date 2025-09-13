"""
MMR-based retriever on top of FAISS results.

Implements Maximal Marginal Relevance to diversify retrieved chunks,
inspired by common RAG pipelines like the one in RAG-GPT.
"""

from typing import List, Dict, Any, Tuple
import os
import numpy as np

from .embeddings import Embeddings
from .vector_store import FaissVectorStore


def _mmr_select(
    q_vec: np.ndarray,
    cand_vecs: np.ndarray,
    k: int,
    lambda_mult: float = 0.5,
) -> List[int]:
    """
    Greedy MMR selection.

    Args:
        q_vec: (1, d) normalized query vector
        cand_vecs: (n, d) normalized candidate vectors
        k: number of items to select
        lambda_mult: tradeoff between relevance and diversity (0..1)
    Returns:
        indices of selected candidates
    """
    n = cand_vecs.shape[0]
    if n == 0:
        return []
    k = min(k, n)

    # similarity to query (cosine, vectors assumed normalized)
    rel = (cand_vecs @ q_vec.T).reshape(-1)

    selected: List[int] = []
    remaining = set(range(n))

    # pick the most relevant first
    first = int(np.argmax(rel))
    selected.append(first)
    remaining.remove(first)

    if k == 1:
        return selected

    # precompute pairwise sims
    pair = cand_vecs @ cand_vecs.T

    while len(selected) < k and remaining:
        # for each remaining, compute max similarity to any selected
        max_sim_to_selected = np.array([
            max(pair[i, selected]) if selected else 0.0 for i in remaining
        ])
        rem_list = list(remaining)
        # MMR score
        mmr_scores = lambda_mult * rel[rem_list] - (1 - lambda_mult) * max_sim_to_selected
        next_idx = rem_list[int(np.argmax(mmr_scores))]
        selected.append(next_idx)
        remaining.remove(next_idx)

    return selected


def mmr_retrieve(
    query: str,
    top_k: int = 5,
    fetch_k: int = 20,
    store_dir: str = "data/vector_store",
    model_name: str = "all-MiniLM-L6-v2",
    lambda_mult: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Retrieve top_k diverse chunks using MMR.
    """
    index_path = os.path.join(store_dir, "index.faiss")
    meta_path = os.path.join(store_dir, "meta.json")
    store = FaissVectorStore(index_path, meta_path)
    embed = Embeddings(model_name)

    # Initial dense retrieval
    from .ingest import semantic_search
    hits = semantic_search(query, store_dir=store_dir, top_k=fetch_k, model_name=model_name)
    if not hits:
        return []

    metas: List[Dict[str, Any]] = [m for _, m in hits]
    texts: List[str] = [m.get("text", "") for m in metas]
    # Re-embed candidate texts for diversification
    cand_vecs = embed.embed_texts(texts)
    q_vec = embed.embed_query(query)

    # Vectors already normalized by embedder; ensure normalization
    # (defensive) normalize
    cand_norm = cand_vecs / (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-12)
    q_norm = q_vec / (np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-12)

    sel_idx = _mmr_select(q_norm, cand_norm, k=top_k, lambda_mult=lambda_mult)
    return [metas[i] for i in sel_idx]


