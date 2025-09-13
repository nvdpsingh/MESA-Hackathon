"""
Ingestion utilities: take extraction results and add to FAISS store.
"""

from typing import List, Dict, Any
import os

from .text_processing import chunk_text
from .embeddings import Embeddings
from .vector_store import FaissVectorStore


def ingest_documents(
    docs: List[Dict[str, Any]],
    store_dir: str = "data/vector_store",
    model_name: str = "all-MiniLM-L6-v2",
) -> str:
    os.makedirs(store_dir, exist_ok=True)
    index_path = os.path.join(store_dir, "index.faiss")
    meta_path = os.path.join(store_dir, "meta.json")

    embed = Embeddings(model_name)
    store = FaissVectorStore(index_path, meta_path)

    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for doc in docs:
        if not doc.get("success"):
            continue
        source = doc.get("metadata", {}).get("file_name") or doc.get("video_id") or "unknown"
        full_text = doc.get("full_text")
        if not full_text:
            # fallbacks
            if "page_texts" in doc:
                # PDF pages
                full_text = "\n".join(p.get("text", "") for p in doc["page_texts"]) or ""
            elif "segments" in doc:
                full_text = "\n".join(s.get("text", "") for s in doc["segments"]) or ""
        chunks = chunk_text(full_text or "")
        for i, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append({
                "source": source,
                "chunk_index": i,
                "char_count": len(chunk),
                "text": chunk,
            })

    if texts:
        vectors = embed.embed_texts(texts)
        store.add(vectors, metadatas)

    return store_dir


def semantic_search(
    query: str,
    store_dir: str = "data/vector_store",
    top_k: int = 5,
    model_name: str = "all-MiniLM-L6-v2",
):
    index_path = os.path.join(store_dir, "index.faiss")
    meta_path = os.path.join(store_dir, "meta.json")
    embed = Embeddings(model_name)
    store = FaissVectorStore(index_path, meta_path)

    qvec = embed.embed_query(query)
    return store.search(qvec, k=top_k)


