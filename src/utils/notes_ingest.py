import os
from typing import List, Dict, Any

from .embeddings import Embeddings
from .vector_store import FaissVectorStore


def ingest_notes_sections(sections: List[str], store_dir: str = "data/notes_index", model_name: str = "all-MiniLM-L6-v2") -> str:
	os.makedirs(store_dir, exist_ok=True)
	index_path = os.path.join(store_dir, "index.faiss")
	meta_path = os.path.join(store_dir, "meta.json")
	embed = Embeddings(model_name)
	store = FaissVectorStore(index_path, meta_path)
	vectors = embed.embed_texts(sections)
	metas: List[Dict[str, Any]] = []
	for i, s in enumerate(sections):
		metas.append({"source": "notes", "chunk_index": i, "text": s})
	store.add(vectors, metas)
	return store_dir
