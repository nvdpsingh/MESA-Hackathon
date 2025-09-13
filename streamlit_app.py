import os
import tempfile
import json
import shutil
import streamlit as st
try:
	from dotenv import load_dotenv  # type: ignore
	load_dotenv(override=True)
except Exception:
	pass

from src.pipelines.text_extraction_pipeline import TextExtractionPipeline
from src.utils.ingest import ingest_documents
from src.utils.retrieval import mmr_retrieve
from src.utils.rag_llm import answer_with_context
from src.utils.notes import generate_notes_from_text, iter_generate_notes_from_texts

st.set_page_config(page_title="LMS RAG Demo", page_icon="📚", layout="wide")
st.title("📚 LMS RAG Demo")

# Vector store helpers (single global store)
def _store_dir() -> str:
	return os.path.join("data", "vector_store")

def _meta_path() -> str:
	return os.path.join(_store_dir(), "meta.json")

def _existing_sources() -> set[str]:
	meta = _meta_path()
	if not os.path.exists(meta):
		return set()
	try:
		with open(meta, "r", encoding="utf-8") as f:
			metas = json.load(f)
		return {m.get("source", "") for m in metas}
	except Exception:
		return set()

def _maybe_clear_index(expected_sources: set[str]) -> None:
	current = _existing_sources()
	if current and current != expected_sources:
		try:
			shutil.rmtree(_store_dir())
		except FileNotFoundError:
			pass

with st.sidebar:
	st.header("Settings")
	embed_model = st.text_input("Embedding model", value="all-MiniLM-L6-v2")
	top_k = st.number_input("Top K", min_value=1, max_value=20, value=5)
	# Auto backend from environment; no key prompts
	if os.getenv("GROQ_API_KEY"):
		os.environ["LLM_BACKEND"] = "groq"
		st.caption(f"Backend: Groq (GROQ_MODEL={os.getenv('GROQ_MODEL','auto')})")
	elif os.getenv("OPENAI_API_KEY"):
		os.environ["LLM_BACKEND"] = "openai"
		st.caption(f"Backend: OpenAI (OPENAI_MODEL={os.getenv('OPENAI_MODEL','gpt-4o-mini')})")
	else:
		os.environ["LLM_BACKEND"] = "ollama"
		st.caption(f"Backend: Ollama (model={os.getenv('OLLAMA_MODEL','llama3.1:8b')})")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
	st.subheader("Ingest PDFs / DOCX")
	uploaded = st.file_uploader("Upload files", type=["pdf", "docx", "doc"], accept_multiple_files=True)
	if st.button("Ingest Uploaded") and uploaded:
		with st.spinner("Extracting and ingesting..."):
			pipe = TextExtractionPipeline()
			results = []
			tmpdir = tempfile.mkdtemp()
			planned_sources = set()
			for f in uploaded:
				path = os.path.join(tmpdir, f.name)
				with open(path, "wb") as out:
					out.write(f.read())
				res = pipe.extract_from_file(path)
				if res.get("success"):
					results.append(res)
					src = res.get("metadata", {}).get("file_name") or "unknown"
					planned_sources.add(src)
			_maybe_clear_index(planned_sources)
			os.makedirs(_store_dir(), exist_ok=True)
			_store = ingest_documents(results, store_dir=_store_dir(), model_name=embed_model)
		st.success("Ingested.")

with col2:
	st.subheader("Ingest YouTube URL / Playlist")
	yt_url = st.text_input("YouTube URL or Playlist")
	if st.button("Ingest YouTube") and yt_url:
		with st.spinner("Fetching transcript(s) and ingesting..."):
			pipe = TextExtractionPipeline()
			res = pipe.extract_from_youtube(yt_url)
			docs = []
			planned_sources = set()
			if res.get("success") and res.get("videos"):
				for v in res["videos"]:
					if v.get("success"):
						docs.append(v)
						planned_sources.add(v.get("video_id") or "unknown")
			elif res.get("success"):
				docs.append(res)
				planned_sources.add(res.get("video_id") or "unknown")
			if docs:
				_maybe_clear_index(planned_sources)
				os.makedirs(_store_dir(), exist_ok=True)
				ingest_documents(docs, store_dir=_store_dir(), model_name=embed_model)
		st.success("Ingested.")

st.markdown("---")

st.subheader("Ask a question")
query = st.text_input("Your question")
if st.button("Answer") and query:
	with st.spinner("Retrieving and answering..."):
		contexts = mmr_retrieve(query, top_k=top_k, store_dir=_store_dir(), model_name=embed_model)
		if not contexts:
			st.warning("No results found. Ingest content first.")
		else:
			answer = answer_with_context(query, contexts)
			st.markdown("### Answer")
			st.write(answer)
			st.markdown("### Sources")
			for c in contexts:
				st.caption(f"{c.get('source')} (chunk {c.get('chunk_index')})")

st.markdown("---")
st.subheader("📝 Generate Lecture-Style Notes")
coln1, coln2 = st.columns([3, 1])
with coln1:
	custom_title = st.text_input("Notes Title (optional)")
with coln2:
	notes_chunk_size = st.number_input("Chunk size", min_value=400, max_value=4000, value=1200, step=100)

if st.button("Start Note-Making"):
	with st.spinner("Generating notes..."):
		# Gather all currently ingested text from vector store metadata
		store_meta = _meta_path()
		if not os.path.exists(store_meta):
			st.warning("No ingested content found. Ingest PDFs/DOCs/YouTube first.")
		else:
			with open(store_meta, "r", encoding="utf-8") as f:
				metas = json.load(f)
			# Sort by source, then chunk_index for a stable reading flow
			metas_sorted = sorted(metas, key=lambda m: (m.get("source", ""), int(m.get("chunk_index", 0))))
			texts = [m.get("text", "") for m in metas_sorted]

			st.markdown("### Notes (Generating)")
			placeholder = st.empty()
			col_a, col_b = st.columns([3,1])
			with col_b:
				group_size = st.number_input("Chunks/group", min_value=1, max_value=10, value=3)

			sections = []
			for idx, sec in enumerate(iter_generate_notes_from_texts(texts, title=custom_title, group_size=int(group_size)), 1):
				sections.append(sec)
				with placeholder.container():
					for i, s in enumerate(sections, 1):
						with st.expander(f"Section {i}", expanded=(i == idx)):
							st.markdown(s)
			st.success("Notes generated.")
