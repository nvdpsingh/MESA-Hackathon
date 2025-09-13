import os
import tempfile
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


with st.sidebar:
	st.header("Settings")
	embed_model = st.text_input("Embedding model", value="all-MiniLM-L6-v2")
	top_k = st.number_input("Top K", min_value=1, max_value=20, value=5)
	backend = st.selectbox("LLM Backend", options=["groq (OpenAI-compatible)", "openai", "ollama"], index=0)
	groq_key = None
	groq_model = None
	opnai_key = None
	opnai_model = None
	if backend.startswith("groq"):
		st.caption("Uses GROQ_API_KEY from .env (or paste below)")
		groq_key = st.text_input("GROQ_API_KEY (optional)", type="password")
		groq_model = st.text_input("GROQ_MODEL (optional)", value=os.getenv("GROQ_MODEL", ""), help="Enter Groq model id, e.g., a current Llama or your OSS 20B id")
	elif backend == "openai":
		st.caption("Uses OPENAI_API_KEY; set model in field below (e.g., gpt-4o-mini or an OSS 20B served via OpenAI if available)")
		opnai_key = st.text_input("OPENAI_API_KEY (optional)", type="password")
		opnai_model = st.text_input("OPENAI_MODEL", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
	else:
		st.caption("Requires local ollama service and OLLAMA_MODEL (default llama3.1:8b)")

	# Allow overriding
	# Dataset isolation
	dataset = st.text_input("Dataset name", value=st.session_state.get("dataset", "default"))
	store_dir = os.path.join("data", "vector_store", dataset)
	st.caption(f"Active dataset: {dataset} → {store_dir}")
	st.session_state["dataset"] = dataset
	if backend.startswith("groq"):
		os.environ["LLM_BACKEND"] = "groq"
		if groq_key:
			os.environ["GROQ_API_KEY"] = groq_key
		if groq_model:
			os.environ["GROQ_MODEL"] = groq_model
	elif backend == "openai":
		os.environ["LLM_BACKEND"] = "openai"
		if opnai_key:
			os.environ["OPENAI_API_KEY"] = opnai_key
		if opnai_model:
			os.environ["OPENAI_MODEL"] = opnai_model
	else:
		os.environ["LLM_BACKEND"] = "ollama"

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
			for f in uploaded:
				path = os.path.join(tmpdir, f.name)
				with open(path, "wb") as out:
					out.write(f.read())
				res = pipe.extract_from_file(path)
				if res.get("success"):
					results.append(res)
			_store = ingest_documents(results, store_dir=store_dir, model_name=embed_model)
		st.success("Ingested.")

with col2:
	st.subheader("Ingest YouTube URL / Playlist")
	yt_url = st.text_input("YouTube URL or Playlist")
	if st.button("Ingest YouTube") and yt_url:
		with st.spinner("Fetching transcript(s) and ingesting..."):
			pipe = TextExtractionPipeline()
			res = pipe.extract_from_youtube(yt_url)
			docs = []
			if res.get("success") and res.get("videos"):
				for v in res["videos"]:
					if v.get("success"):
						docs.append(v)
			elif res.get("success"):
				docs.append(res)
			if docs:
				ingest_documents(docs, store_dir=store_dir, model_name=embed_model)
		st.success("Ingested.")

st.markdown("---")

st.subheader("Ask a question")
query = st.text_input("Your question")
if st.button("Answer") and query:
	with st.spinner("Retrieving and answering..."):
		contexts = mmr_retrieve(query, top_k=top_k, store_dir=store_dir, model_name=embed_model)
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
		store_meta = os.path.join(store_dir, "meta.json")
		if not os.path.exists(store_meta):
			st.warning("No ingested content found. Ingest PDFs/DOCs/YouTube first.")
		else:
			import json
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
