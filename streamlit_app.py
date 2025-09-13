import os
import time
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

# Memory profile selection
st.sidebar.subheader("Profile & Memory")
profile_id = st.sidebar.text_input("Profile ID", value="default")
memory_short_window = st.sidebar.slider("Short-term window", 4, 12, 6)

from src.utils.memory import MemoryStore
mem = MemoryStore()

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

# Auto backend from environment and show a small caption
if os.getenv("GROQ_API_KEY"):
	os.environ["LLM_BACKEND"] = "groq"
	st.caption(f"Backend: Groq (GROQ_MODEL={os.getenv('GROQ_MODEL','auto')})")
elif os.getenv("OPENAI_API_KEY"):
	os.environ["LLM_BACKEND"] = "openai"
	st.caption(f"Backend: OpenAI (OPENAI_MODEL={os.getenv('OPENAI_MODEL','gpt-4o-mini')})")
else:
	os.environ["LLM_BACKEND"] = "ollama"
	st.caption(f"Backend: Ollama (model={os.getenv('OLLAMA_MODEL','llama3.1:8b')})")

# Defaults (hidden from UI)
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "3"))

st.markdown("---")

# Ingestion controls
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
			_store = ingest_documents(results, store_dir=_store_dir(), model_name=EMBED_MODEL)
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
				ingest_documents(docs, store_dir=_store_dir(), model_name=EMBED_MODEL)
		st.success("Ingested.")

st.markdown("---")

# Layout: Notes (main) and QA Chatbot (side)
# Replace columns with tabs
if True:
	tab_notes, tab_chat, tab_learning, tab_resources = st.tabs(["📝 Generate Notes", "🤖 QA Chatbot", "🎓 Learning Mode", "🔗 Recommended Resources"])

	with tab_notes:
		st.subheader("Generate Lecture-Style Notes")
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
					# Ingest sections into separate notes index
					try:
						from src.utils.notes_ingest import ingest_notes_sections
						ingest_notes_sections(sections, store_dir=os.path.join("data","notes_index"), model_name=EMBED_MODEL)
						st.caption("Notes embedded into Notes Index for Guided Learning.")
					except Exception as e:
						st.warning(f"Failed to embed notes into Notes Index: {e}")

	with tab_chat:
		st.subheader("🤖 QA Chatbot")
		# Initialize chat history
		if "chat_messages" not in st.session_state:
			st.session_state.chat_messages = []  # list of dicts: {role, content}
		if "chat_cooldown" not in st.session_state:
			st.session_state.chat_cooldown = 0.0

		# Chat container
		chat_container = st.container()
		
		# Display chat history in scrollable container
		with chat_container:
			for msg in st.session_state.chat_messages:
				if msg.get("role") == "user":
					st.markdown(f"**You:** {msg.get('content','')}")
				else:
					st.markdown(f"**Assistant:** {msg.get('content','')}")

		# Chat input at bottom
		st.markdown("---")
		col1, col2 = st.columns([4, 1])
		with col1:
			user_q = st.text_input("Ask a question about your content...", key="chat_input", placeholder="Type your question here...")
		with col2:
			send_button = st.button("Send", key="chat_send", type="primary")
		
		if send_button and user_q:
			# simple cooldown to avoid hammering Groq
			now = time.time()
			if now - st.session_state.chat_cooldown < 3.0:
				st.warning("Please wait a moment before sending another message.")
				st.stop()
			with st.spinner("Retrieving and answering..."):
				# Retrieval contexts
				contexts = mmr_retrieve(user_q, top_k=TOP_K, store_dir=_store_dir(), model_name=EMBED_MODEL)
				# Memory contexts
				mem_ctx = mem.memory_contexts(st.session_state.chat_messages, profile_id=profile_id, short_window=memory_short_window)
				all_ctx = contexts + mem_ctx
				if not all_ctx:
					answer = "No results found. Please ingest content first."
				else:
					try:
						answer = answer_with_context(user_q, all_ctx)
					except Exception as e:
						answer = f"Rate-limited or error from LLM. Please wait a bit and retry. ({e})"
				st.session_state.chat_messages.append({"role": "user", "content": user_q})
				st.session_state.chat_messages.append({"role": "assistant", "content": answer})
				# Try to update long-term memory occasionally
				_ = mem.summarize_and_store_long_term(st.session_state.chat_messages, profile_id=profile_id)
				st.session_state.chat_cooldown = time.time()
				st.rerun()

	with tab_learning:
		st.subheader("🎓 Learning Mode - AI Tutor Chat")
		
		# Check if notes are available
		store_meta = _meta_path()
		if not os.path.exists(store_meta):
			st.warning("No notes context found. Generate notes first.")
		else:
			# Initialize learning chat
			if "learning_messages" not in st.session_state:
				st.session_state.learning_messages = []
			if "learning_cooldown" not in st.session_state:
				st.session_state.learning_cooldown = 0.0
			if "learning_initialized" not in st.session_state:
				st.session_state.learning_initialized = False

			# Initialize tutor with topics from notes
			if not st.session_state.learning_initialized:
				from src.utils.learning_mode import extract_topics_from_notes
				with open(store_meta, "r", encoding="utf-8") as f:
					metas = json.load(f)
				texts = [m.get("text", "") for m in metas if m.get("text")]
				notes_sections = texts[:50]
				topics = extract_topics_from_notes(notes_sections)
				if not topics:
					topics = [f"Topic {i+1}" for i in range(min(5, len(notes_sections)))]
				
				st.session_state.learning_topics = topics[:10]
				st.session_state.learning_initialized = True
				
				# Welcome message
				welcome_msg = f"Hello! I'm your AI learning tutor. I can help you learn about these topics: {', '.join(topics[:5])}. What would you like to explore first?"
				st.session_state.learning_messages.append({"role": "assistant", "content": welcome_msg})

			# Chat container
			chat_container = st.container()
			
			# Display chat history
			with chat_container:
				for msg in st.session_state.learning_messages:
					if msg.get("role") == "user":
						st.markdown(f"**You:** {msg.get('content','')}")
					else:
						st.markdown(f"**Tutor:** {msg.get('content','')}")

			# Chat input at bottom
			st.markdown("---")
			col1, col2 = st.columns([4, 1])
			with col1:
				user_input = st.text_input("Ask your tutor anything...", key="learning_input", placeholder="What would you like to learn about?")
			with col2:
				send_button = st.button("Send", key="learning_send", type="primary")
			
			if send_button and user_input:
				# Cooldown check
				now = time.time()
				if now - st.session_state.learning_cooldown < 3.0:
					st.warning("Please wait a moment before sending another message.")
					st.stop()
				
				with st.spinner("Tutor is thinking..."):
					# Build context from Notes Index
					from src.utils.ingest import semantic_search
					note_ctx = semantic_search(user_input, store_dir=os.path.join("data","notes_index"), top_k=TOP_K, model_name=EMBED_MODEL)
					
					# Memory context
					mem_ctx = mem.memory_contexts(st.session_state.learning_messages, profile_id=profile_id, short_window=memory_short_window)
					
					# Get recommended resources
					from src.utils.learning_mode import recommend_youtube
					from src.utils.web_search import recommend_articles_ddg
					
					# Find relevant topic for resources
					relevant_topic = user_input
					for topic in st.session_state.learning_topics:
						if topic.lower() in user_input.lower() or user_input.lower() in topic.lower():
							relevant_topic = topic
							break
					
					vlinks = recommend_youtube(relevant_topic)
					alinks = recommend_articles_ddg(relevant_topic)
					res_text = f"Recommended Videos for '{relevant_topic}':\n" + "\n".join([f"- {l['title']}: {l['url']}" for l in vlinks[:2]]) + \
								  f"\nRecommended Articles for '{relevant_topic}':\n" + "\n".join([f"- {a['title']}: {a['url']}" for a in alinks[:2]])
					res_ctx = [{"source": "resources", "chunk_index": 0, "text": res_text}]
					
					all_ctx = note_ctx + mem_ctx + res_ctx
					
					# Tutor prompt
					prompt = f"You are an AI learning tutor. Help the student learn by explaining concepts, asking questions, suggesting resources, and giving small assignments. Be encouraging and educational. Student said: {user_input}"
					
					try:
						reply = answer_with_context(prompt, all_ctx)
					except Exception as e:
						reply = f"I'm having trouble connecting right now. Please try again in a moment. Error: {e}"
					
					st.session_state.learning_messages.append({"role": "user", "content": user_input})
					st.session_state.learning_messages.append({"role": "assistant", "content": reply})
					
					# Update long-term memory
					_ = mem.summarize_and_store_long_term(st.session_state.learning_messages, profile_id=profile_id)
					st.session_state.learning_cooldown = time.time()
					st.rerun()

	with tab_resources:
		st.subheader("🔗 Recommended Resources")
		# Build recommendations per detected topic from notes
		store_meta = _meta_path()
		if not os.path.exists(store_meta):
			st.warning("No notes context found. Generate notes first.")
		else:
			from src.utils.learning_mode import extract_topics_from_notes
			from src.utils.web_search import recommend_articles_ddg, recommend_youtube_ddg
			with open(store_meta, "r", encoding="utf-8") as f:
				metas = json.load(f)
			texts = [m.get("text", "") for m in metas if m.get("text")]
			notes_sections = texts[:50]
			topics = extract_topics_from_notes(notes_sections)
			if not topics:
				st.info("Could not detect headings; using first few sections as topics.")
				topics = [f"Topic {i+1}" for i in range(min(5, len(notes_sections)))]
			for topic in topics[:10]:
				with st.expander(topic, expanded=False):
					st.markdown("**Related Videos (via DuckDuckGo)**")
					vlinks = recommend_youtube_ddg(topic)
					if vlinks:
						for l in vlinks:
							st.write(f"- [{l['title']}]({l['url']})")
					else:
						st.write("No videos found for this topic.")
					
					st.markdown("**Related Articles (via DuckDuckGo)**")
					alinks = recommend_articles_ddg(topic)
					if alinks:
						for a in alinks:
							st.write(f"- [{a['title']}]({a['url']})")
					else:
						st.write("No articles found for this topic.")
					
					if st.button("Explain connections", key=f"explain_{topic}"):
						# Optional concise explanation using LLM with notes context
						from src.utils.learning_mode import build_topic_context
						topic_ctx = build_topic_context(notes_sections, topic)
						# Append resource titles to context to guide the mapping
						res_text = "Videos:\n" + "\n".join([l['title'] for l in vlinks]) + "\nArticles:\n" + "\n".join([a['title'] for a in alinks])
						aug_ctx = topic_ctx + [{"source": "resources", "chunk_index": 0, "text": res_text}]
						prompt = f"For the topic '{topic}', explain in 2-3 sentences how the above videos and articles help learning the topic (what each adds)."
						try:
							msg = answer_with_context(prompt, aug_ctx)
							st.write(msg)
						except Exception as e:
							st.info(f"(Optional) Could not generate explanation now: {e}")
