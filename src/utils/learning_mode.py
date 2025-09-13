import os
import re
import json
import subprocess
from typing import List, Dict, Any

from .rag_llm import answer_with_context


def extract_topics_from_notes(notes_sections: List[str]) -> List[str]:
	"""Extract topic headings (## or ###) from markdown sections."""
	topics: List[str] = []
	pattern = re.compile(r"^#{2,3}\s+(.+)$")
	for sec in notes_sections:
		for line in sec.splitlines():
			m = pattern.match(line.strip())
			if m:
				t = m.group(1).strip()
				if t and t not in topics:
					topics.append(t)
	return topics


def _yt_search(query: str, limit: int = 3) -> List[Dict[str, str]]:
	"""Search YouTube without API key using yt-dlp ytsearch.
	Returns list of dicts with title and url.
	"""
	cmd = [
		"yt-dlp",
		f"ytsearch{limit}:{query}",
		"--dump-json",
		"--no-warnings",
		"--skip-download",
	]
	try:
		proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
		results: List[Dict[str, str]] = []
		for line in proc.stdout.strip().splitlines():
			try:
				data = json.loads(line)
				title = data.get("title", "")
				url = data.get("webpage_url") or (f"https://www.youtube.com/watch?v={data.get('id')}" if data.get('id') else None)
				if title and url:
					results.append({"title": title, "url": url})
			except json.JSONDecodeError:
				continue
		return results[:limit]
	except subprocess.CalledProcessError:
		return []


def build_topic_context(notes_sections: List[str], topic: str) -> List[Dict[str, Any]]:
	"""Build minimal context for LLM from notes related to a topic."""
	context_texts: List[str] = []
	for sec in notes_sections:
		if topic.lower() in sec.lower():
			context_texts.append(sec)
	if not context_texts:
		context_texts = notes_sections[:2]
	contexts = []
	for idx, t in enumerate(context_texts):
		contexts.append({"source": "notes", "chunk_index": idx, "text": t})
	return contexts


def generate_explainer(topic: str, notes_sections: List[str]) -> str:
	contexts = build_topic_context(notes_sections, topic)
	prompt = f"Explain the topic '{topic}' concisely for a student using the context. Emphasize definitions, key steps, and one short example."
	return answer_with_context(prompt, contexts)


def generate_quiz(topic: str, notes_sections: List[str]) -> str:
	contexts = build_topic_context(notes_sections, topic)
	prompt = f"Create 2 short quiz questions (bullet list) to test understanding of '{topic}'."
	return answer_with_context(prompt, contexts)


def generate_assignment(topic: str, notes_sections: List[str]) -> str:
	contexts = build_topic_context(notes_sections, topic)
	prompt = f"Give one small assignment (3-5 steps) for practicing '{topic}'."
	return answer_with_context(prompt, contexts)


def recommend_youtube(topic: str) -> List[Dict[str, str]]:
	q = f"{topic} tutorial"
	return _yt_search(q, limit=3)
