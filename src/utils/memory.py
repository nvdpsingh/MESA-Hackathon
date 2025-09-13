import os
import json
import hashlib
from typing import List, Dict, Any, Optional

from .rag_llm import answer_with_context


class MemoryStore:
	"""Simple JSON-backed memory store for short-term and long-term memory."""

	def __init__(self, root_dir: str = "data/memory") -> None:
		self.root_dir = root_dir
		os.makedirs(self.root_dir, exist_ok=True)

	def _lt_path(self, profile_id: str) -> str:
		fname = f"long_term_{profile_id}.json"
		return os.path.join(self.root_dir, fname)

	def load_long_term(self, profile_id: str = "default") -> Dict[str, Any]:
		path = self._lt_path(profile_id)
		if not os.path.exists(path):
			return {"facts": [], "topic_progress": {}}
		try:
			with open(path, "r", encoding="utf-8") as f:
				return json.load(f)
		except Exception:
			return {"facts": [], "topic_progress": {}}

	def save_long_term(self, data: Dict[str, Any], profile_id: str = "default") -> None:
		path = self._lt_path(profile_id)
		with open(path, "w", encoding="utf-8") as f:
			json.dump(data, f, ensure_ascii=False, indent=2)

	@staticmethod
	def format_recent(messages: List[Dict[str, str]], window: int = 6) -> str:
		recent = messages[-window:]
		lines: List[str] = []
		for m in recent:
			role = m.get("role", "user")
			content = m.get("content", "")
			lines.append(f"{role}: {content}")
		return "\n".join(lines)

	def summarize_and_store_long_term(
		self,
		messages: List[Dict[str, str]],
		profile_id: str = "default",
		min_turns: int = 8,
		max_turns: int = 16,
	) -> Optional[List[str]]:
		"""Summarize recent conversation into persistent facts and merge them into long-term memory.
		Returns new facts if updated, else None. Uses the RAG LLM in context-only mode by passing conversation text as context.
		"""
		if len(messages) < min_turns:
			return None
		context_text = self.format_recent(messages, window=max_turns)
		contexts = [{
			"source": "conversation",
			"chunk_index": 0,
			"text": context_text,
		}]
		prompt = (
			"From the conversation, extract persistent student facts/preferences/goals and any difficulties. "
			"Output 3-6 concise bullet points. Avoid ephemeral content."
		)
		try:
			resp = answer_with_context(prompt, contexts)
		except Exception:
			return None
		# Parse bullets
		facts = []
		for line in resp.splitlines():
			l = line.strip().lstrip("- ")
			if l:
				facts.append(l)
		if not facts:
			return None
		lt = self.load_long_term(profile_id)
		existing = set(lt.get("facts", []))
		changed = False
		for f in facts:
			if f not in existing:
				existing.add(f)
				changed = True
		if changed:
			lt["facts"] = list(existing)
			self.save_long_term(lt, profile_id)
			return facts
		return None

	def memory_contexts(
		self,
		messages: List[Dict[str, str]],
		profile_id: str = "default",
		short_window: int = 6,
	) -> List[Dict[str, Any]]:
		"""Build contexts representing short-term and long-term memory."""
		contexts: List[Dict[str, Any]] = []
		st_text = self.format_recent(messages, window=short_window)
		if st_text:
			contexts.append({"source": "memory:short_term", "chunk_index": 0, "text": st_text})
		lt = self.load_long_term(profile_id)
		facts = lt.get("facts", [])
		if facts:
			contexts.append({"source": "memory:long_term", "chunk_index": 0, "text": "\n".join(f"- {f}" for f in facts)})
		return contexts
