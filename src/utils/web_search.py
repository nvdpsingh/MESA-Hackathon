import html
import re
import urllib.parse
from typing import List, Dict

import requests


def recommend_articles_ddg(topic: str, limit: int = 3) -> List[Dict[str, str]]:
	"""Fetch simple article links from DuckDuckGo HTML endpoint.
	Returns a list of {title, url}. Best-effort HTML parse without extra deps.
	"""
	q = urllib.parse.quote(topic)
	url = f"https://duckduckgo.com/html/?q={q}+tutorial"
	try:
		r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
		r.raise_for_status()
		html_text = r.text
		# crude parse: results are links in <a class="result__a" href="...">Title</a>
		items: List[Dict[str, str]] = []
		for m in re.finditer(r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html_text, flags=re.I|re.S):
			url = html.unescape(m.group(1))
			title = re.sub(r"<.*?>", "", html.unescape(m.group(2))).strip()
			if title and url and url.startswith("http"):
				items.append({"title": title, "url": url})
				if len(items) >= limit:
					break
		return items
	except Exception:
		return []


def recommend_youtube_ddg(topic: str, limit: int = 3) -> List[Dict[str, str]]:
	"""Fetch YouTube video links from DuckDuckGo search.
	Returns a list of {title, url}. Searches specifically for YouTube videos.
	"""
	q = urllib.parse.quote(f"{topic} site:youtube.com")
	url = f"https://duckduckgo.com/html/?q={q}"
	try:
		r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
		r.raise_for_status()
		html_text = r.text
		# Parse YouTube links
		items: List[Dict[str, str]] = []
		for m in re.finditer(r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html_text, flags=re.I|re.S):
			url = html.unescape(m.group(1))
			title = re.sub(r"<.*?>", "", html.unescape(m.group(2))).strip()
			# Filter for YouTube URLs
			if title and url and "youtube.com" in url and ("watch?v=" in url or "youtu.be/" in url):
				items.append({"title": title, "url": url})
				if len(items) >= limit:
					break
		return items
	except Exception:
		return []
