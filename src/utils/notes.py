"""
Lecture-article style notes generator.

Generates well-formatted markdown notes from raw text, chunk-by-chunk.
Code blocks are fenced, headings emphasized, and content is made
easy to follow for students.
"""

from typing import List, Dict, Any
import os

from .text_processing import chunk_text


NOTES_SYSTEM_PROMPT = (
    "You are an expert technical writer creating lecture-article notes for students. "
    "Only include content a diligent student would write in a notebook: crisp definitions, key formulas, step-by-step procedures, concise examples, short summaries, caveats, and essential code snippets. "
    "Exclude fluff, marketing, anecdotes, and repeated text.\n\n"
    "Formatting rules:\n"
    "- Use clear H2/H3 headings (##, ###)\n"
    "- Prefer short bullet lists\n"
    "- Put code in fenced blocks with language hints when apparent (```python, ```js, etc.)\n"
    "- Keep paragraphs short and focused\n"
    "- Do not hallucinate; only use the provided content.\n"
)


def _llm_markdown_ollama(chunk: str, title: str | None) -> str:
    import ollama  # type: ignore
    model_name = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    user_prompt = (
        (f"Title: {title}\n" if title else "")
        + "Create well-formatted lecture notes for the following content.\n\n"
        + chunk
    )
    messages = [
        {"role": "system", "content": NOTES_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    resp = ollama.chat(model=model_name, messages=messages, options={"temperature": 0.2})
    return resp.get("message", {}).get("content", "")


def _llm_markdown_openai_compatible(chunk: str, title: str | None) -> str:
    from openai import OpenAI

    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY/OPENAI_API_KEY for notes generation")
    base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    client = OpenAI(api_key=api_key, base_url=base_url)

    model_name = os.getenv("GROQ_MODEL")
    if not model_name:
        try:
            models = client.models.list()
            llama_models = [m.id for m in getattr(models, "data", []) if "llama" in getattr(m, "id", "")]
            model_name = sorted(llama_models)[0] if llama_models else "llama-3.1-8b-instant"
        except Exception:
            model_name = "llama-3.1-8b-instant"

    user_prompt = (
        (f"Title: {title}\n" if title else "")
        + "Create well-formatted lecture notes for the following content.\n\n"
        + chunk
    )
    messages = [
        {"role": "system", "content": NOTES_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.2,
        max_tokens=800,
    )
    return resp.choices[0].message.content or ""


def generate_notes_from_text(
    text: str,
    title: str | None = None,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> List[str]:
    """
    Generate markdown notes for the given text, chunk-by-chunk.
    Returns a list of markdown sections in order.
    """
    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        return []

    backend = os.getenv("LLM_BACKEND")
    if backend:
        backend = backend.lower()
    else:
        backend = "groq" if os.getenv("GROQ_API_KEY") else "ollama"

    notes_sections: List[str] = []
    for ch in chunks:
        if backend == "ollama":
            md = _llm_markdown_ollama(ch, title)
        else:
            md = _llm_markdown_openai_compatible(ch, title)
        notes_sections.append(md or "")
    return notes_sections


def iter_generate_notes_from_texts(
    texts: list[str],
    title: str | None = None,
    group_size: int = 3,
) -> "list[str]":
    """
    Yield markdown notes incrementally, processing the input texts in groups
    (default 3 chunks per group) to provide on-the-go generation.
    """
    backend = os.getenv("LLM_BACKEND")
    if backend:
        backend = backend.lower()
    else:
        backend = "groq" if os.getenv("GROQ_API_KEY") else "ollama"

    n = len(texts)
    for i in range(0, n, max(1, group_size)):
        group = texts[i : i + max(1, group_size)]
        content = "\n\n".join(group)
        if backend == "ollama":
            md = _llm_markdown_ollama(content, title)
        else:
            md = _llm_markdown_openai_compatible(content, title)
        yield md or ""


