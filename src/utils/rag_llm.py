"""
RAG LLM wrapper.

Defaults to local open-source models via Ollama (Llama/Mistral/etc.).
Fallback to Groq/OpenAI-compatible if OLLAMA is not desired.
"""

import os
import time
from typing import List, Dict, Any


def _format_context(contexts: List[Dict[str, Any]]) -> str:
    formatted: List[str] = []
    for c in contexts:
        text = c.get("text", "")
        # Truncate to keep prompts small and reduce rate limits
        if len(text) > 1500:
            text = text[:1500]
        formatted.append(f"[source: {c.get('source')} chunk: {c.get('chunk_index')}]\n{text}")
    return "\n\n".join(formatted)


def _answer_with_ollama(query: str, contexts: List[Dict[str, Any]], model: str | None) -> str:
    try:
        import ollama  # type: ignore
    except Exception as e:
        raise RuntimeError("Ollama python client not installed. Run: pip install ollama") from e

    model_name = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    context_text = _format_context(contexts)

    system_prompt = (
        "You are an assistant that answers strictly using the provided context. "
        "If the answer cannot be found in the context, say 'I cannot find this in the provided documents.'"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}\nProvide a concise answer with citations."},
    ]

    resp = ollama.chat(model=model_name, messages=messages, options={"temperature": 0.2})
    return resp.get("message", {}).get("content", "")


def _answer_with_openai_compatible(query: str, contexts: List[Dict[str, Any]], model: str | None) -> str:
    from openai import OpenAI
    from openai import BadRequestError

    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY/OPENAI_API_KEY for OpenAI-compatible backend")
    base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=20, max_retries=0)
    model_name = model or os.getenv("GROQ_MODEL")
    # If no model provided, try to auto-select a current Llama model from Groq
    if not model_name:
        try:
            models = client.models.list()
            # Prefer larger llama, then smaller
            llama_models = [m.id for m in getattr(models, "data", []) if "llama" in getattr(m, "id", "")]
            preferred = sorted(
                llama_models,
                key=lambda x: ("70" not in x, "33" not in x, "8" not in x, x),
            )
            if preferred:
                model_name = preferred[0]
        except Exception:
            pass
    # Fallback default if discovery failed
    if not model_name:
        model_name = "llama-3.1-8b-instant"
    context_text = _format_context(contexts)

    system_prompt = (
        "You are an assistant that answers strictly using the provided context. "
        "If the answer cannot be found in the context, say 'I cannot find this in the provided documents.'"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}\nProvide a concise answer with citations."},
    ]

    # Retry with exponential backoff on 429 (short)
    backoff = 1.5
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.2,
                max_tokens=350,
            )
            return resp.choices[0].message.content or ""
        except BadRequestError as e:
            # If model was decommissioned, attempt discovery once
            if "decommissioned" in str(e).lower():
                try:
                    models = client.models.list()
                    llama_models = [m.id for m in getattr(models, "data", []) if "llama" in getattr(m, "id", "")]
                    preferred = sorted(
                        llama_models,
                        key=lambda x: ("70" not in x, "33" not in x, "8" not in x, x),
                    )
                    if preferred and preferred[0] != model_name:
                        model_name = preferred[0]
                        continue  # retry loop will call again with new model
                except Exception:
                    pass
            msg = str(e)
            if "429" in msg or "Too Many Requests" in msg:
                if attempt < 2:
                    time.sleep(backoff)
                    backoff *= 1.5
                    continue
                raise RuntimeError("Rate limited by LLM (Groq). Please retry shortly.")
            raise
        except Exception as e:
            msg = str(e)
            if "429" in msg or "Too Many Requests" in msg:
                if attempt < 2:
                    time.sleep(backoff)
                    backoff *= 1.5
                    continue
                raise RuntimeError("Rate limited by LLM (Groq). Please retry shortly.")
            raise
    # final fallback raise if all retries exhausted
    raise RuntimeError("LLM request failed after retries (Groq-compatible)")


def _answer_with_openai_native(query: str, contexts: List[Dict[str, Any]], model: str | None) -> str:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY for OpenAI backend")
    client = OpenAI(api_key=api_key, timeout=20, max_retries=0)

    # Allow override via model arg or env OPENAI_MODEL; default to a small, cost-effective model
    model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    context_text = _format_context(contexts)

    system_prompt = (
        "You are an assistant that answers strictly using the provided context. "
        "If the answer cannot be found in the context, say 'I cannot find this in the provided documents.'"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}\nProvide a concise answer with citations."},
    ]

    backoff = 1.5
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.2,
                max_tokens=350,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                if attempt < 2:
                    time.sleep(backoff)
                    backoff *= 1.5
                    continue
                raise RuntimeError("Rate limited by LLM (OpenAI). Please retry shortly.")
            raise
    raise RuntimeError("LLM request failed after retries (OpenAI)")


def answer_with_context(query: str, contexts: List[Dict[str, Any]], model: str | None = None) -> str:
    # Backend resolution:
    # 1) If LLM_BACKEND explicitly set, honor it ("ollama" | "groq" | "openai")
    # 2) Else, if GROQ_API_KEY present, prefer Groq (openai-compatible)
    # 3) Else, if OPENAI_API_KEY present, use OpenAI native
    # 4) Else, fallback to Ollama
    explicit = os.getenv("LLM_BACKEND")
    if explicit:
        backend = explicit.lower()
    else:
        if os.getenv("GROQ_API_KEY"):
            backend = "groq"
        elif os.getenv("OPENAI_API_KEY"):
            backend = "openai"
        else:
            backend = "ollama"

    if backend == "ollama":
        return _answer_with_ollama(query, contexts, model)
    if backend == "openai":
        return _answer_with_openai_native(query, contexts, model)
    # groq (openai-compatible)
    try:
        return _answer_with_openai_compatible(query, contexts, model)
    except RuntimeError as e:
        # Optional fallback to Ollama if available
        if "Rate limited" in str(e) and os.getenv("OLLAMA_FALLBACK", "1") == "1":
            try:
                return _answer_with_ollama(query, contexts, model)
            except Exception:
                pass
        raise


