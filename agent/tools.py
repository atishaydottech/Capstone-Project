# The agent's two tools: retriever_tool hits the local Chroma corpus,
# web_search (Tavily) is the fallback when a program isn't in there.

from __future__ import annotations

import os
import sys

import requests
from pydantic_ai import ModelRetry

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import COLLECTION_NAME, PERSIST_DIR
from shared.embedder import embed_query
from vectordb.chroma_store import ChromaStore

_store = None


def _get_store():
    global _store
    if _store is None:
        _store = ChromaStore(collection_name=COLLECTION_NAME, persist_dir=PERSIST_DIR)
    return _store


def retriever_tool(query: str, k: int = 5) -> list[dict]:
    """Search GrantMatch's local corpus of scholarship/grant program rules.

    Use this FIRST for any eligibility question. Returns the top matching rule
    excerpts, each with its source program, section, and similarity score, so
    the answer can quote the exact clause. An empty list means the program
    isn't in the local corpus — fall back to web_search.

    Args:
        query: What to look up, e.g. "Cal Grant A GPA requirement".
        k: How many chunks to return (default 5).
    """
    store = _get_store()
    query_embedding = embed_query(query)
    return store.search(query_embedding, k=k)


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for a program's rules when it isn't in the local corpus.

    Args:
        query: What to search for, e.g. "Ohio War Orphans Scholarship eligibility".
        max_results: How many results to return (default 5).
    """
    api_key = os.environ.get("SEARCH_API_KEY")
    if not api_key:
        raise ModelRetry("SEARCH_API_KEY not set — cannot search the web.")

    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query, "max_results": max_results},
            timeout=10,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
    except Exception as e:
        # Transient (rate limit, network) — let the model retry or rephrase.
        raise ModelRetry(f"Web search failed ({type(e).__name__}). Try again or rephrase the query.")

    if not results:
        return f"No web results for {query!r}."
    return "\n".join(f"- {r.get('title', '')}: {r.get('content', '')}" for r in results)
