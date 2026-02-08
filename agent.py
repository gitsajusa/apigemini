#!/usr/bin/env python3
"""
Gemini Web Search Agent — General web search powered by Google Gemini with grounding.
"""

import os
import sys
from typing import Optional

try:
    from dotenv import load_dotenv
    # Load from project dir so .env is found even when run from elsewhere (e.g. chainlit)
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

from google import genai
from google.genai import types


def get_client_or_none() -> Optional[genai.Client]:
    """Initialize the Gemini client. Returns None if API key is not set."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def get_client() -> genai.Client:
    """Initialize the Gemini client with API key from environment."""
    client = get_client_or_none()
    if client is None:
        print(
            "Error: GEMINI_API_KEY or GOOGLE_API_KEY not set. Add to .env or export:\n"
            "  export GEMINI_API_KEY=your_api_key"
        )
        sys.exit(1)
    return client


def add_citations(response) -> str:
    """Add inline citations to the response text using grounding metadata."""
    text = response.text or ""
    try:
        candidate = response.candidates[0]
        grounding = getattr(candidate, "grounding_metadata", None)
        if not grounding:
            return text

        supports = getattr(grounding, "grounding_supports", None) or []
        chunks = getattr(grounding, "grounding_chunks", None) or []

        if not supports or not chunks:
            return text

        sorted_supports = sorted(
            supports, key=lambda s: getattr(s.segment, "end_index", 0) or 0, reverse=True
        )

        for support in sorted_supports:
            segment = getattr(support, "segment", None)
            if not segment:
                continue
            end_index = getattr(segment, "end_index", None)
            if end_index is None:
                continue

            chunk_indices = getattr(support, "grounding_chunk_indices", None) or []
            if not chunk_indices:
                continue

            citation_links = []
            for i in chunk_indices:
                if i < len(chunks):
                    chunk = chunks[i]
                    web = getattr(chunk, "web", None)
                    if web:
                        uri = getattr(web, "uri", None)
                        if uri:
                            citation_links.append(f"[{i + 1}]({uri})")

            if citation_links:
                citation_string = ", ".join(citation_links)
                text = text[:end_index] + citation_string + text[end_index:]

        return text
    except (IndexError, AttributeError):
        return text


def search(client: genai.Client, query: str, show_citations: bool = True) -> str:
    """
    Run a web search using Gemini with grounding and return the response.
    """
    text, _, _ = search_with_sources(client, query, show_citations)
    return text


def search_with_sources(
    client: genai.Client, query: str, show_citations: bool = True
) -> tuple[str, list[dict], list[str]]:
    """
    Run a web search using Gemini with grounding.
    Returns (text, sources, queries). sources: [{"title": str, "uri": str}].
    """
    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(tools=[grounding_tool])

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=query,
        config=config,
    )

    text = add_citations(response) if show_citations else (response.text or "")
    sources = _extract_sources(response)
    queries = _get_grounding_queries(response)
    return text, sources, queries


def _extract_sources(response) -> list[dict]:
    """Extract source links from grounding metadata."""
    sources = []
    try:
        candidate = response.candidates[0]
        grounding = getattr(candidate, "grounding_metadata", None)
        if not grounding:
            return sources

        chunks = getattr(grounding, "grounding_chunks", None) or []
        for chunk in chunks:
            web = getattr(chunk, "web", None)
            if web:
                uri = getattr(web, "uri", "")
                title = getattr(web, "title", "")
                if uri:
                    sources.append({"title": title or f"Source {len(sources) + 1}", "uri": uri})
    except (IndexError, AttributeError):
        pass
    return sources


def _get_grounding_queries(response) -> list[str]:
    """Extract search queries from grounding metadata."""
    try:
        grounding = getattr(response.candidates[0], "grounding_metadata", None)
        return getattr(grounding, "web_search_queries", None) or []
    except (IndexError, AttributeError):
        return []


def _print_sources_data(sources: list[dict], queries: list[str]):
    """Print sources and queries to stdout."""
    if queries:
        print("\n📋 Search queries used:", ", ".join(queries))
    if sources:
        print("\n📎 Sources:")
        for i, src in enumerate(sources, 1):
            print(f"  [{i}] {src['title']}: {src['uri']}")


def print_sources(response):
    """Print source links from grounding metadata."""
    _print_sources_data(_extract_sources(response), _get_grounding_queries(response))


def main():
    """Interactive agent loop for general web search."""
    client = get_client()

    print("🔍 Gemini Web Search Agent")
    print("Ask anything — responses are grounded in real-time web search.")
    print("Commands: quit, exit, sources on/off")
    print("-" * 50)

    show_sources = True

    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if query.lower() == "sources on":
            show_sources = True
            print("Sources will be shown.")
            continue
        if query.lower() == "sources off":
            show_sources = False
            print("Sources will be hidden.")
            continue

        try:
            text, sources, queries = search_with_sources(client, query, show_citations=True)
            print("\nAgent:", text)
            if show_sources:
                _print_sources_data(sources, queries)

        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
