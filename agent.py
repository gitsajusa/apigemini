#!/usr/bin/env python3
"""
Gemini Web Search Agent — General web search powered by Google Gemini with grounding.
"""

import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from google import genai
from google.genai import types


def get_client() -> genai.Client:
    """Initialize the Gemini client with API key from environment."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print(
            "Error: GEMINI_API_KEY or GOOGLE_API_KEY not set. Add to .env or export:\n"
            "  export GEMINI_API_KEY=your_api_key"
        )
        sys.exit(1)
    return genai.Client(api_key=api_key)


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
    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(tools=[grounding_tool])

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=query,
        config=config,
    )

    if show_citations:
        return add_citations(response)

    return response.text or ""


def print_sources(response):
    """Print source links from grounding metadata."""
    try:
        candidate = response.candidates[0]
        grounding = getattr(candidate, "grounding_metadata", None)
        if not grounding:
            return

        chunks = getattr(grounding, "grounding_chunks", None) or []
        queries = getattr(grounding, "web_search_queries", None) or []

        if queries:
            print("\n📋 Search queries used:", ", ".join(queries))

        if chunks:
            print("\n📎 Sources:")
            for i, chunk in enumerate(chunks, 1):
                web = getattr(chunk, "web", None)
                if web:
                    uri = getattr(web, "uri", "")
                    title = getattr(web, "title", f"Source {i}")
                    print(f"  [{i}] {title}: {uri}")
    except (IndexError, AttributeError):
        pass


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
            grounding_tool = types.Tool(google_search=types.GoogleSearch())
            config = types.GenerateContentConfig(tools=[grounding_tool])

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=query,
                config=config,
            )

            text = add_citations(response)
            print("\nAgent:", text)

            if show_sources:
                print_sources(response)

        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
