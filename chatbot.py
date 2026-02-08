#!/usr/bin/env python3
"""
Chainlit chatbot for the Gemini Web Search Agent.
Run with: chainlit run chatbot.py
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

import asyncio
import chainlit as cl
from agent import get_client_or_none, search_with_sources


@cl.on_chat_start
async def on_chat_start():
    """Initialize the agent when a chat session starts."""
    client = get_client_or_none()
    if client is None:
        await cl.Message(
            content="⚠️ **API key not configured.** Set `GEMINI_API_KEY` or `GOOGLE_API_KEY` in your environment or `.env` file, then restart the app."
        ).send()
        return

    cl.user_session.set("client", client)
    await cl.Message(
        content="🔍 **Gemini Web Search Agent** — Ask anything. Responses are grounded in real-time web search with citations."
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages and respond with web-grounded answers."""
    client = cl.user_session.get("client")
    if client is None:
        # Re-check in case key was set after chat started (e.g. server restart)
        client = get_client_or_none()
        if client:
            cl.user_session.set("client", client)
        else:
            await cl.Message(
                content="API key not found. Set `GEMINI_API_KEY` in `.env`, then **start a new chat** (click + in the sidebar) or restart the app."
            ).send()
            return

    query = (message.content or "").strip()
    if not query:
        await cl.Message(content="Please enter a question.").send()
        return

    msg = cl.Message(content="")
    await msg.send()

    try:
        text, sources, _ = await asyncio.to_thread(
            search_with_sources, client, query, True
        )

        # Build response with optional sources section
        if text:
            response_parts = [text]
            if sources:
                response_parts.append("\n\n**Sources:**")
                for i, src in enumerate(sources, 1):
                    response_parts.append(f"- [{src['title']}]({src['uri']})")
            msg.content = "\n".join(response_parts)
        else:
            msg.content = "No response available."

        await msg.update()
    except Exception as e:
        await cl.Message(content=f"Error: {str(e)}").send()
