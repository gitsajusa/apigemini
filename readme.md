# Gemini Web Search Agent

An interactive agent app powered by Google Gemini with real-time web search grounding. Ask questions and get answers grounded in current web data with citations.

## Features

- **Real-time web search** — Gemini automatically searches the web when it needs fresh information
- **Verifiable citations** — Inline source links and grounding metadata
- **Interactive CLI** — Chat-style interface for continuous queries
- **Chainlit chatbot** — Web UI for the agent

## Setup

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Get an API key** from [Google AI Studio](https://aistudio.google.com/apikey)

3. **Configure the API key**

   ```bash
   cp .env.example .env
   # Edit .env and add your GEMINI_API_KEY
   ```

   Or export directly:

   ```bash
   export GEMINI_API_KEY=your_api_key
   ```

## Usage

### CLI agent

```bash
python agent.py
```

### Chainlit chatbot (web UI)

```bash
chainlit run chatbot.py
```

Then open http://localhost:8000 in your browser.

### CLI commands

- Type any question to search
- `sources on` / `sources off` — Toggle display of source links
- `quit` / `exit` / `q` — Exit the agent

### Example queries

- "Who won the most recent Super Bowl?"
- "What's the latest news about AI?"
- "Current weather in Tokyo today"

## Programmatic use

```python
from agent import get_client, search

client = get_client()
answer = search(client, "Who won Euro 2024?")
print(answer)
```

## Pricing

Grounding with Google Search uses the Gemini API. See [official pricing](https://ai.google.dev/gemini-api/docs/pricing) for details.
