# Everyday AI Agent — LangChain + Groq

A general-purpose chat app with a Streamlit interface. It can answer normal questions and **choose to call real tools** for weather, news, and current facts through web search. A fresh local timestamp is supplied automatically on each model call.

## What is inside?

| Component | What it does |
|---|---|
| LangChain `create_agent` | Handles model/tool interaction loops |
| Groq `openai/gpt-oss-120b` | Generates responses and chooses tools |
| Tavily | Web search for weather, news, and current facts (requires a Tavily key) |
| LangGraph `InMemorySaver` | Multi-turn memory while the server is running |
| Streamlit | Browser-based chat interface |

**Note:** 'Hottest news' means *important recent results found by search*, **not** a validated trending/popularity ranking. Freshness and completeness depend on Tavily's sources. The agent provides links when results are available.

## Requirements

- Python 3.10+
- Groq API key: https://console.groq.com/keys
- Tavily API key (for live news/web search): https://app.tavily.com/
- Internet access when running the app

## Install & run

```bash
cd everyday_groq_agent
python -m venv .venv
```

Activate your environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install packages:

```bash
pip install -r requirements.txt
```

Make your local env file:

```bash
# macOS / Linux
cp .env.example .env

# Windows PowerShell
Copy-Item .env.example .env
```

Edit `.env`, replacing the example values with your API keys. **Never commit or share `.env`.**

For Paragon questions, follow [PARAGON_SETUP.md](PARAGON_SETUP.md) to configure the local index.

Start the app:

```bash
streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`.

## Try these prompts

- "What's the weather today?"
- "Will it rain tomorrow in Tokyo?"
- "What's the latest world news today? Include source links."
- "What are the latest technology headlines?"
- "Who is the current CEO of Microsoft?"
- "Explain what LangChain is, with an example."
- "Remember my favorite color is blue. What did I tell you earlier?" (within the running session)

## File guide

Paths below are relative to `everyday_groq_agent/`. Run the app and test commands from that directory.

- `app.py` — browser chat UI, session state, conversation IDs
- `agent.py` — Groq model, LangChain agent, tool definitions and system rules
- `services.py` — Tavily web search and local timestamp context
- `paragon_directory/paragon_search.py` — local FAISS and lexical retrieval
- `paragon_directory/paragon_tool.py` — agent tool wrapping Paragon retrieval
- `paragon_directory/paragon_store/piu.index` — raw FAISS index
- `paragon_directory/paragon_store/piu_chunks.pkl` — matching trusted document chunks
- [PARAGON_SETUP.md](PARAGON_SETUP.md) — Paragon index setup and retrieval commands
- `requirements.txt` — Python dependencies
- `.env.example` — API key template
- `.env` — your local API keys and optional index path overrides
- `.gitignore` — files excluded from version control
- `tests/` — offline unit tests for search and clock handling

## How it works

```text
You (Streamlit UI)
       |
       v
LangChain agent / Groq model
       |
       +-- Simple question? ------------> Answer directly
       |
       +-- Weather? --> Tavily search ---> Weather sources
       |
       +-- News? ----> Tavily search ---> Recent sources
       |
       +-- Current facts? -> Tavily ----> Web results
       |
       +-- Each model call receives a fresh local timestamp
       |
       v
    Response (links for retrieved news/web sources)
```

The model chooses whether to use a tool. When the user asks a follow-up, the checkpointer preserves the conversation using the same session-specific `thread_id`.

## Limitations and deployment notes

- Not a hosted deployment. Run locally or deploy Streamlit yourself.
- Chat memory is in RAM; it disappears when the app/server restarts. For durable sessions, switch to a database-backed LangGraph checkpointer.
- Results are fetched when you ask, not streamed automatically in the background.
- Models and API endpoints can be changed by their providers. The default `openai/gpt-oss-120b` model was documented by Groq as supporting tool use at the time of writing. The older `llama-3.3-70b-versatile` was deprecated for free/developer tiers in August 2026.
- External API requests have timeouts. If they fail, the model is instructed not to invent live facts.
- Weather and news freshness depend on search coverage and source dates; web search does not guarantee structured forecasts or a 24-hour news filter.
- Tool-calling LLMs are not perfectly reliable. For production: add tests, tool-call tracing, rate limiting, per-user auth, error logging, and data retention policies.
- API usage may incur charges. Review your Groq and Tavily quotas/pricing.

## Offline tests

```bash
python -m unittest discover -s tests -v
```

## Official documentation

- LangChain agents: https://docs.langchain.com/oss/python/langchain/agents
- ChatGroq: https://docs.langchain.com/oss/python/integrations/chat/groq
- Groq model availability: https://console.groq.com/docs/models
- Tavily: https://docs.tavily.com/
