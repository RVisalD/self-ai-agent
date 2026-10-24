"""Run with: streamlit run app.py"""

from __future__ import annotations

import os
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage

from agent import build_agent

load_dotenv()
st.set_page_config(page_title="Everyday AI Agent", page_icon="🌤️", layout="centered")


@st.cache_resource(show_spinner=False)
def get_agent(default_city: str, timezone: str, model: str):
    """Reuse the LangChain graph/checkpointer across Streamlit reruns."""
    return build_agent(default_city, timezone)


def text_from_content(content) -> str:
    """Handle both plain-text and block-style model messages."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
        ) or str(content)
    return str(content)


def this_turn_tools(messages):
    """Only show tool calls made after the most recent user message."""
    used = []
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, ToolMessage) and message.name:
            used.append(message.name)
    return list(dict.fromkeys(reversed(used)))


with st.sidebar:
    st.header("Preferences")
    city = st.text_input("Default weather city", value="Phnom Penh")
    timezone = st.text_input("Your timezone (IANA)", value="Asia/Phnom_Penh")
    st.caption("Examples: Asia/Phnom_Penh, Asia/Bangkok, Europe/London")
    st.divider()
    st.caption("**Model:** " + os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
    st.caption("**Weather, news & web search:** Tavily")
    st.caption("**Paragon knowledge base:** local FAISS + BM25")
    st.caption("Chats are stored in memory for this session, not saved permanently.")
    if st.button("🗑️ New conversation", use_container_width=True):
        st.session_state.pop("messages", None)
        st.session_state["thread_id"] = str(uuid4())
        st.rerun()

st.title("Your AI support Agent")
st.caption("A general-purpose chatbot powered by LangChain + Groq, with live web search for weather and news, plus a Paragon knowledge base.")

if not os.getenv("GROQ_API_KEY", "").strip():
    st.error("Missing GROQ_API_KEY. Copy .env.example to .env and add your Groq key.")
    st.stop()
if not os.getenv("TAVILY_API_KEY", "").strip():
    st.warning("TAVILY_API_KEY is missing. Basic chat works, but weather, news, and web search won't.")
try:
    ZoneInfo(timezone)
except (ZoneInfoNotFoundError, ValueError, KeyError):
    st.error("Invalid timezone. Try Asia/Phnom_Penh or Europe/London.")
    st.stop()

settings = (city.strip() or "Phnom Penh", timezone, os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
if st.session_state.get("settings") != settings:
    st.session_state["settings"] = settings
    st.session_state["thread_id"] = str(uuid4())
    st.session_state["messages"] = []

st.session_state.setdefault("thread_id", str(uuid4()))
st.session_state.setdefault("messages", [])
agent = get_agent(*settings)

if not st.session_state["messages"]:
    st.info("Try asking: **What's the weather today?** · **What's the latest world news?** · **Explain quantum computing simply.**")

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask anything, including real-time questions…"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Please hold one moment while I am thinking…"):
            try:
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": prompt}]},
                    config={
                        "configurable": {"thread_id": st.session_state["thread_id"]},
                        "recursion_limit": 22,
                    },
                )
                answer = text_from_content(result["messages"][-1].content)
                tools = this_turn_tools(result["messages"])
            except Exception as exc:
                answer = (
                    f"I couldn't complete that request ({type(exc).__name__}). "
                    "Check your API keys, network connection, or model availability and try again."
                )
                tools = []
            st.markdown(answer)
    st.session_state["messages"].append({"role": "assistant", "content": answer, "tools": tools})
