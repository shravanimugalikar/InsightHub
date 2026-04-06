"""
Session Memory Utility
──────────────────────
Manages chat history and full session context in Streamlit session_state.

Each insight tab maintains:
  • chat_history  – list of {role, content} exchange messages
  • session_ctx   – rich context dict:
        original_query   : str
        retrieved_docs   : list[str]
        citations        : list[str]
        report           : str
        source           : str
"""

import streamlit as st


# ─── Keys ────────────────────────────────────────────────────────────────────

def _history_key(tab: str) -> str:
    return f"chat_history_{tab.replace(' ', '_').lower()}"

def _ctx_key(tab: str) -> str:
    return f"session_ctx_{tab.replace(' ', '_').lower()}"


# ─── Chat History ─────────────────────────────────────────────────────────────

def init_memory(tab: str):
    """Initialize memory for a given insight tab if not already set."""
    hk = _history_key(tab)
    ck = _ctx_key(tab)
    if hk not in st.session_state:
        st.session_state[hk] = []
    if ck not in st.session_state:
        st.session_state[ck] = {}


def add_to_memory(tab: str, role: str, content: str):
    """Append a message to the chat history for the given tab."""
    hk = _history_key(tab)
    if hk not in st.session_state:
        st.session_state[hk] = []
    st.session_state[hk].append({"role": role, "content": content})


def get_memory(tab: str) -> list:
    """Retrieve the full chat history for the given tab."""
    return st.session_state.get(_history_key(tab), [])


def clear_memory(tab: str):
    """Clear chat history and session context for the given tab."""
    st.session_state[_history_key(tab)] = []
    st.session_state[_ctx_key(tab)] = {}


def get_last_report(tab: str) -> str:
    """Return the last assistant report from memory, if available."""
    for msg in reversed(get_memory(tab)):
        if msg["role"] == "assistant":
            return msg["content"]
    return ""


# ─── Session Context (rich research state) ───────────────────────────────────

def store_session_context(tab: str, result: dict):
    """
    Persist the full research result as the active session context.
    Stores: original_query, retrieved_docs, citations, report, source.
    Called after every successful workflow run (initial + follow-ups).
    """
    ck = _ctx_key(tab)
    ctx = st.session_state.get(ck, {})

    # Accumulate retrieved docs across follow-ups (deduplicated)
    existing_docs  = ctx.get("retrieved_docs", [])
    existing_cits  = ctx.get("citations", [])
    new_docs       = result.get("retrieved_docs", [])
    new_cits       = result.get("citations", [])

    merged_docs = existing_docs + [d for d in new_docs if d not in existing_docs]
    merged_cits = existing_cits + [c for c in new_cits if c not in existing_cits]

    st.session_state[ck] = {
        "original_query": ctx.get("original_query") or result.get("original_query", result.get("query", "")),
        "retrieved_docs":  merged_docs,
        "citations":       merged_cits,
        "report":          result.get("report", ctx.get("report", "")),
        "source":          result.get("source", ctx.get("source", "")),
        "plan":            result.get("plan", ctx.get("plan", "")),
        "sub_questions":   result.get("sub_questions", ctx.get("sub_questions", [])),
    }


def get_session_context(tab: str) -> dict:
    """
    Retrieve the stored session context for the given tab.
    Returns an empty dict if nothing has been stored yet.
    """
    return st.session_state.get(_ctx_key(tab), {})


def has_session_context(tab: str) -> bool:
    """True if a research session has been established for this tab."""
    ctx = get_session_context(tab)
    return bool(ctx.get("report"))
