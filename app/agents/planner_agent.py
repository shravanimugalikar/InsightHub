"""
Planner Agent
─────────────
For INITIAL queries:
  • Writes a research plan (2-3 sentences)
  • Breaks the query into 2-4 focused sub-questions
  • Sets needs_retrieval = True

For FOLLOW-UP queries (chat_history is non-empty):
  • Decides whether the follow-up needs fresh retrieval or can be
    answered from the stored session context.
  • Sets needs_retrieval = True/False accordingly.
  • Generates focused sub-questions only when retrieval is needed.
"""

import json
import re
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY"),
)


def run_planner(state: dict) -> dict:
    """
    LangGraph node: planner.

    Expects state keys:
      query, source, [chat_history], [session_context]

    Returns (merged into state):
      plan, sub_questions, original_query,
      needs_retrieval (bool), is_followup (bool)
    """
    query         = state["query"]
    source        = state["source"]
    chat_history  = state.get("chat_history", [])
    session_ctx   = state.get("session_context", {})
    is_followup   = bool(chat_history) and bool(session_ctx.get("report"))

    # ── FOLLOW-UP PATH ─────────────────────────────────────────────────────
    if is_followup:
        original_query = session_ctx.get("original_query", query)
        prior_report   = session_ctx.get("report", "")[:800]   # trimmed for prompt

        # Build recent history snippet
        recent = []
        for msg in chat_history[-4:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            recent.append(f"{role}: {msg['content'][:200]}")
        history_snippet = "\n".join(recent)

        followup_prompt = f"""You are a research planning agent handling a FOLLOW-UP question.

Original Research Topic: {original_query}
Source: {source}

Prior Report Summary (first 800 chars):
{prior_report}

Recent Conversation:
{history_snippet}

New Follow-up Question: {query}

Decide:
1. Can this follow-up be answered from the existing report and conversation context alone?
   → If YES: set needs_retrieval to false.
   → If NO (the question asks for NEW information not present in the report): set needs_retrieval to true,
     and provide 1-2 focused sub-questions for retrieval.

Respond ONLY with valid JSON:
{{
  "needs_retrieval": true or false,
  "plan": "Brief 1-2 sentence plan for answering this follow-up...",
  "sub_questions": ["sub-question 1 (only if needs_retrieval is true)"]
}}"""

        response = llm.invoke(followup_prompt).content.strip()

        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
        else:
            parsed = {
                "needs_retrieval": False,
                "plan": f"Answering follow-up: {query}",
                "sub_questions": [],
            }

        return {
            **state,
            "plan":            parsed.get("plan", f"Answering follow-up: {query}"),
            "sub_questions":   parsed.get("sub_questions", [query]) if parsed.get("needs_retrieval") else [],
            "needs_retrieval": parsed.get("needs_retrieval", False),
            "is_followup":     True,
            "original_query":  original_query,
        }

    # ── INITIAL QUERY PATH ─────────────────────────────────────────────────
    prompt = f"""You are a research planning agent. A user wants to research the following topic using {source}.

User Query: {query}

Your job:
1. Write a concise research plan (2-3 sentences) describing the approach.
2. Break down the query into 2-4 focused sub-questions that will guide the retrieval process.

Respond ONLY with valid JSON in this exact format:
{{
  "plan": "Brief description of the research approach...",
  "sub_questions": [
    "Sub-question 1?",
    "Sub-question 2?",
    "Sub-question 3?"
  ]
}}"""

    response = llm.invoke(prompt).content.strip()

    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        parsed = json.loads(json_match.group())
    else:
        parsed = {
            "plan": response if len(response) < 500 else f"Researching: {query}",
            "sub_questions": [query],
        }

    return {
        **state,
        "plan":            parsed.get("plan", f"Researching: {query}"),
        "sub_questions":   parsed.get("sub_questions", [query]),
        "needs_retrieval": True,
        "is_followup":     False,
        "original_query":  query,
    }
