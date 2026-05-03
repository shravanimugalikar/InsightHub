"""
Research Synthesizer Agent
──────────────────────────
For INITIAL queries:
  • Produces a full structured research report (Executive Summary, Key Findings,
    Detailed Analysis, Conclusion, References).

For FOLLOW-UP queries:
  • Generates a focused analytical response that directly answers the follow-up
    question, grounded in the prior report + any newly retrieved content.
  • Keeps the response concise (not a full report re-generation).
"""

import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY"),
)


def _truncate_context(docs: list, max_chars: int = 25000) -> str:
    """Join docs and truncate to a safe character limit."""
    full_text = "\n\n".join(docs)
    if len(full_text) <= max_chars:
        return full_text
    return full_text[:max_chars] + "... [Context truncated due to size limits]"


def run_synthesizer(state: dict) -> dict:
    """
    LangGraph node: synthesizer.

    Expects state keys:
      original_query, plan, retrieved_docs, citations,
      [chat_history], [session_context], [is_followup]
    Returns: report (str)
    """
    query         = state.get("original_query", state.get("query", ""))
    followup_q    = state.get("query", query)          # the actual question asked
    plan          = state.get("plan", "")
    retrieved_docs = state.get("retrieved_docs", [])
    citations     = state.get("citations", [])
    chat_history  = state.get("chat_history", [])
    session_ctx   = state.get("session_context", {})
    is_followup   = state.get("is_followup", False)
    source        = state.get("source", "")

    # Cap context to avoid token overflow — roughly 6-8k tokens
    context = _truncate_context(retrieved_docs, max_chars=25000)

    citation_block = "\n".join(
        [f"{i+1}. {c}" for i, c in enumerate(citations[:15])]
    ) if citations else ""

    # ── FOLLOW-UP SYNTHESIS ───────────────────────────────────────────────
    if is_followup:
        prior_report = session_ctx.get("report", "")

        # Build recent conversation turns
        recent_turns = []
        for msg in chat_history[-6:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            recent_turns.append(f"{role}: {msg['content'][:300]}")
        history_block = "\n".join(recent_turns)

        prompt = f"""You are an expert research analyst answering a follow-up question.

Original Research Topic: {query}
Source: {source}

Prior Research Report:
{prior_report[:2000]}

Recent Conversation:
{history_block}

Follow-up Question: {followup_q}

Research Plan for this Follow-up: {plan}

{"Additional Retrieved Content:" + chr(10) + context if context.strip() else ""}

{"Available Citations:" + chr(10) + citation_block if citation_block else ""}

Instructions:
- Answer the follow-up question directly and analytically.
- Ground your answer in the prior report and any additional retrieved content.
- Be concise but thorough — this is a focused response, NOT a full report re-generation.
- Use markdown formatting: bold key terms, bullet points for lists, ## for section headers if needed.
- Cite sources inline with [N] notation where applicable.
- For the References section (if you include one), provide the source URL as a markdown link [Link](url) immediately after the title/source name.
- Do NOT repeat the entire prior report; build on it."""

        report = llm.invoke(prompt).content

        return {
            **state,
            "report":    report,
            "citations": citations,
        }

    # ── INITIAL SYNTHESIS ─────────────────────────────────────────────────
    prompt = f"""You are an expert research synthesizer. Produce a comprehensive, structured research report.

Research Query: {query}
Source: {source}
Research Plan: {plan}

Retrieved Content:
{context}

Available Citations:
{citation_block}

Generate a well-structured research report with the following sections. Use markdown formatting:

## Executive Summary
(2-3 sentence overview of key findings)

## Key Findings
(3-5 numbered bullet points with the most important insights)

## Detailed Analysis
(Thorough paragraphs exploring the topic, referencing the retrieved content)

## Conclusion
(Brief closing thoughts and implications)

## References
(List only the academic/web sources provided in the 'Available Citations' section above. Number them [1], [2], etc. For each source, include the title/name followed by its URL in markdown format: [Link](url). If no citations were provided, do NOT include this section at all.)

Final Instruction: Be thorough and accurate. Cite sources inline using [1], [2], etc., corresponding to the References list. Do NOT use placeholder text like '[Insert Citation]'."""

    report = llm.invoke(prompt).content

    # Explicitly return important state fields to ensure they survive the graph merge
    return {
        "report":          report,
        "citations":       list(dict.fromkeys(citations)),
        "retrieved_docs":  retrieved_docs,
        "plan":            plan,
        "sub_questions":   state.get("sub_questions", []),
        "is_followup":     is_followup,
        "original_query":  query,
        "needs_retrieval": state.get("needs_retrieval", False),
    }
