"""
LLM Generation Module
─────────────────────
Uses Groq (llama-3.3-70b-versatile) for answer generation.
generate_answer() — legacy / simple Q&A
generate_structured_report() — used by the Synthesizer Agent
"""

import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)


def generate_answer(query: str, docs: list, history: list = None) -> str:
    """Simple RAG answer — used as a fallback / legacy interface."""
    context = "\n".join([doc.page_content for doc in docs])

    history_block = ""
    if history:
        lines = []
        for msg in history[-6:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        history_block = "\nConversation History:\n" + "\n".join(lines) + "\n"

    prompt = f"""You are an expert research assistant.
{history_block}
Answer the question based on the context below. Be thorough and structured.

Context:
{context}

Question:
{query}

Provide a clear, well-structured answer with sections where appropriate."""

    return llm.invoke(prompt).content


def generate_structured_report(
    query: str,
    docs: list,
    plan: str = "",
    citations: list = None,
    history: list = None,
) -> str:
    """
    Generates a full structured research report.
    Used directly by the Synthesizer agent via workflow.py.
    """
    context = "\n\n".join([doc.page_content for doc in docs])
    if len(context) > 25000:
        context = context[:25000] + "... [Truncated]"
    citation_block = ""
    if citations:
        citation_block = "\n".join(
            [f"{i+1}. {c}" for i, c in enumerate(citations[:15])]
        )

    history_block = ""
    if history:
        lines = []
        for msg in history[-6:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        history_block = "\nPrevious Conversation:\n" + "\n".join(lines)

    prompt = f"""You are an expert research synthesizer.
{history_block}
Research Query: {query}
Research Plan: {plan}

Retrieved Content:
{context}

Citations:
{citation_block}

Produce a structured report with these exact sections (use markdown):

## Executive Summary
## Key Findings
## Detailed Analysis
## Conclusion
## References

Be thorough. Cite sources inline as [1], [2], etc."""

    return llm.invoke(prompt).content