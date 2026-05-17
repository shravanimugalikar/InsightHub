"""
Retrieval Agent
───────────────
Dispatches retrieval based on the source tab selected by the user.
Iterates over sub-questions from the Planner and retrieves relevant docs/snippets.

Respects the needs_retrieval flag set by the Planner:
  • needs_retrieval = True  → fetch fresh documents from source
  • needs_retrieval = False → skip retrieval; reuse docs from session_context
"""

from typing import TypedDict
import time


class RetrievalOutput(TypedDict):
    query: str
    source: str
    retrieved_docs: list
    citations: list


def run_retrieval(state: dict) -> dict:
    """
    LangGraph node: retrieval.

    Expects: query, source, sub_questions, needs_retrieval,
             [uploaded_db], [session_context]
    Returns: retrieved_docs (list of str), citations (list of str)
    """
    needs_retrieval = state.get("needs_retrieval", True)
    session_ctx     = state.get("session_context", {})

    # ── SKIP: reuse stored context ────────────────────────────────────────
    if not needs_retrieval:
        return {
            **state,
            "retrieved_docs": session_ctx.get("retrieved_docs", []),
            "citations":      session_ctx.get("citations", []),
        }

    # ── FETCH: retrieve fresh documents ───────────────────────────────────
    from app.retrieval.arxiv_search import search_arxiv

    source        = state.get("source", "")
    sub_questions = state.get("sub_questions", [])
    uploaded_db   = state.get("uploaded_db")
    year_from     = state.get("year_from")
    # Support both state representations: year_to and year_until
    year_to       = state.get("year_to") if state.get("year_to") is not None else state.get("year_until")
    # Support both state representations: arxiv_sort and sort_by
    sort_by       = state.get("arxiv_sort") if state.get("arxiv_sort") is not None else state.get("sort_by", "Relevance")

    # If no sub-questions were generated, fall back to the original query
    if not sub_questions:
        sub_questions = [state.get("query", "")]

    all_docs   = []
    all_citations = []

    if source == "Global Insight":
        main_query = state.get("query", "")

        # Primary: search the main query (up to 8 results)
        main_docs = search_arxiv(
            query=main_query,
            max_results=8,
            year_from=year_from,
            year_to=year_to,
            sort_by=sort_by,
        )
        all_docs.extend(main_docs)

        # Secondary: search first sub-question if different from main query
        if sub_questions and sub_questions[0].strip().lower() != main_query.strip().lower():
            extra_docs = search_arxiv(
                query=sub_questions[0],
                max_results=5,
                year_from=year_from,
                year_to=year_to,
                sort_by=sort_by,
            )
            all_docs.extend(extra_docs)

        # Deduplicate by URL
        seen = set()
        unique_docs = []
        for doc in all_docs:
            url = doc.metadata.get("url", "")
            if url and url not in seen:
                seen.add(url)
                unique_docs.append(doc)
        all_docs = unique_docs[:10]   # keep max 10

        # Build citations list — synthesizer will also rebuild this from retrieved_docs
        all_citations = [
            f"[{i+1}] {doc.metadata.get('title', 'Untitled')}. "
            f"{doc.metadata.get('url', '')}"
            for i, doc in enumerate(all_docs)
        ]

    elif source == "Local Insight":
        if uploaded_db is not None:
            from app.retrieval.retriever import get_retriever
            retriever = get_retriever(uploaded_db)
            for sq in sub_questions:
                docs = retriever.invoke(sq)
                for doc in docs:
                    all_docs.append(doc)
        else:
            all_docs = ["No document uploaded. Please upload a document first."]
            all_citations = []

    elif source == "Web Insight":
        from app.retrieval.web_search import search_web
        for sq in sub_questions:
            result = search_web(sq)
            for r in result:
                all_docs.append(r)

    # Merge with any existing docs from session context (for follow-ups
    # where the planner DID request new retrieval).
    existing_docs = session_ctx.get("retrieved_docs", [])
    existing_cits = session_ctx.get("citations", [])

    merged_docs = existing_docs + [d for d in all_docs if d not in existing_docs]
    merged_cits = existing_cits + [c for c in all_citations if c not in existing_cits]

    return {
        **state,
        "retrieved_docs": merged_docs,
        "citations":      list(dict.fromkeys(merged_cits)),
    }
