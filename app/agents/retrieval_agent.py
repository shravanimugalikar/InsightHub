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
    source        = state.get("source", "Global Insight")
    sub_questions = state.get("sub_questions", [state["query"]])
    uploaded_db   = state.get("uploaded_db", None)

    # If no sub-questions were generated, fall back to the original query
    if not sub_questions:
        sub_questions = [state.get("query", "")]

    retrieved_docs = []
    citations      = []

    if source == "Global Insight":
        from app.retrieval.arxiv_search import search_arxiv
        year_from = state.get("year_from")
        year_until = state.get("year_until")
        sort_by   = state.get("sort_by", "relevance")
        for sq in sub_questions:
            # Respect arXiv rate limit: at most one request every 3 seconds
            if sq != sub_questions[0]:
                time.sleep(3.1)
            results = search_arxiv(
                sq,
                max_results=3,
                year_from=year_from,
                year_until=year_until,
                sort_by=sort_by,
            )
            for r in results:
                retrieved_docs.append(
                    f"[ARXIV] {r['title']}\n{r['summary']}"
                )
                year_tag = f" [{r['year']}]" if r.get("year") else ""
                citations.append(
                    f"arXiv{year_tag} — {r['title']} ({r.get('url', 'arxiv.org')})"
                )

    elif source == "Local Insight":
        if uploaded_db is not None:
            from app.retrieval.retriever import get_retriever
            retriever = get_retriever(uploaded_db)
            for sq in sub_questions:
                docs = retriever.invoke(sq)
                for doc in docs:
                    retrieved_docs.append(doc.page_content)
                    source_meta = doc.metadata.get("source", "Uploaded Document")
                    citations.append(f"Local — {source_meta}")
        else:
            retrieved_docs = ["No document uploaded. Please upload a document first."]
            citations      = []

    elif source == "Web Insight":
        from app.retrieval.web_search import search_web
        for sq in sub_questions:
            result = search_web(sq)
            for r in result:
                retrieved_docs.append(f"[WEB] {r['snippet']}")
                if r.get("link"):
                    citations.append(
                        f"Web — {r.get('title', sq)} ({r['link']})"
                    )
                else:
                    citations.append(f"Web — {r.get('title', sq)}")

    # Merge with any existing docs from session context (for follow-ups
    # where the planner DID request new retrieval).
    existing_docs = session_ctx.get("retrieved_docs", [])
    existing_cits = session_ctx.get("citations", [])

    merged_docs = existing_docs + [d for d in retrieved_docs if d not in existing_docs]
    merged_cits = existing_cits + [c for c in citations      if c not in existing_cits]

    return {
        **state,
        "retrieved_docs": merged_docs,
        "citations":      list(dict.fromkeys(merged_cits)),
    }
