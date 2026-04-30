"""
InsightHub LangGraph Workflow
─────────────────────────────
Defines the StateGraph connecting:

  Initial:   Planner → Retrieval    → Synthesizer → END
  Follow-up: Planner → [Retrieval?] → Synthesizer → END
             (Retrieval is skipped when Planner sets needs_retrieval=False)

Exposes run_workflow() as the single entry point.
"""

from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END

from app.agents.planner_agent    import run_planner
from app.agents.retrieval_agent  import run_retrieval
from app.agents.synthesizer_agent import run_synthesizer


class AgentState(TypedDict, total=False):
    # ── Inputs ──────────────────────────────────────────────────────────
    query:          str
    source:         str
    uploaded_db:    Optional[object]
    chat_history:   Optional[List[dict]]
    session_context: Optional[dict]

    # ── arXiv filters (Global Insight only) ─────────────────────────────
    year_from:      Optional[int]
    year_until:     Optional[int]
    sort_by:        Optional[str]   # "relevance" | "latest"

    # ── Planner output ───────────────────────────────────────────────────
    plan:            str
    sub_questions:   List[str]
    original_query:  str
    needs_retrieval: bool
    is_followup:     bool

    # ── Retrieval output ─────────────────────────────────────────────────
    retrieved_docs:  List[str]
    citations:       List[str]

    # ── Synthesizer output ───────────────────────────────────────────────
    report:          str


def _route_after_planner(state: dict) -> str:
    if state.get("needs_retrieval", True):
        return "retrieval"
    return "synthesizer"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("planner",     run_planner)
    graph.add_node("retrieval",   run_retrieval)
    graph.add_node("synthesizer", run_synthesizer)

    graph.set_entry_point("planner")

    graph.add_conditional_edges(
        "planner",
        _route_after_planner,
        {
            "retrieval":   "retrieval",
            "synthesizer": "synthesizer",
        },
    )

    graph.add_edge("retrieval",   "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()


# Compile once at module level for reuse across calls
_compiled_graph = build_graph()


def run_workflow(
    query:          str,
    source:         str,
    uploaded_db=None,
    chat_history:   list = None,
    session_context: dict = None,
    year_from:      int = None,
    year_until:     int = None,
    sort_by:        str = "relevance",
) -> dict:
    """
    Main entry point.

    Args:
        query           – The user's question (initial or follow-up).
        source          – Insight source label.
        uploaded_db     – Vector store for Local Insights (or None).
        chat_history    – List of prior {role, content} messages.
        session_context – Stored research context from memory.get_session_context().
        year_from       – Filter arXiv papers from this year (Global Insight only).
        year_until      – Filter arXiv papers up to this year (Global Insight only).
        sort_by         – "relevance" or "latest" (Global Insight only).

    Returns:
        dict with keys: plan, sub_questions, needs_retrieval, is_followup,
                        retrieved_docs, citations, report
    """
    initial_state: AgentState = {
        "query":           query,
        "source":          source,
        "uploaded_db":     uploaded_db,
        "chat_history":    chat_history    or [],
        "session_context": session_context or {},
        "year_from":       year_from,
        "year_until":      year_until,
        "sort_by":         sort_by,
    }

    return _compiled_graph.invoke(initial_state)
