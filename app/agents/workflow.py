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
    session_context: Optional[dict]      # stored from prior run

    # ── Planner output ───────────────────────────────────────────────────
    plan:            str
    sub_questions:   List[str]
    original_query:  str
    needs_retrieval: bool                # True → run retrieval; False → skip
    is_followup:     bool

    # ── Retrieval output ─────────────────────────────────────────────────
    retrieved_docs:  List[str]
    citations:       List[str]

    # ── Synthesizer output ───────────────────────────────────────────────
    report:          str


def _route_after_planner(state: dict) -> str:
    """
    Conditional router: if Planner says retrieval is not needed,
    jump straight to the synthesizer node.
    """
    if state.get("needs_retrieval", True):
        return "retrieval"
    return "synthesizer"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("planner",     run_planner)
    graph.add_node("retrieval",   run_retrieval)
    graph.add_node("synthesizer", run_synthesizer)

    graph.set_entry_point("planner")

    # Conditional edge: retrieval may be skipped for context-only follow-ups
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
) -> dict:
    """
    Main entry point.

    Args:
        query           – The user's question (initial or follow-up).
        source          – Insight source label.
        uploaded_db     – Vector store for Local Insights (or None).
        chat_history    – List of prior {role, content} messages.
        session_context – Stored research context from memory.get_session_context().

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
    }

    return _compiled_graph.invoke(initial_state)
