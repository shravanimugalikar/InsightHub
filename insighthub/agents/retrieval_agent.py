# ============================================================
#  InsightHub — insighthub/agents/retrieval_agent.py
#  Executes per-source retrieval using the QueryPlan
#  from PlannerAgent and returns ranked chunks
# ============================================================

import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from langchain.schema import Document

from insighthub.config.settings import (
    RETRIEVAL_TOP_K,
    RETRIEVAL_FINAL_K,
    SIMILARITY_THRESHOLD,
    COHERE_API_KEY,
)
from insighthub.knowledge.vector_store import VectorStore
from insighthub.agents.planner_agent import PlannerAgent, QueryPlan, SubQuery

# ── Logging setup ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """A single retrieved chunk with its source info and score."""
    document    : Document
    source_tab  : str
    score       : float = 0.0
    rank        : int   = 0
    sub_query   : str   = ""


@dataclass
class RetrievalResult:
    """
    Full retrieval result returned by the Retrieval Agent.
    Contains ranked chunks from all source tabs.
    """
    query           : str
    chunks          : List[RetrievedChunk]
    by_source       : Dict[str, List[RetrievedChunk]] = field(default_factory=dict)
    total_retrieved : int = 0
    reranked        : bool = False


# ─────────────────────────────────────────────────────────────
#  RETRIEVAL AGENT CLASS
# ─────────────────────────────────────────────────────────────

class RetrievalAgent:
    """
    Executes per-source hybrid retrieval based on the QueryPlan
    produced by the PlannerAgent.

    Workflow:
        1. Receive QueryPlan with sub-queries per source tab
        2. Execute hybrid search (dense + BM25) per sub-query
        3. Apply recency + relevance weighting
        4. Rerank with Cohere if API key available
        5. Return top-k chunks with full source metadata

    Usage:
        retriever = RetrievalAgent(vector_store)
        plan      = planner.plan("What is RAG?")
        result    = retriever.retrieve(plan)

        for chunk in result.chunks:
            print(chunk.source_tab, chunk.document.page_content[:100])
    """

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.reranker     = self._load_reranker()
        logger.info("RetrievalAgent initialised (reranker=%s)",
                    "Cohere" if self.reranker else "disabled")

    # ── Public Methods ────────────────────────────────────────

    def retrieve(
        self,
        plan    : QueryPlan,
        top_k   : int = RETRIEVAL_FINAL_K,
    ) -> RetrievalResult:
        """
        Main method — executes retrieval for all sub-queries in the plan.

        Args:
            plan:  QueryPlan from PlannerAgent
            top_k: Final number of chunks to return across all sources

        Returns:
            RetrievalResult with ranked chunks from all source tabs
        """
        logger.info("Retrieving for query: '%s' (sources: %s)",
                    plan.original_query[:60], plan.active_sources)

        all_chunks : List[RetrievedChunk] = []
        by_source  : Dict[str, List[RetrievedChunk]] = {}

        # ── Step 1: Retrieve per sub-query ────────────────────
        for sub_query in plan.sub_queries:
            chunks = self._retrieve_for_subquery(sub_query)
            all_chunks.extend(chunks)

            # Group by source tab
            tab = sub_query.source_tab
            if tab not in by_source:
                by_source[tab] = []
            by_source[tab].extend(chunks)

        logger.info("Retrieved %d total chunks across %d sources",
                    len(all_chunks), len(by_source))

        if not all_chunks:
            logger.warning("No chunks retrieved — vector store may be empty")
            return RetrievalResult(
                query           = plan.original_query,
                chunks          = [],
                by_source       = by_source,
                total_retrieved = 0,
            )

        # ── Step 2: Deduplicate ───────────────────────────────
        all_chunks = self._deduplicate(all_chunks)
        logger.info("After deduplication: %d chunks", len(all_chunks))

        # ── Step 3: Score and rank ────────────────────────────
        all_chunks = self._score_and_rank(all_chunks, plan.original_query)

        # ── Step 4: Rerank with Cohere (if available) ─────────
        reranked = False
        if self.reranker and len(all_chunks) > top_k:
            all_chunks = self._cohere_rerank(
                plan.original_query, all_chunks, top_k
            )
            reranked = True
        else:
            all_chunks = all_chunks[:top_k]

        # ── Step 5: Update rank indices ───────────────────────
        for i, chunk in enumerate(all_chunks):
            chunk.rank = i + 1

        # Rebuild by_source with final chunks only
        final_by_source: Dict[str, List[RetrievedChunk]] = {}
        for chunk in all_chunks:
            tab = chunk.source_tab
            if tab not in final_by_source:
                final_by_source[tab] = []
            final_by_source[tab].append(chunk)

        logger.info("✓ Final retrieval: %d chunks (reranked=%s)", len(all_chunks), reranked)

        return RetrievalResult(
            query           = plan.original_query,
            chunks          = all_chunks,
            by_source       = final_by_source,
            total_retrieved = len(all_chunks),
            reranked        = reranked,
        )

    def retrieve_simple(
        self,
        query      : str,
        source_tabs: Optional[List[str]] = None,
        top_k      : int = RETRIEVAL_FINAL_K,
    ) -> RetrievalResult:
        """
        Simplified retrieval without a QueryPlan.
        Creates a simple plan internally and retrieves.
        Used for quick testing and simple queries.
        """
        from insighthub.agents.planner_agent import PlannerAgent
        planner = PlannerAgent(active_sources=source_tabs or ["document", "website", "paper"])
        plan    = planner.plan_simple(query)
        return self.retrieve(plan, top_k=top_k)

    # ── Per-SubQuery Retrieval ────────────────────────────────

    def _retrieve_for_subquery(self, sub_query: SubQuery) -> List[RetrievedChunk]:
        """Execute hybrid search for a single sub-query."""
        try:
            docs = self.vector_store.hybrid_search(
                query           = sub_query.query,
                k               = RETRIEVAL_TOP_K,
                filter_metadata = {"source_tab": sub_query.source_tab},
            )

            chunks = []
            for doc in docs:
                chunks.append(RetrievedChunk(
                    document   = doc,
                    source_tab = sub_query.source_tab,
                    score      = 0.5,   # base score, updated in _score_and_rank
                    sub_query  = sub_query.query,
                ))

            logger.info("Sub-query [%s]: '%s' → %d chunks",
                        sub_query.source_tab.upper(),
                        sub_query.query[:50],
                        len(chunks))
            return chunks

        except Exception as e:
            logger.warning("Retrieval failed for sub-query '%s': %s",
                           sub_query.query[:50], str(e))
            return []

    # ── Scoring and Ranking ───────────────────────────────────

    def _score_and_rank(
        self,
        chunks : List[RetrievedChunk],
        query  : str,
    ) -> List[RetrievedChunk]:
        """
        Score chunks using a combination of:
        - Recency: newer sources score higher
        - Source priority: papers > documents > websites for academic queries
        - Position: chunks retrieved at higher rank score higher
        """
        for i, chunk in enumerate(chunks):
            base_score = 1.0 / (i + 1)   # position-based score

            # Recency boost — newer content scores higher
            recency_boost = self._get_recency_boost(chunk.document)

            # Source type weight based on query nature
            source_weight = self._get_source_weight(chunk.source_tab)

            chunk.score = base_score * recency_boost * source_weight

        # Sort by score descending
        chunks.sort(key=lambda x: x.score, reverse=True)
        return chunks

    def _get_recency_boost(self, doc: Document) -> float:
        """Give higher scores to more recent content."""
        year = doc.metadata.get("year", "")
        if not year:
            scraped_date = doc.metadata.get("scraped_date", "")
            if scraped_date:
                year = scraped_date[:4]

        try:
            year_int = int(str(year)[:4])
            if year_int >= 2024:
                return 1.3
            elif year_int >= 2022:
                return 1.1
            elif year_int >= 2020:
                return 1.0
            else:
                return 0.9
        except (ValueError, TypeError):
            return 1.0     # no year info — neutral

    def _get_source_weight(self, source_tab: str) -> float:
        """
        Weight sources by reliability for academic research.
        Papers > Documents > Websites for factual accuracy.
        """
        weights = {
            "paper"   : 1.2,    # peer-reviewed, most reliable
            "document": 1.0,    # user-provided documents
            "website" : 0.9,    # web content, less verified
        }
        return weights.get(source_tab, 1.0)

    # ── Deduplication ─────────────────────────────────────────

    def _deduplicate(
        self,
        chunks    : List[RetrievedChunk],
        threshold : float = 0.9,
    ) -> List[RetrievedChunk]:
        """
        Remove near-duplicate chunks based on content overlap.
        Keeps the first occurrence of very similar chunks.
        """
        seen      = []
        unique    = []

        for chunk in chunks:
            text = chunk.document.page_content.strip()[:200]

            # Check overlap with already seen chunks
            is_duplicate = False
            for seen_text in seen:
                overlap = self._text_overlap(text, seen_text)
                if overlap > threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(chunk)
                seen.append(text)

        return unique

    def _text_overlap(self, text1: str, text2: str) -> float:
        """Calculate word overlap ratio between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union        = words1 | words2
        return len(intersection) / len(union)

    # ── Cohere Reranker ───────────────────────────────────────

    def _load_reranker(self):
        """Load Cohere reranker if API key is available."""
        if not COHERE_API_KEY:
            logger.info("Cohere API key not set — reranking disabled")
            return None
        try:
            import cohere
            client = cohere.Client(COHERE_API_KEY)
            logger.info("✓ Cohere reranker loaded")
            return client
        except ImportError:
            logger.warning("cohere package not installed — run: pip install cohere")
            return None
        except Exception as e:
            logger.warning("Failed to load Cohere reranker: %s", str(e))
            return None

    def _cohere_rerank(
        self,
        query  : str,
        chunks : List[RetrievedChunk],
        top_k  : int,
    ) -> List[RetrievedChunk]:
        """Rerank chunks using Cohere's cross-encoder reranker."""
        try:
            texts    = [c.document.page_content for c in chunks]
            response = self.reranker.rerank(
                model     = "rerank-english-v3.0",
                query     = query,
                documents = texts,
                top_n     = top_k,
            )

            reranked = []
            for result in response.results:
                chunk       = chunks[result.index]
                chunk.score = result.relevance_score
                reranked.append(chunk)

            logger.info("✓ Cohere reranked %d → %d chunks", len(chunks), len(reranked))
            return reranked

        except Exception as e:
            logger.warning("Cohere reranking failed: %s — using score-based ranking", str(e))
            return chunks[:top_k]


# ─────────────────────────────────────────────────────────────
#  QUICK TEST
#  Command: python insighthub/agents/retrieval_agent.py
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "="*55)
    print("  InsightHub — Retrieval Agent Test")
    print("="*55)

    # ── Step 1: Set up vector store with sample docs ──────────
    print("\n[1/3] Setting up vector store with sample documents...")
    try:
        from insighthub.knowledge.vector_store import VectorStore

        store = VectorStore()

        # Add sample documents if store is empty
        if store.get_stats()["total_docs"] == 0:
            sample_docs = [
                Document(
                    page_content = "RAG (Retrieval-Augmented Generation) combines a retrieval component with a generative language model. It retrieves relevant documents from a knowledge base and uses them as context for generation, reducing hallucinations significantly.",
                    metadata     = {"source_tab": "paper", "title": "RAG Survey", "year": "2023", "chunk_id": 0}
                ),
                Document(
                    page_content = "Self-RAG improves upon standard RAG by introducing special reflection tokens that allow the model to adaptively decide when to retrieve, critique its own outputs, and select the most relevant passages.",
                    metadata     = {"source_tab": "paper", "title": "Self-RAG Paper", "year": "2023", "chunk_id": 1}
                ),
                Document(
                    page_content = "LangChain provides a framework for building RAG applications with built-in support for vector stores, document loaders, and chain abstractions that simplify multi-step reasoning.",
                    metadata     = {"source_tab": "website", "domain": "langchain.com", "scraped_date": "2024-01-15", "chunk_id": 2}
                ),
                Document(
                    page_content = "Multi-agent systems in AI involve multiple autonomous agents working together to solve complex tasks. Each agent specializes in a specific subtask and communicates with others through a shared state or message passing.",
                    metadata     = {"source_tab": "document", "filename": "ai_notes.pdf", "year": "2024", "chunk_id": 3}
                ),
                Document(
                    page_content = "FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors. It supports billion-scale datasets and runs efficiently on both CPU and GPU.",
                    metadata     = {"source_tab": "document", "filename": "faiss_guide.txt", "year": "2024", "chunk_id": 4}
                ),
                Document(
                    page_content = "The Critic Agent evaluates response quality across four dimensions: relevance to the query, coherence of reasoning, grounding in retrieved sources, and completeness of coverage.",
                    metadata     = {"source_tab": "document", "filename": "insighthub_design.txt", "year": "2024", "chunk_id": 5}
                ),
            ]
            store.add_documents(sample_docs)
            print(f"  ✓ Added {len(sample_docs)} sample documents")
        else:
            print(f"  ✓ Using existing index ({store.get_stats()['total_docs']} docs)")

    except Exception as e:
        print(f"  ✗ Vector store setup failed: {e}")
        sys.exit(1)

    # ── Step 2: Set up agents ─────────────────────────────────
    print("\n[2/3] Initialising Planner and Retrieval agents...")
    try:
        planner   = PlannerAgent()
        retriever = RetrievalAgent(store)
        print("  ✓ PlannerAgent initialised")
        print("  ✓ RetrievalAgent initialised")
    except Exception as e:
        print(f"  ✗ Agent setup failed: {e}")
        sys.exit(1)

    # ── Step 3: Run retrieval tests ───────────────────────────
    print("\n[3/3] Running retrieval tests...")

    test_queries = [
        "How does RAG reduce hallucinations in language models?",
        "Compare RAG and Self-RAG approaches",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n  Test {i}: '{query}'")
        print("  " + "-"*45)
        try:
            # Plan the query
            plan = planner.plan(query)
            print(f"  ✓ Plan intent   : {plan.intent}")
            print(f"  ✓ Sub-queries   : {len(plan.sub_queries)}")

            # Retrieve
            result = retriever.retrieve(plan, top_k=4)
            print(f"  ✓ Total chunks  : {result.total_retrieved}")
            print(f"  ✓ Reranked      : {result.reranked}")
            print(f"  ✓ By source     : { {k: len(v) for k, v in result.by_source.items()} }")
            print(f"\n  Top results:")
            for chunk in result.chunks[:3]:
                print(f"    [{chunk.rank}] [{chunk.source_tab.upper()}] "
                      f"score={chunk.score:.3f}")
                print(f"        '{chunk.document.page_content[:90]}...'")

        except Exception as e:
            print(f"  ✗ Failed: {e}")

    print("\n" + "="*55)
    print("  Retrieval Agent Test Complete!")
    print("="*55 + "\n")