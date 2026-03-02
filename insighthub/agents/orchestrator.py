# ============================================================
#  InsightHub — insighthub/agents/orchestrator.py
#  Ties all 4 agents into a single end-to-end pipeline
#  Query → Plan → Retrieve → Analyse → Critique → Response
# ============================================================

import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from langchain.schema import Document

from insighthub.config.settings import SOURCE_TYPES
from insighthub.knowledge.vector_store import VectorStore
from insighthub.agents.planner_agent   import PlannerAgent,   QueryPlan
from insighthub.agents.retrieval_agent import RetrievalAgent, RetrievalResult
from insighthub.agents.analysis_agent  import AnalysisAgent,  AnalysisResult
from insighthub.agents.critic_agent    import CriticAgent,    CriticResult

# ── Logging setup ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """
    Complete end-to-end result from the Orchestrator pipeline.
    Contains all intermediate results plus the final response.
    """
    query            : str
    final_response   : str
    key_findings     : List[str]
    citations        : List[Dict]
    by_source        : Dict[str, str]
    overall_score    : float
    approved         : bool
    has_conflicts    : bool
    retry_count      : int
    elapsed_seconds  : float

    # Intermediate results (for debugging/UI display)
    plan             : Optional[QueryPlan]       = None
    retrieval_result : Optional[RetrievalResult] = None
    analysis_result  : Optional[AnalysisResult]  = None
    critic_result    : Optional[CriticResult]    = None

    # Status
    error            : str = ""
    success          : bool = True


# ─────────────────────────────────────────────────────────────
#  ORCHESTRATOR CLASS
# ─────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Central coordinator for the InsightHub multi-agent pipeline.

    Manages the full research workflow:
        1. PlannerAgent   — decomposes query into sub-queries
        2. RetrievalAgent — fetches chunks per source tab
        3. AnalysisAgent  — synthesizes chunks into response
        4. CriticAgent    — evaluates and refines the response

    Also manages document ingestion:
        - ingest_document() — PDF, DOCX, TXT
        - ingest_url()      — website
        - ingest_paper()    — arXiv paper

    Usage:
        orch = Orchestrator()

        # Ingest sources
        orch.ingest_document("path/to/paper.pdf")
        orch.ingest_url("https://example.com/article")
        orch.ingest_paper("2310.11511")

        # Run a query
        result = orch.run("What is retrieval-augmented generation?")
        print(result.final_response)
        print(result.overall_score)
    """

    def __init__(
        self,
        active_sources  : Optional[List[str]] = None,
        enable_critic   : bool = True,
        progress_callback: Optional[Callable] = None,
    ):
        """
        Args:
            active_sources:    Which source tabs to query (default: all 3)
            enable_critic:     Whether to run the Critic Agent (default: True)
            progress_callback: Optional function called at each pipeline step
                              signature: callback(step: str, message: str)
        """
        self.active_sources    = active_sources or SOURCE_TYPES
        self.enable_critic     = enable_critic
        self.progress_callback = progress_callback

        self._report_progress("init", "Initialising InsightHub pipeline...")

        # Initialise shared vector store
        self.vector_store = VectorStore()
        self._report_progress("init", "✓ Vector store ready")

        # Initialise all agents
        self.planner   = PlannerAgent(active_sources=self.active_sources)
        self._report_progress("init", "✓ Planner Agent ready")

        self.retriever = RetrievalAgent(self.vector_store)
        self._report_progress("init", "✓ Retrieval Agent ready")

        self.analyser  = AnalysisAgent()
        self._report_progress("init", "✓ Analysis Agent ready")

        if self.enable_critic:
            self.critic = CriticAgent()
            self._report_progress("init", "✓ Critic Agent ready")
        else:
            self.critic = None

        logger.info("✓ Orchestrator ready (sources=%s, critic=%s)",
                    self.active_sources, self.enable_critic)

    # ── Main Pipeline ─────────────────────────────────────────

    def run(
        self,
        query   : str,
        top_k   : int = 8,
    ) -> PipelineResult:
        """
        Run the full 4-agent pipeline for a research query.

        Args:
            query: The user's research question
            top_k: Number of chunks to retrieve

        Returns:
            PipelineResult with final response and all intermediate results
        """
        start_time = time.time()

        if not query or not query.strip():
            return self._error_result(query, "Query cannot be empty")

        if self.vector_store.get_stats()["total_docs"] == 0:
            return self._error_result(
                query,
                "No documents indexed yet! Please add documents, websites, or "
                "papers using the source tabs before running a query."
            )

        logger.info("="*50)
        logger.info("Running pipeline for: '%s'", query[:70])
        logger.info("="*50)

        try:
            # ── Step 1: Plan ──────────────────────────────────
            self._report_progress("planning", "🧠 Planning query decomposition...")
            plan = self.planner.plan(query)
            self._report_progress("planning",
                f"✓ Plan ready — intent: {plan.intent}, "
                f"{len(plan.sub_queries)} sub-queries"
            )
            logger.info("Step 1 complete — Plan: intent=%s", plan.intent)

            # ── Step 2: Retrieve ──────────────────────────────
            self._report_progress("retrieving", "🔍 Retrieving from all sources...")
            retrieval_result = self.retriever.retrieve(plan, top_k=top_k)
            self._report_progress("retrieving",
                f"✓ Retrieved {retrieval_result.total_retrieved} chunks "
                f"from {len(retrieval_result.by_source)} sources"
            )
            logger.info("Step 2 complete — Retrieved: %d chunks", retrieval_result.total_retrieved)

            if retrieval_result.total_retrieved == 0:
                return self._error_result(
                    query,
                    "No relevant content found for your query. Try adding more "
                    "documents or rephrasing your question.",
                    plan=plan,
                )

            # ── Step 3: Analyse ───────────────────────────────
            self._report_progress("analysing", "📝 Synthesizing response...")
            analysis_result = self.analyser.analyse(query, retrieval_result)
            self._report_progress("analysing",
                f"✓ Response synthesized — "
                f"{len(analysis_result.key_findings)} key findings, "
                f"{len(analysis_result.citations)} citations"
            )
            logger.info("Step 3 complete — Analysis: %d findings", len(analysis_result.key_findings))

            # ── Step 4: Critique ──────────────────────────────
            if self.enable_critic and self.critic:
                self._report_progress("critiquing", "🔬 Evaluating response quality...")
                critic_result = self.critic.evaluate(
                    query, analysis_result, retrieval_result,
                    analyser=self.analyser,
                )
                self._report_progress("critiquing",
                    f"✓ Quality score: {critic_result.overall_score:.1f}/10 "
                    f"({'APPROVED' if critic_result.approved else 'BEST ATTEMPT'})"
                    + (f" after {critic_result.retry_count} refinement(s)" if critic_result.retry_count > 0 else "")
                )
                logger.info("Step 4 complete — Score: %.1f/10 approved=%s retries=%d",
                            critic_result.overall_score, critic_result.approved,
                            critic_result.retry_count)

                final_response = critic_result.final_response or analysis_result.response
            else:
                critic_result  = None
                final_response = analysis_result.response

            # ── Build final result ────────────────────────────
            elapsed = round(time.time() - start_time, 2)

            result = PipelineResult(
                query            = query,
                final_response   = final_response,
                key_findings     = analysis_result.key_findings,
                citations        = [
                    {
                        "source_tab": c.source_tab,
                        "title"     : c.title,
                        "year"      : c.year,
                        "url"       : c.url,
                    }
                    for c in analysis_result.citations
                ],
                by_source        = analysis_result.by_source,
                overall_score    = critic_result.overall_score if critic_result else 0.0,
                approved         = critic_result.approved      if critic_result else True,
                has_conflicts    = analysis_result.has_conflicts,
                retry_count      = critic_result.retry_count   if critic_result else 0,
                elapsed_seconds  = elapsed,
                plan             = plan,
                retrieval_result = retrieval_result,
                analysis_result  = analysis_result,
                critic_result    = critic_result,
            )

            self._report_progress("complete",
                f"✅ Done in {elapsed:.1f}s — score: "
                f"{result.overall_score:.1f}/10"
            )
            logger.info("Pipeline complete in %.2fs", elapsed)
            return result

        except Exception as e:
            logger.error("Pipeline failed: %s", str(e), exc_info=True)
            return self._error_result(query, f"Pipeline error: {str(e)}")

    # ── Ingestion Methods ─────────────────────────────────────

    def ingest_document(
        self,
        file_path       : str,
        progress_callback: Optional[Callable] = None,
    ) -> Dict:
        """
        Ingest a PDF, DOCX, or TXT file into the vector store.

        Returns:
            {"success": bool, "chunks": int, "filename": str, "error": str}
        """
        from insighthub.ingestion.document_loader import DocumentLoader

        cb = progress_callback or self.progress_callback
        self._report_progress("ingesting", f"📄 Loading document: {Path(file_path).name}", cb)

        try:
            loader = DocumentLoader()
            chunks = loader.load_and_chunk(file_path)

            self._report_progress("ingesting", f"Embedding {len(chunks)} chunks...", cb)
            self.vector_store.add_documents_from_source(chunks, "document")

            stats = self.vector_store.get_stats()
            self._report_progress("ingesting",
                f"✓ Ingested {len(chunks)} chunks — "
                f"total: {stats['total_docs']} docs", cb
            )
            logger.info("Ingested document: %s (%d chunks)", file_path, len(chunks))

            return {
                "success" : True,
                "chunks"  : len(chunks),
                "filename": Path(file_path).name,
                "error"   : "",
            }

        except Exception as e:
            logger.error("Document ingestion failed: %s", str(e))
            return {"success": False, "chunks": 0, "filename": Path(file_path).name, "error": str(e)}

    def ingest_url(
        self,
        url             : str,
        progress_callback: Optional[Callable] = None,
    ) -> Dict:
        """
        Scrape and ingest a website URL into the vector store.

        Returns:
            {"success": bool, "chunks": int, "url": str, "error": str}
        """
        from insighthub.ingestion.web_loader import WebLoader

        cb = progress_callback or self.progress_callback
        self._report_progress("ingesting", f"🌐 Scraping: {url[:60]}...", cb)

        try:
            loader = WebLoader()
            chunks = loader.load_and_chunk(url)

            self._report_progress("ingesting", f"Embedding {len(chunks)} chunks...", cb)
            self.vector_store.add_documents_from_source(chunks, "website")

            stats = self.vector_store.get_stats()
            self._report_progress("ingesting",
                f"✓ Ingested {len(chunks)} chunks — "
                f"total: {stats['total_docs']} docs", cb
            )
            logger.info("Ingested URL: %s (%d chunks)", url, len(chunks))

            return {
                "success": True,
                "chunks" : len(chunks),
                "url"    : url,
                "error"  : "",
            }

        except Exception as e:
            logger.error("URL ingestion failed: %s", str(e))
            return {"success": False, "chunks": 0, "url": url, "error": str(e)}

    def ingest_paper(
        self,
        paper_id        : str,
        progress_callback: Optional[Callable] = None,
    ) -> Dict:
        """
        Download and ingest an arXiv paper into the vector store.

        Args:
            paper_id: arXiv ID (e.g. "2310.11511") or full URL

        Returns:
            {"success": bool, "chunks": int, "title": str, "error": str}
        """
        from insighthub.ingestion.paper_loader import PaperLoader

        cb = progress_callback or self.progress_callback
        self._report_progress("ingesting", f"🔬 Loading paper: arXiv:{paper_id}...", cb)

        try:
            loader   = PaperLoader()
            metadata = loader.get_metadata_only(paper_id)
            title    = metadata.get("title", paper_id)

            self._report_progress("ingesting", f"Downloading PDF: {title[:50]}...", cb)
            chunks = loader.load_and_chunk(paper_id)

            self._report_progress("ingesting", f"Embedding {len(chunks)} chunks...", cb)
            self.vector_store.add_documents_from_source(chunks, "paper")

            stats = self.vector_store.get_stats()
            self._report_progress("ingesting",
                f"✓ Ingested '{title[:40]}' ({len(chunks)} chunks) — "
                f"total: {stats['total_docs']} docs", cb
            )
            logger.info("Ingested paper: %s (%d chunks)", title, len(chunks))

            return {
                "success": True,
                "chunks" : len(chunks),
                "title"  : title,
                "error"  : "",
            }

        except Exception as e:
            logger.error("Paper ingestion failed: %s", str(e))
            return {"success": False, "chunks": 0, "title": paper_id, "error": str(e)}

    # ── Utility Methods ───────────────────────────────────────

    def get_stats(self) -> Dict:
        """Return current vector store statistics."""
        return self.vector_store.get_stats()

    def clear_index(self):
        """Clear all indexed documents."""
        self.vector_store.clear()
        logger.info("Vector store cleared")

    def set_active_sources(self, sources: List[str]):
        """Update which source tabs are active."""
        valid = [s for s in sources if s in SOURCE_TYPES]
        self.active_sources = valid
        self.planner.active_sources = valid
        logger.info("Active sources updated: %s", valid)

    # ── Private Methods ───────────────────────────────────────

    def _report_progress(
        self,
        step    : str,
        message : str,
        callback: Optional[Callable] = None,
    ):
        """Call progress callback if set."""
        cb = callback or self.progress_callback
        if cb:
            try:
                cb(step, message)
            except Exception:
                pass

    def _error_result(
        self,
        query : str,
        error : str,
        plan  : Optional[QueryPlan] = None,
    ) -> PipelineResult:
        """Return a failed pipeline result."""
        logger.error("Pipeline error: %s", error)
        return PipelineResult(
            query           = query,
            final_response  = f"⚠️ {error}",
            key_findings    = [],
            citations       = [],
            by_source       = {},
            overall_score   = 0.0,
            approved        = False,
            has_conflicts   = False,
            retry_count     = 0,
            elapsed_seconds = 0.0,
            plan            = plan,
            error           = error,
            success         = False,
        )


# ─────────────────────────────────────────────────────────────
#  QUICK TEST
#  Command: python insighthub/agents/orchestrator.py
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "="*55)
    print("  InsightHub — Orchestrator Test")
    print("="*55)

    # Progress callback for terminal display
    def show_progress(step: str, message: str):
        icons = {
            "init"      : "⚙️ ",
            "planning"  : "🧠",
            "retrieving": "🔍",
            "analysing" : "📝",
            "critiquing": "🔬",
            "ingesting" : "📥",
            "complete"  : "✅",
        }
        icon = icons.get(step, "  ")
        print(f"  {icon} {message}")

    # ── Initialise orchestrator ───────────────────────────────
    print("\n[1/3] Initialising orchestrator...")
    try:
        orch = Orchestrator(progress_callback=show_progress)
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    # ── Add sample docs if index is empty ─────────────────────
    stats = orch.get_stats()
    print(f"\n[2/3] Vector store status: {stats['total_docs']} docs indexed")

    if stats["total_docs"] == 0:
        print("  Adding sample documents for testing...")
        sample_docs = [
            Document(
                page_content = "RAG (Retrieval-Augmented Generation) reduces hallucinations by grounding language model responses in retrieved documents. It retrieves relevant evidence from a knowledge base and conditions generation on that evidence, making responses factually accurate.",
                metadata     = {"source_tab": "paper", "title": "RAG Paper", "year": "2020", "chunk_id": 0}
            ),
            Document(
                page_content = "Self-RAG extends RAG with adaptive retrieval using special reflection tokens. The model learns when to retrieve, how to critique retrieved passages, and how to generate well-grounded responses.",
                metadata     = {"source_tab": "paper", "title": "Self-RAG", "year": "2023", "chunk_id": 1}
            ),
            Document(
                page_content = "LangChain provides a complete framework for building RAG applications. It supports multiple vector stores (FAISS, Chroma, Pinecone), document loaders, and retrieval chains.",
                metadata     = {"source_tab": "website", "domain": "langchain.com", "page_title": "LangChain RAG", "scraped_date": "2024-01-10", "chunk_id": 2}
            ),
            Document(
                page_content = "FAISS (Facebook AI Similarity Search) enables fast approximate nearest neighbor search over dense vector embeddings. It supports billion-scale datasets and runs efficiently on CPU.",
                metadata     = {"source_tab": "document", "filename": "faiss_notes.pdf", "year": "2024", "chunk_id": 3}
            ),
            Document(
                page_content = "Multi-agent AI systems use specialized agents working together to solve complex tasks. Each agent handles a specific subtask: planning, retrieval, analysis, or evaluation.",
                metadata     = {"source_tab": "document", "filename": "ai_agents.txt", "year": "2024", "chunk_id": 4}
            ),
        ]
        orch.vector_store.add_documents(sample_docs)
        print(f"  ✓ Added {len(sample_docs)} sample documents")

    # ── Run full pipeline ─────────────────────────────────────
    print("\n[3/3] Running full pipeline...")
    query = "How does RAG reduce hallucinations and what are its key benefits?"
    print(f"  Query: '{query}'")
    print("-"*55)

    result = orch.run(query)

    print("\n" + "="*55)
    if result.success:
        print(f"  ✅ Pipeline succeeded in {result.elapsed_seconds:.1f}s")
        print(f"  📈 Quality score : {result.overall_score:.1f}/10")
        print(f"  ✅ Approved      : {result.approved}")
        print(f"  🔄 Refinements   : {result.retry_count}")
        print(f"  📚 Citations     : {len(result.citations)}")
        print(f"  📌 Key findings  : {len(result.key_findings)}")

        print(f"\n  📝 Final Response:")
        print("  " + "-"*45)
        for line in result.final_response.split("\n"):
            print(f"  {line}")

        print(f"\n  📌 Key Findings:")
        for i, f in enumerate(result.key_findings, 1):
            print(f"    {i}. {f}")

        print(f"\n  📚 Citations:")
        for c in result.citations:
            year = f" ({c['year']})" if c.get("year") else ""
            print(f"    [{c['source_tab'].upper()}] {c['title']}{year}")

        if result.has_conflicts:
            print(f"\n  ⚠️  Conflicts detected between sources!")

        if result.plan:
            print(f"\n  🧠 Query intent  : {result.plan.intent}")

    else:
        print(f"  ✗ Pipeline failed: {result.error}")

    print("="*55 + "\n")