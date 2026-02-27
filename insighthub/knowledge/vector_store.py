# ============================================================
#  InsightHub — insighthub/knowledge/vector_store.py
#  Stores and searches embedded chunks using FAISS
#  Supports: add documents, similarity search, hybrid search
# ============================================================

import sys
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi

from insighthub.config.settings import (
    FAISS_INDEX_PATH,
    RETRIEVAL_TOP_K,
    RETRIEVAL_FINAL_K,
    SIMILARITY_THRESHOLD,
    BM25_WEIGHT,
    DENSE_WEIGHT,
)
from insighthub.knowledge.embedder import Embedder

# ── Logging setup ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  VECTOR STORE CLASS
# ─────────────────────────────────────────────────────────────

class VectorStore:
    """
    Manages the FAISS vector store for InsightHub.

    Handles:
        - Adding chunks from all 3 source types (documents, websites, papers)
        - Saving and loading the index to/from disk
        - Dense similarity search (FAISS)
        - Hybrid search (dense + BM25 keyword, fused with RRF)
        - Metadata filtering by source tab

    Usage:
        store = VectorStore()

        # Add documents
        store.add_documents(chunks)

        # Search
        results = store.hybrid_search("What is RAG?", k=5)
        results = store.search_by_source("What is RAG?", source_tab="paper", k=5)
    """

    def __init__(self):
        self.embedder    = Embedder()
        self.embeddings  = self.embedder.get_langchain_embeddings()
        self._faiss      = None        # FAISS index
        self._all_docs   : List[Document] = []   # all documents (for BM25)
        self._bm25       = None        # BM25 index
        self._index_path = FAISS_INDEX_PATH
        self._meta_path  = FAISS_INDEX_PATH + "_meta.json"

        # Load existing index if available
        if self._index_exists():
            self._load()
        else:
            logger.info("No existing index found — starting fresh")

    # ── Adding Documents ──────────────────────────────────────

    def add_documents(self, docs: List[Document]) -> int:
        """
        Add chunked documents to the vector store.
        Automatically rebuilds BM25 index after adding.

        Returns:
            Number of documents successfully added
        """
        if not docs:
            logger.warning("No documents to add")
            return 0

        logger.info("Adding %d documents to vector store...", len(docs))

        try:
            if self._faiss is None:
                # Create new FAISS index from scratch
                self._faiss = FAISS.from_documents(docs, self.embeddings)
                logger.info("Created new FAISS index with %d documents", len(docs))
            else:
                # Add to existing index
                self._faiss.add_documents(docs)
                logger.info("Added %d documents to existing FAISS index", len(docs))

            # Track all docs for BM25
            self._all_docs.extend(docs)

            # Rebuild BM25 index
            self._build_bm25()

            # Save updated index to disk
            self._save()

            logger.info("✓ Vector store now contains %d total documents", len(self._all_docs))
            return len(docs)

        except Exception as e:
            logger.error("Failed to add documents: %s", str(e))
            raise

    def add_documents_from_source(
        self,
        docs: List[Document],
        source_tab: str
    ) -> int:
        """
        Add documents from a specific source tab.
        Ensures source_tab metadata is set correctly.
        """
        # Make sure source_tab is set in metadata
        for doc in docs:
            doc.metadata["source_tab"] = source_tab

        return self.add_documents(docs)

    # ── Search Methods ────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        k: int = RETRIEVAL_TOP_K,
        filter_metadata: Optional[Dict] = None,
    ) -> List[Document]:
        """
        Pure dense vector similarity search using FAISS.

        Args:
            query:           Search query string
            k:               Number of results to return
            filter_metadata: Optional dict to filter by metadata field
                             e.g. {"source_tab": "paper"}
        """
        self._check_index()

        try:
            if filter_metadata:
                results = self._faiss.similarity_search(
                    query, k=k, filter=filter_metadata
                )
            else:
                results = self._faiss.similarity_search(query, k=k)

            logger.info("Dense search: %d results for '%s'", len(results), query[:50])
            return results

        except Exception as e:
            logger.error("Similarity search failed: %s", str(e))
            raise

    def hybrid_search(
        self,
        query: str,
        k: int = RETRIEVAL_FINAL_K,
        filter_metadata: Optional[Dict] = None,
    ) -> List[Document]:
        """
        Hybrid search combining dense FAISS + BM25 keyword search.
        Results fused using Reciprocal Rank Fusion (RRF).

        This gives better results than pure dense search because:
        - Dense search captures semantic meaning
        - BM25 captures exact keyword matches
        - RRF fusion combines the best of both

        Args:
            query:           Search query string
            k:               Number of final results to return
            filter_metadata: Optional metadata filter
        """
        self._check_index()

        # Get more candidates than needed for fusion
        candidate_k = min(k * 4, len(self._all_docs))

        # ── Dense search ──────────────────────────────────────
        try:
            dense_results = self.similarity_search(query, k=candidate_k, filter_metadata=filter_metadata)
        except Exception as e:
            logger.warning("Dense search failed, falling back to BM25 only: %s", e)
            dense_results = []

        # ── BM25 keyword search ───────────────────────────────
        try:
            bm25_results = self._bm25_search(query, k=candidate_k, filter_metadata=filter_metadata)
        except Exception as e:
            logger.warning("BM25 search failed, using dense only: %s", e)
            bm25_results = []

        # ── Reciprocal Rank Fusion ────────────────────────────
        fused = self._reciprocal_rank_fusion(
            dense_results, bm25_results,
            dense_weight=DENSE_WEIGHT,
            bm25_weight=BM25_WEIGHT,
        )

        results = fused[:k]
        logger.info("Hybrid search: %d results for '%s'", len(results), query[:50])
        return results

    def search_by_source(
        self,
        query: str,
        source_tab: str,
        k: int = RETRIEVAL_FINAL_K,
    ) -> List[Document]:
        """
        Search within a specific source tab only.

        Args:
            source_tab: "document", "website", or "paper"
        """
        return self.hybrid_search(
            query,
            k=k,
            filter_metadata={"source_tab": source_tab},
        )

    def search_all_sources(
        self,
        query: str,
        k_per_source: int = 3,
    ) -> Dict[str, List[Document]]:
        """
        Search across all 3 source tabs separately.
        Returns results grouped by source tab.
        Used by the Planner Agent for per-source retrieval.

        Returns:
            {
                "document": [...],
                "website":  [...],
                "paper":    [...],
            }
        """
        results = {}
        for source_tab in ["document", "website", "paper"]:
            try:
                docs = self.search_by_source(query, source_tab, k=k_per_source)
                results[source_tab] = docs
                logger.info("Source '%s': %d results", source_tab, len(docs))
            except Exception as e:
                logger.warning("Search failed for source '%s': %s", source_tab, e)
                results[source_tab] = []

        return results

    # ── Index Management ──────────────────────────────────────

    def get_stats(self) -> Dict:
        """Return statistics about the current index."""
        if self._faiss is None:
            return {"total_docs": 0, "by_source": {}, "status": "empty"}

        by_source = {}
        for doc in self._all_docs:
            tab = doc.metadata.get("source_tab", "unknown")
            by_source[tab] = by_source.get(tab, 0) + 1

        return {
            "total_docs" : len(self._all_docs),
            "by_source"  : by_source,
            "index_path" : self._index_path,
            "status"     : "ready",
        }

    def clear(self):
        """Clear the entire index — removes all documents."""
        self._faiss    = None
        self._all_docs = []
        self._bm25     = None

        # Remove saved files
        import shutil
        faiss_dir = Path(self._index_path)
        if faiss_dir.exists():
            shutil.rmtree(faiss_dir)

        meta_file = Path(self._meta_path)
        if meta_file.exists():
            meta_file.unlink()

        logger.info("Vector store cleared")

    # ── BM25 Methods ──────────────────────────────────────────

    def _build_bm25(self):
        """Build BM25 index from all current documents."""
        if not self._all_docs:
            return

        # Tokenise all documents for BM25
        tokenised = [
            doc.page_content.lower().split()
            for doc in self._all_docs
        ]
        self._bm25 = BM25Okapi(tokenised)
        logger.debug("BM25 index built with %d documents", len(self._all_docs))

    def _bm25_search(
        self,
        query: str,
        k: int,
        filter_metadata: Optional[Dict] = None,
    ) -> List[Document]:
        """BM25 keyword search over all documents."""
        if self._bm25 is None or not self._all_docs:
            return []

        # Get BM25 scores for all documents
        tokenised_query = query.lower().split()
        scores          = self._bm25.get_scores(tokenised_query)

        # Pair scores with documents and sort
        scored_docs = sorted(
            zip(scores, self._all_docs),
            key=lambda x: x[0],
            reverse=True,
        )

        # Apply metadata filter if provided
        results = []
        for score, doc in scored_docs:
            if score <= 0:
                continue
            if filter_metadata:
                match = all(
                    doc.metadata.get(key) == val
                    for key, val in filter_metadata.items()
                )
                if not match:
                    continue
            results.append(doc)
            if len(results) >= k:
                break

        return results

    # ── Reciprocal Rank Fusion ────────────────────────────────

    def _reciprocal_rank_fusion(
        self,
        dense_results : List[Document],
        bm25_results  : List[Document],
        dense_weight  : float = 0.7,
        bm25_weight   : float = 0.3,
        rrf_k         : int   = 60,
    ) -> List[Document]:
        """
        Fuse dense and BM25 results using Reciprocal Rank Fusion.

        RRF score = dense_weight * 1/(rank+k) + bm25_weight * 1/(rank+k)
        Higher score = better combined result.
        """
        scores = {}
        doc_map = {}

        # Score dense results
        for rank, doc in enumerate(dense_results):
            doc_id = self._doc_id(doc)
            scores[doc_id]  = scores.get(doc_id, 0) + dense_weight * (1 / (rank + rrf_k))
            doc_map[doc_id] = doc

        # Score BM25 results
        for rank, doc in enumerate(bm25_results):
            doc_id = self._doc_id(doc)
            scores[doc_id]  = scores.get(doc_id, 0) + bm25_weight * (1 / (rank + rrf_k))
            doc_map[doc_id] = doc

        # Sort by combined score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [doc_map[doc_id] for doc_id in sorted_ids]

    def _doc_id(self, doc: Document) -> str:
        """Generate a unique ID for a document based on content hash."""
        return str(hash(doc.page_content[:100]))

    # ── Save / Load ───────────────────────────────────────────

    def _save(self):
        """Save FAISS index and metadata to disk."""
        try:
            # Save FAISS index
            Path(self._index_path).mkdir(parents=True, exist_ok=True)
            self._faiss.save_local(self._index_path)

            # Save metadata (all doc metadata for BM25 reconstruction)
            meta = [
                {
                    "page_content": doc.page_content,
                    "metadata":     doc.metadata,
                }
                for doc in self._all_docs
            ]
            with open(self._meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            logger.info("✓ Index saved to %s (%d docs)", self._index_path, len(self._all_docs))

        except Exception as e:
            logger.error("Failed to save index: %s", str(e))

    def _load(self):
        """Load FAISS index and metadata from disk."""
        try:
            logger.info("Loading existing index from %s...", self._index_path)

            # Load FAISS index
            self._faiss = FAISS.load_local(
                self._index_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

            # Load document metadata
            if Path(self._meta_path).exists():
                with open(self._meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                self._all_docs = [
                    Document(
                        page_content = m["page_content"],
                        metadata     = m["metadata"],
                    )
                    for m in meta
                ]

                # Rebuild BM25 index
                self._build_bm25()

            logger.info("✓ Loaded index with %d documents", len(self._all_docs))

        except Exception as e:
            logger.error("Failed to load index: %s — starting fresh", str(e))
            self._faiss    = None
            self._all_docs = []

    def _index_exists(self) -> bool:
        """Check if a saved index exists on disk."""
        return (
            Path(self._index_path).exists() and
            len(list(Path(self._index_path).glob("*"))) > 0
        )

    def _check_index(self):
        """Raise error if index is empty."""
        if self._faiss is None or not self._all_docs:
            raise RuntimeError(
                "Vector store is empty!\n"
                "Add documents first using: store.add_documents(chunks)"
            )


# ─────────────────────────────────────────────────────────────
#  QUICK TEST
#  Command: python insighthub/knowledge/vector_store.py
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "="*55)
    print("  InsightHub — Vector Store Test")
    print("="*55)

    # ── Sample documents ──────────────────────────────────────
    sample_docs = [
        Document(
            page_content = "RAG (Retrieval-Augmented Generation) combines retrieval with generation to reduce hallucinations in LLMs.",
            metadata     = {"source_tab": "paper", "title": "RAG Paper", "chunk_id": 0}
        ),
        Document(
            page_content = "Self-RAG introduces adaptive retrieval using reflection tokens, allowing the model to decide when to retrieve.",
            metadata     = {"source_tab": "paper", "title": "Self-RAG", "chunk_id": 1}
        ),
        Document(
            page_content = "LangChain is an open-source framework for building applications with large language models.",
            metadata     = {"source_tab": "website", "domain": "langchain.com", "chunk_id": 2}
        ),
        Document(
            page_content = "FAISS (Facebook AI Similarity Search) enables fast nearest neighbor search in high-dimensional spaces.",
            metadata     = {"source_tab": "document", "filename": "faiss_notes.pdf", "chunk_id": 3}
        ),
        Document(
            page_content = "The Critic Agent in InsightHub evaluates responses on Relevance, Coherence, Grounding, and Completeness.",
            metadata     = {"source_tab": "document", "filename": "insighthub_notes.txt", "chunk_id": 4}
        ),
    ]

    print("\n[1/5] Initialising vector store...")
    try:
        store = VectorStore()
        print("  ✓ Vector store initialised")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        sys.exit(1)

    print("\n[2/5] Adding sample documents...")
    try:
        count = store.add_documents(sample_docs)
        stats = store.get_stats()
        print(f"  ✓ Added {count} documents")
        print(f"  ✓ Total in store : {stats['total_docs']}")
        print(f"  ✓ By source      : {stats['by_source']}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    print("\n[3/5] Testing dense similarity search...")
    try:
        results = store.similarity_search("How does RAG reduce hallucinations?", k=2)
        print(f"  ✓ Results: {len(results)}")
        for i, r in enumerate(results):
            print(f"  ✓ Result {i+1}: '{r.page_content[:80]}...'")
            print(f"             Source: {r.metadata.get('source_tab')}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    print("\n[4/5] Testing hybrid search (dense + BM25)...")
    try:
        results = store.hybrid_search("adaptive retrieval reflection tokens", k=2)
        print(f"  ✓ Results: {len(results)}")
        for i, r in enumerate(results):
            print(f"  ✓ Result {i+1}: '{r.page_content[:80]}...'")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    print("\n[5/5] Testing search by source tab...")
    try:
        paper_results = store.search_by_source("retrieval generation", source_tab="paper", k=2)
        print(f"  ✓ Paper results  : {len(paper_results)}")
        web_results   = store.search_by_source("LangChain framework", source_tab="website", k=2)
        print(f"  ✓ Website results: {len(web_results)}")
        doc_results   = store.search_by_source("FAISS similarity", source_tab="document", k=2)
        print(f"  ✓ Document results: {len(doc_results)}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    print("\n" + "="*55)
    print("  Vector Store Test Complete!")
    print("="*55 + "\n")