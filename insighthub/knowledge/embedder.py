# ============================================================
#  InsightHub — insighthub/knowledge/embedder.py
#  Converts text chunks into dense vectors using gte-large
#  Model: thenlper/gte-large (FREE, runs locally via HuggingFace)
# ============================================================

import sys
import logging
from pathlib import Path
from typing import List, Union

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings

from insighthub.config.settings import (
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    EMBEDDING_DEVICE,
    HUGGINGFACE_API_TOKEN,
)

# ── Logging setup ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  EMBEDDER CLASS
# ─────────────────────────────────────────────────────────────

class Embedder:
    """
    Converts text chunks into dense vector embeddings using gte-large.

    gte-large is a state-of-the-art multilingual embedding model from
    HuggingFace that runs completely locally — no API calls needed
    after the first download (~670MB, downloaded once automatically).

    Usage:
        embedder = Embedder()

        # Embed a single text
        vector = embedder.embed_text("What is RAG?")

        # Embed multiple texts
        vectors = embedder.embed_texts(["text 1", "text 2"])

        # Get the LangChain embeddings object (used by vector store)
        lc_embeddings = embedder.get_langchain_embeddings()
    """

    def __init__(self):
        logger.info("Loading embedding model: %s (device: %s)",
                    EMBEDDING_MODEL, EMBEDDING_DEVICE)
        logger.info("First run will download ~670MB — subsequent runs load from cache")

        self._embeddings = self._load_embeddings()
        logger.info("✓ Embedding model loaded successfully")

    # ── Public Methods ────────────────────────────────────────

    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string into a dense vector.

        Returns:
            List of floats with length EMBEDDING_DIMENSION (1024 for gte-large)
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        vector = self._embeddings.embed_query(text)
        logger.debug("Embedded text (%d chars) → vector dim: %d", len(text), len(vector))
        return vector

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts in batch (more efficient than one by one).

        Returns:
            List of vectors, one per input text
        """
        if not texts:
            raise ValueError("Cannot embed empty list")

        # Filter out empty texts
        clean_texts = [t for t in texts if t and t.strip()]
        if not clean_texts:
            raise ValueError("All texts are empty")

        logger.info("Embedding %d texts in batch...", len(clean_texts))
        vectors = self._embeddings.embed_documents(clean_texts)
        logger.info("✓ Embedded %d texts → vectors of dim %d",
                    len(vectors), len(vectors[0]) if vectors else 0)
        return vectors

    def embed_documents(self, docs: List[Document]) -> List[List[float]]:
        """
        Embed a list of LangChain Documents.
        Extracts page_content from each document and embeds in batch.

        Returns:
            List of vectors, one per document
        """
        if not docs:
            raise ValueError("Cannot embed empty document list")

        texts = [doc.page_content for doc in docs]
        return self.embed_texts(texts)

    def get_langchain_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Return the underlying LangChain embeddings object.
        This is what gets passed to FAISS and ChromaDB vector stores.

        Usage:
            embeddings = embedder.get_langchain_embeddings()
            vector_store = FAISS.from_documents(docs, embeddings)
        """
        return self._embeddings

    def get_embedding_dimension(self) -> int:
        """Return the dimension of the embedding vectors."""
        return EMBEDDING_DIMENSION

    def test_embedding(self) -> bool:
        """
        Quick sanity check — embeds a test sentence and verifies output.
        Returns True if working correctly, False otherwise.
        """
        try:
            test_text = "This is a test sentence for InsightHub."
            vector    = self.embed_text(test_text)

            assert len(vector) == EMBEDDING_DIMENSION, (
                f"Expected dimension {EMBEDDING_DIMENSION}, got {len(vector)}"
            )
            assert all(isinstance(v, float) for v in vector[:5]), (
                "Vector values should be floats"
            )

            logger.info("✓ Embedding test passed (dim=%d)", len(vector))
            return True

        except Exception as e:
            logger.error("✗ Embedding test failed: %s", str(e))
            return False

    # ── Private Methods ───────────────────────────────────────

    def _load_embeddings(self) -> HuggingFaceEmbeddings:
        """Load the gte-large model via LangChain HuggingFaceEmbeddings."""
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name = EMBEDDING_MODEL,
                model_kwargs = {
                    "device": EMBEDDING_DEVICE,
                    "trust_remote_code": True,
                },
                encode_kwargs = {
                    "normalize_embeddings": True,   # normalise to unit length
                    "batch_size": 32,               # process 32 chunks at once
                },
                cache_folder = str(
                    Path(__file__).resolve().parent.parent.parent / ".cache" / "embeddings"
                ),
            )
            return embeddings

        except Exception as e:
            logger.error("Failed to load embedding model: %s", str(e))
            logger.error("Make sure you have run: pip install sentence-transformers")
            raise


# ─────────────────────────────────────────────────────────────
#  QUICK TEST
#  Command: python insighthub/knowledge/embedder.py
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "="*55)
    print("  InsightHub — Embedder Test")
    print("="*55)
    print("  Model  : thenlper/gte-large")
    print("  Device : CPU")
    print("  Note   : First run downloads ~670MB model")
    print("="*55)

    # ── Load embedder ─────────────────────────────────────────
    print("\n[1/4] Loading embedding model...")
    try:
        embedder = Embedder()
        print("  ✓ Model loaded successfully")
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        sys.exit(1)

    # ── Test single text embedding ────────────────────────────
    print("\n[2/4] Testing single text embedding...")
    try:
        text   = "Retrieval-Augmented Generation combines retrieval with language model generation."
        vector = embedder.embed_text(text)
        print(f"  ✓ Input text   : '{text[:60]}...'")
        print(f"  ✓ Vector dim   : {len(vector)}")
        print(f"  ✓ First 5 vals : {[round(v, 4) for v in vector[:5]]}")
    except Exception as e:
        print(f"  ✗ Single embedding failed: {e}")

    # ── Test batch embedding ──────────────────────────────────
    print("\n[3/4] Testing batch embedding...")
    try:
        texts = [
            "RAG systems retrieve documents from a knowledge base.",
            "The Critic Agent evaluates response quality on 4 dimensions.",
            "FAISS is a fast similarity search library for dense vectors.",
            "LangGraph enables cyclical multi-agent workflows.",
        ]
        vectors = embedder.embed_texts(texts)
        print(f"  ✓ Input texts  : {len(texts)}")
        print(f"  ✓ Output vecs  : {len(vectors)}")
        print(f"  ✓ Vector dim   : {len(vectors[0])}")
    except Exception as e:
        print(f"  ✗ Batch embedding failed: {e}")

    # ── Test similarity check ─────────────────────────────────
    print("\n[4/4] Testing semantic similarity...")
    try:
        import numpy as np

        q   = embedder.embed_text("How does RAG reduce hallucinations?")
        d1  = embedder.embed_text("RAG grounds responses in retrieved documents, reducing hallucinations.")
        d2  = embedder.embed_text("The weather in London is often cloudy and rainy.")

        sim1 = np.dot(q, d1)   # should be HIGH (related)
        sim2 = np.dot(q, d2)   # should be LOW  (unrelated)

        print(f"  ✓ Query vs relevant doc   : {sim1:.4f}  (should be HIGH > 0.7)")
        print(f"  ✓ Query vs irrelevant doc : {sim2:.4f}  (should be LOW  < 0.5)")

        if sim1 > sim2:
            print("  ✓ Semantic similarity is working correctly!")
        else:
            print("  ⚠ Unexpected similarity scores — check model loading")

    except Exception as e:
        print(f"  ✗ Similarity test failed: {e}")

    print("\n" + "="*55)
    print("  Embedder Test Complete!")
    print("="*55 + "\n")