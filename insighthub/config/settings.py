# ============================================================
#  InsightHub — config/settings.py
#  Central configuration for all models, paths & thresholds
#  Stack: HuggingFace (FREE) — no OpenAI key needed
# ============================================================

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()

# ── Project Paths ────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
DATA_DIR        = BASE_DIR / "data"
VECTOR_STORE_DIR= BASE_DIR / "vector_store"
UPLOADS_DIR     = DATA_DIR / "uploads"
EXPORTS_DIR     = BASE_DIR / "exports"

# Create directories if they don't exist
for d in [DATA_DIR, VECTOR_STORE_DIR, UPLOADS_DIR, EXPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── HuggingFace API (Free) ───────────────────────────────────
# Get your free token at: https://huggingface.co/settings/tokens
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "")

# ── Cohere API (Free tier available) ─────────────────────────
# Get free key at: https://dashboard.cohere.com  (1000 calls/month free)
# Leave empty to skip reranking — system still works without it
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

# ── LangSmith (Free for personal use) ────────────────────────
# Sign up at: https://smith.langchain.com
# Enables agent trace logging — highly recommended for debugging
LANGCHAIN_API_KEY      = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_TRACING_V2   = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_PROJECT       = os.getenv("LANGCHAIN_PROJECT", "insighthub")

# Set LangSmith env vars if key is provided
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_API_KEY"]    = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
    os.environ["LANGCHAIN_PROJECT"]    = LANGCHAIN_PROJECT

# ── Embedding Model (FREE via HuggingFace) ───────────────────
# gte-large: excellent multilingual embeddings, runs locally, no API needed
EMBEDDING_MODEL      = "thenlper/gte-large"
EMBEDDING_DIMENSION  = 1024         # gte-large output dimension
EMBEDDING_DEVICE     = "cpu"        # Change to "cuda" if you have a GPU

# ── LLM Model (FREE via HuggingFace Inference API) ───────────
# Mistral-7B-Instruct: best free model for instruction following
# Runs via HuggingFace Inference API — no local GPU needed
LLM_MODEL            = "mistralai/Mistral-7B-Instruct-v0.3"
LLM_MAX_NEW_TOKENS   = 1024
LLM_TEMPERATURE      = 0.1          # Low temp = more factual, less creative
LLM_TOP_P            = 0.9

# Alternative FREE models (uncomment to switch):
# LLM_MODEL = "google/flan-t5-large"          # Smaller, faster, less capable
# LLM_MODEL = "HuggingFaceH4/zephyr-7b-beta"  # Good instruction follower
# LLM_MODEL = "microsoft/Phi-3-mini-4k-instruct"  # Lightweight and capable

# ── Chunking Settings ─────────────────────────────────────────
CHUNK_SIZE_DOCUMENT = 512           # tokens per chunk for documents
CHUNK_SIZE_WEB      = 800           # tokens per chunk for web pages
CHUNK_SIZE_PAPER    = 512           # tokens per chunk for research papers
CHUNK_OVERLAP       = 50            # overlap between consecutive chunks

# ── Retrieval Settings ────────────────────────────────────────
RETRIEVAL_TOP_K         = 10        # number of candidates to retrieve
RETRIEVAL_FINAL_K       = 5         # number after reranking
SIMILARITY_THRESHOLD    = 0.65      # minimum cosine similarity score
MMR_DIVERSITY_SCORE     = 0.3       # MMR lambda (0=max diversity, 1=max relevance)
BM25_WEIGHT             = 0.3       # weight for BM25 in hybrid search (0-1)
DENSE_WEIGHT            = 0.7       # weight for dense search in hybrid search

# ── Critic Agent Thresholds ───────────────────────────────────
CRITIC_RELEVANCE_THRESHOLD    = 7.0
CRITIC_COHERENCE_THRESHOLD    = 7.0
CRITIC_GROUNDING_THRESHOLD    = 8.0
CRITIC_COMPLETENESS_THRESHOLD = 6.5
CRITIC_MAX_RETRIES            = 2   # max refinement cycles before accepting

# ── Source Tab Settings ───────────────────────────────────────
SOURCE_TYPES = ["document", "website", "paper"]

DOCUMENT_EXTENSIONS = [".pdf", ".docx", ".txt", ".doc"]
PAPER_ID_PATTERNS   = ["arxiv", "doi", "10."]   # patterns to detect paper IDs

# ── Vector Store ─────────────────────────────────────────────
VECTOR_STORE_TYPE   = "faiss"       # options: "faiss", "chroma"
FAISS_INDEX_NAME    = "insighthub_index"
FAISS_INDEX_PATH    = str(VECTOR_STORE_DIR / FAISS_INDEX_NAME)

# ── Web Scraping ─────────────────────────────────────────────
REQUEST_TIMEOUT     = 15            # seconds before request times out
MAX_PAGE_CHARS      = 50000         # max characters to keep from a webpage
USE_PLAYWRIGHT      = True          # fallback to Playwright for JS pages

# ── Export Settings ───────────────────────────────────────────
EXPORT_DIR          = str(EXPORTS_DIR)
DEFAULT_EXPORT_FORMAT = "markdown"  # options: "markdown", "pdf"

# ── Streamlit UI ─────────────────────────────────────────────
APP_TITLE           = "InsightHub"
APP_SUBTITLE        = "Multi-Agent RAG · Research Assistant"
APP_ICON            = "🔭"

# ── Validation helper ────────────────────────────────────────
def validate_config():
    """Check critical settings and warn about missing optional ones."""
    warnings = []
    errors   = []

    if not HUGGINGFACE_API_TOKEN:
        errors.append(
            "HUGGINGFACE_API_TOKEN is not set.\n"
            "  → Get your free token at: https://huggingface.co/settings/tokens\n"
            "  → Add to .env file: HUGGINGFACE_API_TOKEN=hf_xxxx"
        )

    if not COHERE_API_KEY:
        warnings.append(
            "COHERE_API_KEY not set — reranking disabled.\n"
            "  → System will still work, but retrieval quality may be lower.\n"
            "  → Free key at: https://dashboard.cohere.com"
        )

    if not LANGCHAIN_API_KEY:
        warnings.append(
            "LANGCHAIN_API_KEY not set — LangSmith tracing disabled.\n"
            "  → Recommended for debugging agents during development.\n"
            "  → Free at: https://smith.langchain.com"
        )

    if errors:
        print("\n[InsightHub Config] ERRORS — Fix before running:")
        for e in errors:
            print(f"  ✗ {e}")

    if warnings:
        print("\n[InsightHub Config] WARNINGS — Optional but recommended:")
        for w in warnings:
            print(f"  ⚠  {w}")

    if not errors:
        print("[InsightHub Config] ✓ Core configuration valid.")

    return len(errors) == 0


if __name__ == "__main__":
    validate_config()
    print(f"\n  Embedding model : {EMBEDDING_MODEL}")
    print(f"  LLM model       : {LLM_MODEL}")
    print(f"  Vector store    : {VECTOR_STORE_TYPE.upper()} → {FAISS_INDEX_PATH}")
    print(f"  Chunk size      : {CHUNK_SIZE_DOCUMENT} tokens (overlap: {CHUNK_OVERLAP})")
    print(f"  Retrieval top-k : {RETRIEVAL_TOP_K} → reranked to {RETRIEVAL_FINAL_K}")