# ============================================================
#  InsightHub — insighthub/ingestion/paper_loader.py
#  Handles: Research Papers via arXiv ID or DOI
#  Parser:  arXiv API (metadata) + PyMuPDF (PDF content)
# ============================================================

import sys
import re
import time
import logging
import tempfile
import os
from pathlib import Path
from typing import List, Optional, Dict

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import requests
import arxiv
import fitz  # PyMuPDF
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from insighthub.config.settings import (
    CHUNK_SIZE_DOCUMENT,
    CHUNK_OVERLAP,
    REQUEST_TIMEOUT,
)

# ── Logging setup ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  MAIN LOADER CLASS
# ─────────────────────────────────────────────────────────────

class PaperLoader:
    """
    Loads and chunks research papers from arXiv into LangChain Documents.

    Workflow:
        1. Fetch paper metadata via arXiv API (title, authors, abstract, year)
        2. Download the PDF directly from arXiv
        3. Extract text section by section using PyMuPDF
        4. Chunk with section-aware splitting
        5. Attach rich academic metadata to every chunk

    Usage:
        loader = PaperLoader()

        # Load by arXiv ID
        chunks = loader.load_and_chunk("2310.11511")       # Self-RAG
        chunks = loader.load_and_chunk("1706.03762")       # Attention Is All You Need

        # Load by full arXiv URL
        chunks = loader.load_and_chunk("https://arxiv.org/abs/2312.10997")

        # Load multiple papers
        chunks = loader.load_multiple(["2310.11511", "1706.03762", "2308.08155"])
    """

    # Common section headers in research papers
    SECTION_HEADERS = [
        "abstract", "introduction", "related work", "background",
        "methodology", "methods", "approach", "model", "architecture",
        "experiments", "experimental setup", "results", "evaluation",
        "discussion", "conclusion", "conclusions", "future work",
        "references", "appendix", "acknowledgements",
    ]

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size      = CHUNK_SIZE_DOCUMENT,
            chunk_overlap   = CHUNK_OVERLAP,
            separators      = ["\n\n", "\n", ". ", " ", ""],
            length_function = len,
        )
        # arXiv API client
        self.arxiv_client = arxiv.Client(
            page_size      = 10,
            delay_seconds  = 3,    # polite delay between requests
            num_retries    = 3,
        )
        logger.info("PaperLoader initialised (chunk_size=%d, overlap=%d)",
                    CHUNK_SIZE_DOCUMENT, CHUNK_OVERLAP)

    # ── Public Methods ────────────────────────────────────────

    def load(self, paper_id: str) -> List[Document]:
        """
        Load a research paper by arXiv ID or URL.
        Returns Documents with section-aware content and rich metadata.
        """
        # Clean and normalise the paper ID
        arxiv_id = self._extract_arxiv_id(paper_id)
        logger.info("Loading paper: arXiv:%s", arxiv_id)

        # Step 1 — Fetch metadata from arXiv API
        metadata = self._fetch_metadata(arxiv_id)
        logger.info("Fetched metadata: '%s' by %s (%s)",
                    metadata["title"], metadata["authors_short"], metadata["year"])

        # Step 2 — Download and parse PDF
        docs = self._download_and_parse(arxiv_id, metadata)
        logger.info("Extracted %d sections from arXiv:%s", len(docs), arxiv_id)

        return docs

    def load_and_chunk(self, paper_id: str) -> List[Document]:
        """
        Load a paper and split into chunks ready for embedding.
        This is the main method used by the ingestion pipeline.
        """
        docs   = self.load(paper_id)
        chunks = self.splitter.split_documents(docs)

        # Add chunk index to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"]     = i
            chunk.metadata["total_chunks"] = len(chunks)
            chunk.metadata["source_tab"]   = "paper"

        logger.info("Split into %d chunks from arXiv:%s",
                    len(chunks), self._extract_arxiv_id(paper_id))
        return chunks

    def load_multiple(self, paper_ids: List[str], delay: float = 3.0) -> List[Document]:
        """
        Load and chunk multiple papers with polite delay between requests.
        delay: seconds to wait between papers (respect arXiv rate limits)
        """
        all_chunks = []
        for i, paper_id in enumerate(paper_ids):
            try:
                chunks = self.load_and_chunk(paper_id)
                all_chunks.extend(chunks)
                logger.info("✓ Loaded: arXiv:%s (%d chunks)",
                            self._extract_arxiv_id(paper_id), len(chunks))
            except Exception as e:
                logger.error("✗ Failed: arXiv:%s — %s", paper_id, str(e))

            # Polite delay between papers
            if i < len(paper_ids) - 1:
                logger.info("Waiting %.0fs before next paper (arXiv rate limit)...", delay)
                time.sleep(delay)

        logger.info("Total chunks from %d papers: %d", len(paper_ids), len(all_chunks))
        return all_chunks

    def get_metadata_only(self, paper_id: str) -> Dict:
        """
        Fetch only paper metadata without downloading the PDF.
        Useful for displaying paper info in the UI.
        """
        arxiv_id = self._extract_arxiv_id(paper_id)
        return self._fetch_metadata(arxiv_id)

    # ── Metadata Fetching ─────────────────────────────────────

    def _fetch_metadata(self, arxiv_id: str) -> Dict:
        """Fetch paper metadata from arXiv API."""
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            results = list(self.arxiv_client.results(search))

            if not results:
                raise ValueError(f"No paper found for arXiv ID: {arxiv_id}")

            paper = results[0]

            # Format authors
            authors      = [str(a) for a in paper.authors]
            authors_short = self._format_authors(authors)

            return {
                "arxiv_id"     : arxiv_id,
                "title"        : paper.title.strip(),
                "authors"      : authors,
                "authors_short": authors_short,
                "year"         : str(paper.published.year),
                "abstract"     : paper.summary.strip(),
                "categories"   : [str(c) for c in paper.categories],
                "pdf_url"      : paper.pdf_url,
                "arxiv_url"    : f"https://arxiv.org/abs/{arxiv_id}",
                "source_type"  : "paper",
                "source_tab"   : "paper",
            }

        except Exception as e:
            logger.error("Failed to fetch metadata for arXiv:%s — %s", arxiv_id, e)
            raise

    # ── PDF Download and Parsing ──────────────────────────────

    def _download_and_parse(self, arxiv_id: str, metadata: Dict) -> List[Document]:
        """Download PDF from arXiv and extract text section by section."""

        pdf_url = metadata.get("pdf_url", f"https://arxiv.org/pdf/{arxiv_id}")

        # Download PDF to a temp file
        logger.info("Downloading PDF from: %s", pdf_url)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            response = requests.get(
                pdf_url,
                headers = {"User-Agent": "InsightHub-Research-Assistant/1.0"},
                timeout = 60,
                stream  = True,
            )
            response.raise_for_status()

            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info("PDF downloaded: %.1f MB", os.path.getsize(tmp_path) / (1024*1024))

            # Parse the downloaded PDF
            docs = self._parse_pdf(tmp_path, metadata)

            # Always include abstract as first document
            abstract_doc = self._make_abstract_doc(metadata)
            return [abstract_doc] + docs

        finally:
            # Always clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _parse_pdf(self, pdf_path: str, metadata: Dict) -> List[Document]:
        """Extract text from PDF using PyMuPDF with section detection."""
        pdf_doc  = fitz.open(pdf_path)
        docs     = []
        current_section = "Introduction"
        current_text    = []

        for page_num in range(len(pdf_doc)):
            page  = pdf_doc[page_num]
            blocks = page.get_text("blocks")  # get text blocks with positions

            for block in blocks:
                if block[6] != 0:  # skip non-text blocks (images)
                    continue

                text = block[4].strip()
                if not text or len(text) < 10:
                    continue

                # Detect section headers
                detected_section = self._detect_section(text)
                if detected_section:
                    # Save accumulated text under previous section
                    if current_text:
                        combined = " ".join(current_text).strip()
                        if len(combined) > 100:
                            docs.append(self._make_section_doc(
                                combined, current_section, page_num + 1, metadata
                            ))
                        current_text = []
                    current_section = detected_section
                else:
                    current_text.append(text)

                # Flush every 10 blocks to avoid very large sections
                if len(current_text) >= 10:
                    combined = " ".join(current_text).strip()
                    if len(combined) > 100:
                        docs.append(self._make_section_doc(
                            combined, current_section, page_num + 1, metadata
                        ))
                    current_text = []

        # Flush remaining text
        if current_text:
            combined = " ".join(current_text).strip()
            if len(combined) > 100:
                docs.append(self._make_section_doc(
                    combined, current_section, len(pdf_doc), metadata
                ))

        pdf_doc.close()

        # If no sections detected (scanned PDF), return full text
        if not docs:
            logger.warning("No sections detected — extracting full text")
            docs = self._parse_pdf_fulltext(pdf_path, metadata)

        return docs

    def _parse_pdf_fulltext(self, pdf_path: str, metadata: Dict) -> List[Document]:
        """Fallback: extract full text page by page."""
        pdf_doc = fitz.open(pdf_path)
        docs    = []

        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            text = page.get_text("text").strip()

            if len(text) > 100:
                docs.append(Document(
                    page_content = text,
                    metadata     = {
                        **metadata,
                        "section_name" : "Full Text",
                        "page_number"  : page_num + 1,
                        "total_pages"  : len(pdf_doc),
                        "parser"       : "pymupdf-fulltext",
                    }
                ))

        pdf_doc.close()
        return docs

    # ── Helper Methods ────────────────────────────────────────

    def _make_abstract_doc(self, metadata: Dict) -> Document:
        """Create a Document from the paper abstract."""
        abstract_text = (
            f"Title: {metadata['title']}\n"
            f"Authors: {', '.join(metadata['authors'])}\n"
            f"Year: {metadata['year']}\n"
            f"Categories: {', '.join(metadata['categories'])}\n\n"
            f"Abstract:\n{metadata['abstract']}"
        )
        return Document(
            page_content = abstract_text,
            metadata     = {
                **metadata,
                "section_name" : "Abstract",
                "page_number"  : 1,
                "parser"       : "arxiv-api",
            }
        )

    def _make_section_doc(self, text: str, section: str,
                          page_num: int, metadata: Dict) -> Document:
        """Create a Document for a paper section."""
        return Document(
            page_content = text,
            metadata     = {
                **metadata,
                "section_name" : section,
                "page_number"  : page_num,
                "parser"       : "pymupdf-sections",
            }
        )

    def _detect_section(self, text: str) -> Optional[str]:
        """Detect if a text block is a section header."""
        # Clean text for comparison
        clean = text.strip().lower()
        clean = re.sub(r"^\d+\.?\s*", "", clean)  # remove leading numbers

        # Check against known section headers
        for header in self.SECTION_HEADERS:
            if clean == header or clean.startswith(header + " "):
                return text.strip().title()

        # Detect numbered sections like "1. Introduction" or "2.1 Related Work"
        if re.match(r"^\d+\.?\d*\s+[A-Z][a-z]+", text.strip()):
            if len(text.strip()) < 60:  # headers are usually short
                return text.strip()

        return None

    def _extract_arxiv_id(self, paper_id: str) -> str:
        """
        Extract clean arXiv ID from various input formats.

        Handles:
            "2310.11511"
            "https://arxiv.org/abs/2310.11511"
            "https://arxiv.org/pdf/2310.11511"
            "arxiv:2310.11511"
            "arXiv:2310.11511v2"
        """
        # Remove whitespace
        paper_id = paper_id.strip()

        # Extract from URL
        url_match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d+\.\d+)", paper_id, re.IGNORECASE)
        if url_match:
            return url_match.group(1)

        # Remove "arxiv:" prefix
        paper_id = re.sub(r"^arxiv:", "", paper_id, flags=re.IGNORECASE)

        # Remove version suffix (e.g. v2, v3)
        paper_id = re.sub(r"v\d+$", "", paper_id)

        # Validate format (should be YYMM.NNNNN)
        if re.match(r"^\d{4}\.\d{4,5}$", paper_id):
            return paper_id

        raise ValueError(
            f"Invalid arXiv ID format: '{paper_id}'\n"
            f"Expected formats:\n"
            f"  '2310.11511'\n"
            f"  'https://arxiv.org/abs/2310.11511'\n"
            f"  'arxiv:2310.11511'"
        )

    def _format_authors(self, authors: List[str]) -> str:
        """Format author list for display."""
        if len(authors) == 0:
            return "Unknown"
        elif len(authors) == 1:
            return authors[0]
        elif len(authors) == 2:
            return f"{authors[0]} and {authors[1]}"
        else:
            return f"{authors[0]} et al."


# ─────────────────────────────────────────────────────────────
#  QUICK TEST
#  Command: python insighthub/ingestion/paper_loader.py
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("\n" + "="*55)
    print("  InsightHub — Paper Loader Test")
    print("="*55)

    loader = PaperLoader()

    # Default test paper — Self-RAG (directly relevant to InsightHub)
    test_id = "2310.11511"

    # Use custom paper ID if provided
    if len(sys.argv) > 1:
        test_id = sys.argv[1]

    print(f"\n[Test 1] Fetching metadata for arXiv:{test_id}...")
    try:
        meta = loader.get_metadata_only(test_id)
        print(f"  ✓ Title   : {meta['title']}")
        print(f"  ✓ Authors : {meta['authors_short']}")
        print(f"  ✓ Year    : {meta['year']}")
        print(f"  ✓ Category: {', '.join(meta['categories'])}")
        print(f"  ✓ Abstract: '{meta['abstract'][:120]}...'")
    except Exception as e:
        print(f"  ✗ Metadata fetch failed: {e}")

    print(f"\n[Test 2] Loading and chunking arXiv:{test_id}...")
    print("  (This downloads the PDF — may take 30-60 seconds)")
    try:
        chunks = loader.load_and_chunk(test_id)
        print(f"  ✓ Total chunks    : {len(chunks)}")
        print(f"  ✓ First section   : {chunks[0].metadata.get('section_name', 'N/A')}")
        print(f"  ✓ Parser used     : {chunks[0].metadata.get('parser', 'N/A')}")
        print(f"  ✓ Source tab      : {chunks[0].metadata.get('source_tab', 'N/A')}")
        print(f"  ✓ First chunk     : '{chunks[0].page_content[:150]}...'")

        # Show all unique sections found
        sections = list(dict.fromkeys(
            c.metadata.get("section_name", "Unknown") for c in chunks
        ))
        print(f"  ✓ Sections found  : {sections}")

    except Exception as e:
        print(f"  ✗ Loading failed: {e}")

    print("\n" + "="*55)
    print("  Paper Loader Test Complete!")
    print("  To test with a different paper:")
    print("  python insighthub/ingestion/paper_loader.py 1706.03762")
    print("="*55 + "\n")