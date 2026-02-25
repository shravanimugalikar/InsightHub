# ============================================================
#  InsightHub — insighthub/ingestion/web_loader.py
#  Handles: Website URL scraping
#  Parser:  BeautifulSoup4 (primary), Playwright (fallback)
# ============================================================

import sys
import re
import time
import logging
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from insighthub.config.settings import (
    CHUNK_SIZE_WEB,
    CHUNK_OVERLAP,
    REQUEST_TIMEOUT,
    MAX_PAGE_CHARS,
    USE_PLAYWRIGHT,
)

# ── Logging setup ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  MAIN LOADER CLASS
# ─────────────────────────────────────────────────────────────

class WebLoader:
    """
    Loads and chunks web pages into LangChain Documents.

    Uses BeautifulSoup4 as the primary scraper for static pages.
    Falls back to Playwright for JavaScript-heavy pages that
    BeautifulSoup cannot render (e.g. React/Vue sites).

    Usage:
        loader = WebLoader()
        chunks = loader.load_and_chunk("https://example.com/article")
    """

    # Browser-like headers to avoid being blocked
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    # HTML tags that contain main content
    CONTENT_TAGS = [
        "article", "main", "section", "div",
        "p", "h1", "h2", "h3", "h4", "h5", "h6",
    ]

    # Tags to completely remove before extracting text
    REMOVE_TAGS = [
        "script", "style", "nav", "footer", "header",
        "aside", "advertisement", "cookie", "popup",
        "iframe", "noscript", "form", "button",
    ]

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size      = CHUNK_SIZE_WEB,
            chunk_overlap   = CHUNK_OVERLAP,
            separators      = ["\n\n", "\n", ". ", " ", ""],
            length_function = len,
        )
        logger.info("WebLoader initialised (chunk_size=%d, overlap=%d)",
                    CHUNK_SIZE_WEB, CHUNK_OVERLAP)

    # ── Public methods ────────────────────────────────────────

    def load(self, url: str) -> List[Document]:
        """
        Scrape a URL and return LangChain Documents.
        Uses BeautifulSoup4 first, Playwright as fallback.
        """
        self._validate_url(url)
        logger.info("Loading URL: %s", url)

        # Try BeautifulSoup first (fast, no browser needed)
        try:
            docs = self._load_with_beautifulsoup(url)
            if docs:
                return docs
            logger.warning("BeautifulSoup returned empty content — trying Playwright")
        except Exception as e:
            logger.warning("BeautifulSoup failed: %s — trying Playwright", str(e))

        # Fallback to Playwright for JS-heavy pages
        if USE_PLAYWRIGHT:
            try:
                return self._load_with_playwright(url)
            except Exception as e:
                logger.error("Playwright also failed: %s", str(e))
                raise RuntimeError(
                    f"Could not scrape {url}.\n"
                    f"Both BeautifulSoup and Playwright failed.\n"
                    f"The site may be blocking scrapers or require login."
                )
        else:
            raise RuntimeError(f"Could not scrape {url} — BeautifulSoup returned no content.")

    def load_and_chunk(self, url: str) -> List[Document]:
        """
        Scrape a URL and split into chunks ready for embedding.
        This is the main method used by the ingestion pipeline.
        """
        docs   = self.load(url)
        chunks = self.splitter.split_documents(docs)

        # Add chunk index to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"]     = i
            chunk.metadata["total_chunks"] = len(chunks)
            chunk.metadata["source_tab"]   = "website"

        logger.info("Split into %d chunks from %s", len(chunks), url)
        return chunks

    def load_multiple(self, urls: List[str], delay: float = 1.0) -> List[Document]:
        """
        Scrape multiple URLs with a polite delay between requests.
        delay: seconds to wait between requests (be respectful to servers)
        """
        all_chunks = []
        for i, url in enumerate(urls):
            try:
                chunks = self.load_and_chunk(url)
                all_chunks.extend(chunks)
                logger.info("✓ Scraped: %s (%d chunks)", url, len(chunks))
            except Exception as e:
                logger.error("✗ Failed to scrape %s: %s", url, str(e))

            # Polite delay between requests (except after last URL)
            if i < len(urls) - 1:
                time.sleep(delay)

        logger.info("Total chunks from %d URLs: %d", len(urls), len(all_chunks))
        return all_chunks

    # ── BeautifulSoup Scraper ─────────────────────────────────

    def _load_with_beautifulsoup(self, url: str) -> List[Document]:
        """Primary scraper — fast, no browser needed."""

        response = requests.get(
            url,
            headers = self.HEADERS,
            timeout = REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "lxml")

        # Extract page metadata
        title       = self._extract_title(soup)
        description = self._extract_description(soup)
        domain      = urlparse(url).netloc

        # Remove noise tags
        for tag in soup(self.REMOVE_TAGS):
            tag.decompose()

        # Try to find main content area first
        content_text = self._extract_main_content(soup)

        if not content_text or len(content_text) < 200:
            # Fall back to full body text
            content_text = soup.get_text(separator="\n", strip=True)

        # Clean up the text
        content_text = self._clean_text(content_text)

        if not content_text or len(content_text) < 100:
            return []

        # Truncate if too long
        if len(content_text) > MAX_PAGE_CHARS:
            content_text = content_text[:MAX_PAGE_CHARS]
            logger.info("Page truncated to %d chars", MAX_PAGE_CHARS)

        doc = Document(
            page_content = content_text,
            metadata     = {
                "source"       : url,
                "url"          : url,
                "domain"       : domain,
                "page_title"   : title,
                "description"  : description,
                "source_type"  : "website",
                "source_tab"   : "website",
                "js_rendered"  : False,
                "parser"       : "beautifulsoup4",
                "scraped_date" : self._get_date(),
            }
        )

        return [doc]

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Try to find the main content area of the page."""

        # Priority order for main content containers
        selectors = [
            {"name": "main"},
            {"name": "article"},
            {"id": "main-content"},
            {"id": "content"},
            {"class": "content"},
            {"class": "post-content"},
            {"class": "article-body"},
            {"class": "entry-content"},
            {"role": "main"},
        ]

        for selector in selectors:
            element = soup.find(**selector)
            if element:
                text = element.get_text(separator="\n", strip=True)
                if len(text) > 200:
                    return text

        # If no main content found, extract all paragraphs
        paragraphs = soup.find_all("p")
        if paragraphs:
            text = "\n\n".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50)
            if len(text) > 200:
                return text

        return ""

    # ── Playwright Scraper ────────────────────────────────────

    def _load_with_playwright(self, url: str) -> List[Document]:
        """
        Fallback scraper using Playwright browser.
        Handles JavaScript-rendered pages (React, Vue, Angular sites).
        Requires: playwright install chromium
        """
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise ImportError(
                "Playwright not installed. Run: pip install playwright\n"
                "Then run: playwright install chromium"
            )

        logger.info("Using Playwright for JS-rendered page: %s", url)
        domain = urlparse(url).netloc

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page    = browser.new_page()

            # Set browser headers
            page.set_extra_http_headers(self.HEADERS)

            # Navigate and wait for content to load
            page.goto(url, wait_until="networkidle", timeout=30000)

            # Get page title
            title = page.title()

            # Get full rendered HTML
            html = page.content()
            browser.close()

        # Parse rendered HTML with BeautifulSoup
        soup = BeautifulSoup(html, "lxml")

        for tag in soup(self.REMOVE_TAGS):
            tag.decompose()

        content_text = self._extract_main_content(soup)

        if not content_text or len(content_text) < 200:
            content_text = soup.get_text(separator="\n", strip=True)

        content_text = self._clean_text(content_text)

        if len(content_text) > MAX_PAGE_CHARS:
            content_text = content_text[:MAX_PAGE_CHARS]

        doc = Document(
            page_content = content_text,
            metadata     = {
                "source"       : url,
                "url"          : url,
                "domain"       : domain,
                "page_title"   : title,
                "description"  : "",
                "source_type"  : "website",
                "source_tab"   : "website",
                "js_rendered"  : True,
                "parser"       : "playwright",
                "scraped_date" : self._get_date(),
            }
        )

        return [doc]

    # ── Helper Methods ────────────────────────────────────────

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title from HTML."""
        # Try Open Graph title first (usually cleaner)
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            return og_title["content"].strip()

        # Fall back to <title> tag
        title_tag = soup.find("title")
        if title_tag:
            return title_tag.get_text(strip=True)

        return "Unknown Title"

    def _extract_description(self, soup: BeautifulSoup) -> str:
        """Extract page meta description."""
        og_desc = soup.find("meta", property="og:description")
        if og_desc and og_desc.get("content"):
            return og_desc["content"].strip()

        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            return meta_desc["content"].strip()

        return ""

    def _clean_text(self, text: str) -> str:
        """Clean extracted text — remove excessive whitespace and noise."""
        # Replace multiple newlines with double newline
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Replace multiple spaces with single space
        text = re.sub(r" {2,}", " ", text)
        # Remove lines that are just whitespace
        lines = [line.strip() for line in text.split("\n")]
        lines = [line for line in lines if line]
        # Remove very short lines (likely navigation/button text)
        lines = [line for line in lines if len(line) > 15]
        return "\n".join(lines)

    def _validate_url(self, url: str):
        """Check URL is valid and accessible."""
        if not url.startswith(("http://", "https://")):
            raise ValueError(
                f"Invalid URL: '{url}'\n"
                f"URL must start with http:// or https://"
            )
        parsed = urlparse(url)
        if not parsed.netloc:
            raise ValueError(f"Invalid URL format: {url}")

    def _get_date(self) -> str:
        """Return current date as string."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d")


# ─────────────────────────────────────────────────────────────
#  QUICK TEST — run this file directly to test your loader
#  Command: python insighthub/ingestion/web_loader.py
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("\n" + "="*55)
    print("  InsightHub — Web Loader Test")
    print("="*55)

    loader = WebLoader()

    # Test URLs — these are public pages that allow scraping
    test_urls = [
        "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
    ]

    # Use custom URL if provided
    if len(sys.argv) > 1:
        test_urls = [sys.argv[1]]

    for url in test_urls:
        print(f"\n[Test] Scraping: {url}")
        try:
            chunks = loader.load_and_chunk(url)
            print(f"  ✓ Scraped {len(chunks)} chunks")
            print(f"  ✓ Page title  : {chunks[0].metadata.get('page_title', 'N/A')}")
            print(f"  ✓ Domain      : {chunks[0].metadata.get('domain', 'N/A')}")
            print(f"  ✓ Parser used : {chunks[0].metadata.get('parser', 'N/A')}")
            print(f"  ✓ JS rendered : {chunks[0].metadata.get('js_rendered', False)}")
            print(f"  ✓ First chunk : '{chunks[0].page_content[:120]}...'")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\n" + "="*55)
    print("  Web Loader Test Complete!")
    print("  To test with your own URL:")
    print("  python insighthub/ingestion/web_loader.py https://your-url.com")
    print("="*55 + "\n")