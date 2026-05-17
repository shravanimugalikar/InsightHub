"""
arXiv Search Module
────────────────────
Searches arXiv for relevant research papers with full author metadata.

Strategy (two-stage, each stage falls back gracefully):
  1. OpenAlex API  — free, no key, returns title + authors + year + arXiv URL
  2. Serper API    — fallback if OpenAlex is unavailable (no authors)

Returns a list of LangChain Document objects formatted as:
  [N] Last, F., Last2, F. et al. (Year) — Title. arXiv: URL
"""

import os
import re
import logging
import requests
from dotenv import load_dotenv

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

load_dotenv()
logger = logging.getLogger(__name__)

# OpenAlex source ID for arXiv
_OPENALEX_ARXIV_SOURCE = "s4306400194"
_OPENALEX_HEADERS = {"User-Agent": "InsightHub/1.0 (Research Assistant; mailto:research@insighthub.app)"}


def _extract_arxiv_id(url: str) -> str:
    """Extract the bare arXiv paper ID (e.g. '2304.10557') from any arXiv URL."""
    m = re.search(r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})', url)
    return m.group(1) if m else ""


def _format_authors(authorships: list) -> str:
    """
    Convert OpenAlex authorships list to 'Last, F., Last2, F. et al.' format.
    """
    if not authorships:
        return ""
    authors = []
    for a in authorships[:3]:
        if not a:
            continue
        display = a.get("author", {}).get("display_name", "").strip() if a.get("author") else ""
        if not display:
            continue
        parts = display.split()
        if len(parts) >= 2:
            last  = parts[-1]
            first = parts[0][0] + "."
            authors.append(f"{last}, {first}")
        else:
            authors.append(display)
    author_str = ", ".join(authors)
    if len(authorships) > 3:
        author_str += " et al."
    return author_str


def _search_openalex(query: str, max_results: int, year_from, year_to, sort_by: str) -> list:
    """
    Search arXiv papers via OpenAlex API.
    Returns list of Document objects with full author metadata.
    """
    try:
        # Build sort parameter
        sort_param = "cited_by_count:desc"
        if sort_by in ("Latest First", "latest"):
            sort_param = "publication_date:desc"
        elif sort_by in ("Oldest First",):
            sort_param = "publication_date:asc"

        # Build year filter
        year_filter = ""
        if year_from and year_to:
            year_filter = f",publication_year:{year_from}-{year_to}"
        elif year_from:
            year_filter = f",publication_year:>{year_from - 1}"
        elif year_to:
            year_filter = f",publication_year:<{year_to + 1}"

        params = {
            "filter":   f"locations.source.id:{_OPENALEX_ARXIV_SOURCE}{year_filter}",
            "search":   query,
            "per-page": min(max_results, 15),
            "sort":     sort_param,
            "select":   "title,authorships,publication_year,locations",
        }

        r = requests.get(
            "https://api.openalex.org/works",
            params=params,
            headers=_OPENALEX_HEADERS,
            timeout=12,
        )

        if r.status_code != 200:
            logger.warning(f"OpenAlex returned {r.status_code}")
            return []

        results = r.json().get("results") or []
        docs = []

        for work in results:
            if not work:
                continue
            title = (work.get("title") or "").strip()
            if not title:
                continue

            year = work.get("publication_year") or ""

            # Year filter (fallback)
            if year_from and year and int(year) < year_from:
                continue
            if year_to and year and int(year) > year_to:
                continue

            # Find the arXiv landing URL
            arxiv_url = ""
            for loc in (work.get("locations") or []):
                if not loc:
                    continue
                landing = loc.get("landing_page_url") or ""
                if "arxiv.org/abs/" in landing or "arxiv.org/pdf/" in landing:
                    arxiv_url = landing
                    break
            if not arxiv_url:
                continue   # skip non-arXiv locations

            # Normalise to canonical https URL
            arxiv_id = _extract_arxiv_id(arxiv_url)
            canonical_url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else arxiv_url

            # Format authors
            authors = _format_authors(work.get("authorships"))

            docs.append(Document(
                page_content=title,   # abstract not returned; use title as content
                metadata={
                    "title":   title,
                    "authors": authors,
                    "year":    year,
                    "url":     canonical_url,
                    "source":  canonical_url,
                    "type":    "arxiv",
                }
            ))

            if len(docs) >= max_results:
                break

        logger.info(f"OpenAlex returned {len(docs)} docs for: {query!r}")
        return docs

    except Exception as e:
        logger.warning(f"OpenAlex search failed ({e}), will try Serper fallback")
        return []


def _search_serper_fallback(query: str, max_results: int, year_from, year_to, sort_by: str) -> list:
    """
    Fallback: search arXiv via Serper API (site:arxiv.org/abs).
    Authors will be empty — synthesizer handles gracefully.
    """
    try:
        api_key = os.getenv("SERPER_API_KEY", "")
        if not api_key:
            return []

        year_clause = ""
        if year_from and year_to:
            year_clause = f" after:{year_from} before:{year_to}"
        elif year_from:
            year_clause = f" after:{year_from}"
        elif year_to:
            year_clause = f" before:{year_to}"

        payload = {
            "q":   f"site:arxiv.org/abs {query}{year_clause}",
            "num": min(max_results, 10),
        }
        if sort_by in ("Latest First", "latest"):
            payload["tbs"] = "qdr:y"

        response = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=15,
        )
        response.raise_for_status()

        docs = []
        seen_ids = set()

        for item in response.json().get("organic", []):
            url     = item.get("link", "")
            title   = item.get("title", "").strip()
            snippet = item.get("snippet", "").strip()

            if "arxiv.org/abs/" not in url and "arxiv.org/pdf/" not in url:
                continue

            arxiv_id = _extract_arxiv_id(url)
            if not arxiv_id or arxiv_id in seen_ids:
                continue
            seen_ids.add(arxiv_id)

            # Clean title
            clean_title = re.sub(r'^\[?\d{4}\.\d{4,5}\]?\s*', '', title).strip()
            clean_title = re.sub(r'\s*[-–|]\s*arXiv.*$', '', clean_title, flags=re.IGNORECASE).strip()
            if not clean_title:
                clean_title = title

            # Guess year from ID
            guessed_year = ""
            try:
                yy = int(arxiv_id[:2])
                guessed_year = 2000 + yy if yy <= 30 else 1900 + yy
            except Exception:
                pass

            if year_from and guessed_year and int(guessed_year) < year_from:
                continue
            if year_to and guessed_year and int(guessed_year) > year_to:
                continue

            docs.append(Document(
                page_content=snippet or clean_title,
                metadata={
                    "title":   clean_title,
                    "authors": "",
                    "year":    guessed_year,
                    "url":     f"https://arxiv.org/abs/{arxiv_id}",
                    "source":  f"https://arxiv.org/abs/{arxiv_id}",
                    "type":    "arxiv",
                }
            ))

            if len(docs) >= max_results:
                break

        logger.info(f"Serper fallback returned {len(docs)} docs for: {query!r}")
        return docs

    except Exception as e:
        logger.error(f"Serper fallback also failed: {e}")
        return []


def search_arxiv(
    query:       str,
    max_results: int = 7,
    year_from:   int = None,
    year_to:     int = None,
    sort_by:     str = "Relevance",
) -> list:
    """
    Search arXiv and return list of LangChain Documents with full author metadata.
    Returns empty list on failure — never raises.

    Citation format produced:
      [N] Last, F., Last2, F. et al. (Year) — Title. arXiv: URL
    """
    # Normalise sort_by strings from workflow state
    if sort_by in ("relevance", None, ""):
        sort_by = "Relevance"
    elif sort_by == "latest":
        sort_by = "Latest First"

    # Stage 1 — OpenAlex (has authors, free, no key)
    docs = _search_openalex(query, max_results, year_from, year_to, sort_by)

    # Stage 2 — Serper fallback (no authors, but reliable)
    if not docs:
        docs = _search_serper_fallback(query, max_results, year_from, year_to, sort_by)

    return docs