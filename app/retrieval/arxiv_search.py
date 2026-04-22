"""
arXiv Search Module
────────────────────
Searches arXiv for relevant research papers.
Returns a list of structured dicts: {title, summary, url}
"""

import arxiv
import time
import logging

logger = logging.getLogger(__name__)

def search_arxiv(query: str, max_results: int = 5) -> list:
    """
    Search arXiv and return structured paper results.
    Includes a retry mechanism for rate-limiting (HTTP 429).
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            results = []
            # results() returns a generator; we consume it to trigger the API call
            for paper in client.results(search):
                results.append({
                    "title": paper.title,
                    "summary": paper.summary,
                    "url": paper.entry_id,
                    "authors": ", ".join(str(a) for a in paper.authors[:3]),
                })
            return results
        except Exception as e:
            # Check for 429 in the exception message
            if "429" in str(e) and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                logger.warning(f"arXiv 429 Rate Limit hit. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            logger.error(f"arXiv search failed: {e}")
            raise e

    return []