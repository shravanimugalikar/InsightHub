"""
arXiv Search Module
────────────────────
Searches arXiv for relevant research papers.
Returns a list of structured dicts: {title, summary, url, year, authors}

Supports:
  - year_from / year_until  → filters results to a date range
  - sort_by                 → "relevance" (default) or "latest"
"""

import arxiv
import time
import logging
import datetime

logger = logging.getLogger(__name__)


def search_arxiv(
    query: str,
    max_results: int = 5,
    year_from: int = None,
    year_until: int = None,
    sort_by: str = "relevance",   # "relevance" | "latest"
) -> list:
    """
    Search arXiv and return structured paper results.

    Args:
        query       – Search string.
        max_results – Max papers to return.
        year_from   – Filter: only include papers from this year onwards.
        year_until  – Filter: only include papers up to this year.
        sort_by     – "relevance" sorts by arXiv relevance score;
                      "latest" sorts by submission date (newest first).
    """
    dt_from  = datetime.datetime(year_from,  1, 1,  tzinfo=datetime.timezone.utc) if year_from  else None
    dt_until = datetime.datetime(year_until, 12, 31, 23, 59, 59, tzinfo=datetime.timezone.utc) if year_until else None

    sort_criterion = (
        arxiv.SortCriterion.SubmittedDate
        if sort_by == "latest"
        else arxiv.SortCriterion.Relevance
    )

    # Fetch extra to compensate for date-filtered-out results
    fetch_n = max_results * 5 if (dt_from or dt_until) else max_results

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=fetch_n,
        sort_by=sort_criterion,
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            results = []
            for paper in client.results(search):
                pub = paper.published
                if pub and pub.tzinfo is None:
                    pub = pub.replace(tzinfo=datetime.timezone.utc)

                # Apply year-range filter
                if dt_from  and pub and pub < dt_from:
                    continue
                if dt_until and pub and pub > dt_until:
                    continue

                results.append({
                    "title":   paper.title,
                    "summary": paper.summary,
                    "url":     paper.entry_id,
                    "authors": ", ".join(str(a) for a in paper.authors[:3]),
                    "year":    pub.year if pub else None,
                })

                if len(results) >= max_results:
                    break

            return results

        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                logger.warning(f"arXiv 429 Rate Limit. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            logger.error(f"arXiv search failed: {e}")
            raise e

    return []