"""
arXiv Search Module
────────────────────
Searches arXiv for relevant research papers.
Returns a list of structured dicts: {title, summary, url}
"""

import arxiv


def search_arxiv(query: str, max_results: int = 5) -> list:
    """
    Search arXiv and return structured paper results.

    Returns:
        list of dicts with keys: title, summary, url, authors
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    results = []
    for paper in client.results(search):
        results.append({
            "title": paper.title,
            "summary": paper.summary,
            "url": paper.entry_id,
            "authors": ", ".join(str(a) for a in paper.authors[:3]),
        })

    return results