"""
Web Search Module
─────────────────
Uses Serper API via LangChain GoogleSerperAPIWrapper.
Returns a list of structured dicts: {title, snippet, link}
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()


def search_web(query: str, num_results: int = 5) -> list:
    """
    Search the web via Serper API.

    Returns:
        list of dicts with keys: title, snippet, link
    """
    api_key = os.getenv("SERPER_API_KEY", "")
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    payload = {"q": query, "num": num_results}

    try:
        response = requests.post(
            "https://google.serper.dev/search",
            headers=headers,
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("organic", []):
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "link": item.get("link", ""),
            })
        return results if results else [{"title": query, "snippet": "No results found.", "link": ""}]

    except Exception as e:
        return [{"title": "Search Error", "snippet": str(e), "link": ""}]