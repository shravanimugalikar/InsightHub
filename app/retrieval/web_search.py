"""
Web Search Module
─────────────────
Uses Serper API via LangChain GoogleSerperAPIWrapper.
Returns a list of LangChain Document objects.
"""

import os
import requests
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()


def search_web(query: str, num_results: int = 10) -> list:
    """
    Search the web via Serper API.

    Returns:
        list of Document objects
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
        results = response.json()

        docs = []
        for result in results.get("organic", []):
            docs.append(Document(
                page_content=result.get("snippet", ""),
                metadata={
                    "source":  result.get("link", ""),    # full URL
                    "title":   result.get("title", ""),
                    "url":     result.get("link", ""),
                    "domain":  result.get("displayLink", ""),
                    "year":    "",                        # web results rarely have year
                    "type":    "web",
                }
            ))
        return docs

    except Exception as e:
        return []