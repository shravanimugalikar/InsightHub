# ============================================================
#  InsightHub — insighthub/agents/planner_agent.py
#  Decomposes user queries into per-source sub-queries
#  Model: Mistral-7B via HuggingFace Inference API (FREE)
# ============================================================

import sys
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from openai import OpenAI

from insighthub.config.settings import (
    HUGGINGFACE_API_TOKEN,
    LLM_MODEL,
    LLM_MAX_NEW_TOKENS,
    LLM_TEMPERATURE,
    SOURCE_TYPES,
)

# ── Logging setup ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class SubQuery:
    """A single sub-query targeting a specific source tab."""
    source_tab  : str           # "document", "website", or "paper"
    query       : str           # the sub-query text
    priority    : int = 1       # 1=high, 2=medium, 3=low
    keywords    : List[str] = field(default_factory=list)  # key terms


@dataclass
class QueryPlan:
    """
    The full plan produced by the Planner Agent.
    Contains the decomposed sub-queries and query intent.
    """
    original_query : str
    intent         : str                    # factual / comparative / exploratory
    sub_queries    : List[SubQuery]
    active_sources : List[str]              # which source tabs to query
    reasoning      : str = ""              # why this plan was chosen


# ─────────────────────────────────────────────────────────────
#  PLANNER AGENT CLASS
# ─────────────────────────────────────────────────────────────

class PlannerAgent:
    """
    Decomposes a user research query into targeted sub-queries
    for each active source tab (Documents, Websites, Papers).

    For simple queries it generates one sub-query per source.
    For complex comparative queries it generates multiple
    sub-queries per source to ensure comprehensive coverage.

    Usage:
        planner = PlannerAgent()
        plan    = planner.plan("Compare RAG vs Self-RAG for research assistance")

        for sub_query in plan.sub_queries:
            print(sub_query.source_tab, sub_query.query)
    """

    # ── Prompt Templates ──────────────────────────────────────

    SYSTEM_PROMPT = """You are a research query planner for InsightHub, a multi-agent RAG system.
Your job is to analyse a user's research question and decompose it into specific sub-queries
for three source types: Documents, Websites, and Research Papers.

Rules:
1. Generate 1-2 sub-queries per source type depending on query complexity
2. Each sub-query must be tailored to that source type's strengths
3. Documents: good for detailed technical content, reports, notes
4. Websites: good for current events, tutorials, recent news, practical guides
5. Papers: good for academic methods, experiments, citations, theoretical foundations
6. Identify the query intent: factual (specific answer), comparative (A vs B), or exploratory (overview)
7. Always respond with valid JSON only — no extra text before or after

Output format (JSON only):
{
  "intent": "factual|comparative|exploratory",
  "reasoning": "brief explanation of your decomposition strategy",
  "sub_queries": [
    {
      "source_tab": "document",
      "query": "specific sub-query for documents",
      "priority": 1,
      "keywords": ["keyword1", "keyword2"]
    },
    {
      "source_tab": "website",
      "query": "specific sub-query for websites",
      "priority": 1,
      "keywords": ["keyword1", "keyword2"]
    },
    {
      "source_tab": "paper",
      "query": "specific sub-query for research papers",
      "priority": 1,
      "keywords": ["keyword1", "keyword2"]
    }
  ]
}"""

    USER_PROMPT_TEMPLATE = """Research question: "{query}"

Active source tabs: {sources}

Decompose this into targeted sub-queries for each source type.
Return JSON only."""

    def __init__(self, active_sources: Optional[List[str]] = None):
        """
        Args:
            active_sources: List of source tabs to query.
                           Defaults to all 3: ["document", "website", "paper"]
        """
        self.active_sources = active_sources or SOURCE_TYPES

        # Set up HuggingFace Inference client
        if not HUGGINGFACE_API_TOKEN:
            raise ValueError(
                "HUGGINGFACE_API_TOKEN not set!\n"
                "Add it to your .env file: HUGGINGFACE_API_TOKEN=hf_xxxx"
            )

        # Llama 3.3 70B via Cerebras — confirmed working, no license needed
        self._model = "meta-llama/Llama-3.1-8B-Instruct:cerebras"
        self.client = OpenAI(
            base_url = "https://router.huggingface.co/v1",
            api_key  = HUGGINGFACE_API_TOKEN,
        )
        logger.info("PlannerAgent initialised (model=%s, sources=%s)",
                    self._model, self.active_sources)

    # ── Public Methods ────────────────────────────────────────

    def plan(self, query: str) -> QueryPlan:
        """
        Main method — takes a research query and returns a QueryPlan.

        Args:
            query: The user's research question

        Returns:
            QueryPlan with decomposed sub-queries for each source
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        logger.info("Planning query: '%s'", query[:80])

        # Try LLM-based planning first
        try:
            plan = self._llm_plan(query)
            logger.info("✓ LLM plan: intent=%s, sub_queries=%d",
                        plan.intent, len(plan.sub_queries))
            return plan

        except Exception as e:
            logger.warning("LLM planning failed: %s — using rule-based fallback", str(e))
            return self._rule_based_plan(query)

    def plan_simple(self, query: str) -> QueryPlan:
        """
        Faster planning using rules only — no LLM call.
        Use this when speed is more important than plan quality.
        """
        return self._rule_based_plan(query)

    # ── LLM Planning ─────────────────────────────────────────

    def _llm_plan(self, query: str) -> QueryPlan:
        """Generate a query plan using Mistral-7B via HuggingFace API."""

        user_message = self.USER_PROMPT_TEMPLATE.format(
            query   = query,
            sources = ", ".join(self.active_sources),
        )

        # Build messages for chat completion
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ]

        logger.info("Calling LLM for query planning...")

        response = self.client.chat.completions.create(
            model       = self._model,
            messages    = messages,
            max_tokens  = 800,
            temperature = LLM_TEMPERATURE,
        )

        raw_text = response.choices[0].message.content.strip()
        logger.debug("LLM response: %s", raw_text[:200])

        # Parse JSON response
        plan_data = self._parse_json_response(raw_text)

        # Build QueryPlan from parsed data
        return self._build_plan(query, plan_data)

    # ── Rule-Based Fallback ───────────────────────────────────

    def _rule_based_plan(self, query: str) -> QueryPlan:
        """
        Rule-based query decomposition — no LLM needed.
        Used as fallback when LLM is unavailable or fails.
        Generates sensible sub-queries based on query keywords.
        """
        logger.info("Using rule-based query planner")

        query_lower = query.lower()

        # Detect intent
        intent = self._detect_intent(query_lower)

        # Extract keywords
        keywords = self._extract_keywords(query)

        sub_queries = []

        for source_tab in self.active_sources:
            sub_query = self._generate_sub_query(
                query, source_tab, intent, keywords
            )
            sub_queries.append(sub_query)

        return QueryPlan(
            original_query = query,
            intent         = intent,
            sub_queries    = sub_queries,
            active_sources = self.active_sources,
            reasoning      = f"Rule-based decomposition (intent: {intent})",
        )

    def _generate_sub_query(
        self,
        query      : str,
        source_tab : str,
        intent     : str,
        keywords   : List[str],
    ) -> SubQuery:
        """Generate a source-specific sub-query using templates."""

        templates = {
            "document": {
                "factual"     : f"{query} detailed explanation",
                "comparative" : f"comparison of {' and '.join(keywords[:2])} in documents",
                "exploratory" : f"overview of {query}",
            },
            "website": {
                "factual"     : f"{query} tutorial guide",
                "comparative" : f"{query} practical differences real-world",
                "exploratory" : f"{query} recent developments 2024",
            },
            "paper": {
                "factual"     : f"{query} academic research methodology",
                "comparative" : f"{query} empirical comparison experimental results",
                "exploratory" : f"{query} survey literature review",
            },
        }

        sub_query_text = templates.get(source_tab, {}).get(intent, query)

        return SubQuery(
            source_tab = source_tab,
            query      = sub_query_text,
            priority   = 1,
            keywords   = keywords[:5],
        )

    # ── Helper Methods ────────────────────────────────────────

    def _detect_intent(self, query_lower: str) -> str:
        """Detect query intent from keywords."""
        comparative_words = ["compare", "vs", "versus", "difference", "better",
                             "contrast", "which", "pros and cons"]
        factual_words     = ["what is", "how does", "explain", "define",
                             "what are", "how to", "why does"]

        if any(w in query_lower for w in comparative_words):
            return "comparative"
        elif any(w in query_lower for w in factual_words):
            return "factual"
        else:
            return "exploratory"

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query."""
        # Remove common stop words
        stop_words = {
            "what", "is", "are", "how", "does", "the", "a", "an",
            "in", "of", "for", "to", "and", "or", "with", "that",
            "this", "these", "those", "can", "could", "would", "should",
            "explain", "describe", "tell", "me", "about", "why", "when",
        }
        words    = re.findall(r"\b[a-zA-Z]{3,}\b", query)
        keywords = [w for w in words if w.lower() not in stop_words]
        return list(dict.fromkeys(keywords))[:8]  # unique, max 8

    def _parse_json_response(self, text: str) -> Dict:
        """Parse JSON from LLM response — handles common formatting issues."""
        # Try to extract JSON block if wrapped in markdown
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if json_match:
            text = json_match.group(1)

        # Find first { and last } to extract JSON
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON found in LLM response: {text[:200]}")

        json_str = text[start:end]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON from LLM: {e}\nRaw: {json_str[:300]}")

    def _build_plan(self, original_query: str, plan_data: Dict) -> QueryPlan:
        """Build a QueryPlan object from parsed LLM JSON data."""
        sub_queries = []

        for sq_data in plan_data.get("sub_queries", []):
            source_tab = sq_data.get("source_tab", "document")

            # Only include active sources
            if source_tab not in self.active_sources:
                continue

            sub_queries.append(SubQuery(
                source_tab = source_tab,
                query      = sq_data.get("query", original_query),
                priority   = sq_data.get("priority", 1),
                keywords   = sq_data.get("keywords", []),
            ))

        # Ensure at least one sub-query per active source
        covered = {sq.source_tab for sq in sub_queries}
        for source_tab in self.active_sources:
            if source_tab not in covered:
                logger.warning("LLM missed source '%s' — adding fallback", source_tab)
                sub_queries.append(SubQuery(
                    source_tab = source_tab,
                    query      = original_query,
                    priority   = 2,
                    keywords   = self._extract_keywords(original_query),
                ))

        return QueryPlan(
            original_query = original_query,
            intent         = plan_data.get("intent", "exploratory"),
            sub_queries    = sub_queries,
            active_sources = self.active_sources,
            reasoning      = plan_data.get("reasoning", ""),
        )


# ─────────────────────────────────────────────────────────────
#  QUICK TEST
#  Command: python insighthub/agents/planner_agent.py
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "="*55)
    print("  InsightHub — Planner Agent Test")
    print("="*55)

    planner = PlannerAgent()

    test_queries = [
        "What is retrieval-augmented generation and how does it work?",
        "Compare RAG vs Self-RAG for academic research assistance",
        "What are the latest developments in multi-agent AI systems?",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n[Test {i}] Query: '{query}'")
        print("-" * 50)

        try:
            # Try LLM plan first
            plan = planner.plan(query)
            print(f"  ✓ Intent    : {plan.intent}")
            print(f"  ✓ Reasoning : {plan.reasoning[:80]}...")
            print(f"  ✓ Sub-queries ({len(plan.sub_queries)}):")
            for sq in plan.sub_queries:
                print(f"      [{sq.source_tab.upper()}] {sq.query}")
                print(f"       Keywords: {sq.keywords}")
        except Exception as e:
            print(f"  ✗ LLM plan failed: {e}")
            print("  → Trying rule-based fallback...")
            try:
                plan = planner.plan_simple(query)
                print(f"  ✓ Intent    : {plan.intent}")
                print(f"  ✓ Sub-queries ({len(plan.sub_queries)}):")
                for sq in plan.sub_queries:
                    print(f"      [{sq.source_tab.upper()}] {sq.query}")
            except Exception as e2:
                print(f"  ✗ Fallback also failed: {e2}")

    print("\n" + "="*55)
    print("  Planner Agent Test Complete!")
    print("="*55 + "\n")