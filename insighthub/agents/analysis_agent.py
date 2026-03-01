# ============================================================
#  InsightHub — insighthub/agents/analysis_agent.py
#  Synthesizes retrieved chunks into a structured response
#  using MapReduce with source attribution and citation
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
    LLM_TEMPERATURE,
)
from insighthub.agents.retrieval_agent import RetrievalResult, RetrievedChunk

# ── Logging setup ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class Citation:
    """A single citation linking a claim to its source."""
    source_tab  : str       # "document", "website", "paper"
    title       : str       # document title or URL
    chunk_id    : int = 0
    year        : str = ""
    url         : str = ""


@dataclass
class AnalysisResult:
    """
    Full analysis result returned by the Analysis Agent.
    Contains the synthesized response with citations.
    """
    query           : str
    response        : str                           # full synthesized answer
    by_source       : Dict[str, str]                # per-source summaries
    citations       : List[Citation]                # all citations used
    conflicts       : List[str]                     # detected contradictions
    key_findings    : List[str]                     # bullet point highlights
    has_conflicts   : bool  = False
    model_used      : str   = ""


# ─────────────────────────────────────────────────────────────
#  ANALYSIS AGENT CLASS
# ─────────────────────────────────────────────────────────────

class AnalysisAgent:
    """
    Synthesizes retrieved chunks into a structured research response
    using a MapReduce approach with source attribution.

    Workflow:
        MAP:    Extract key claims from each chunk individually
        REDUCE: Synthesize all claims into a coherent answer
        CITE:   Map every claim back to its source
        CHECK:  Detect conflicts between sources

    Usage:
        analyser = AnalysisAgent()
        result   = analyser.analyse(query, retrieval_result)

        print(result.response)
        print(result.by_source)
        print(result.key_findings)
    """

    # ── Prompt Templates ──────────────────────────────────────

    MAP_SYSTEM_PROMPT = """You are a research analyst extracting key claims from a source chunk.
Extract only factual claims directly stated in the text.
Be concise — 2-4 bullet points maximum.
Always respond with a JSON object only."""

    MAP_USER_TEMPLATE = """Source type: {source_tab}
Source title: {title}

Text chunk:
{text}

Extract the key claims from this chunk relevant to the query: "{query}"

Respond with JSON only:
{{
  "claims": ["claim 1", "claim 2", "claim 3"],
  "source_type": "{source_tab}"
}}"""

    REDUCE_SYSTEM_PROMPT = """You are a research synthesis expert writing structured academic summaries.
Synthesize the provided claims into a clear, well-organized research response.
Always cite which source type each claim comes from using [Document], [Website], or [Paper] tags.
Detect and explicitly flag any contradictions between sources.
Structure your response with clear sections.
Respond with JSON only."""

    REDUCE_USER_TEMPLATE = """Research query: "{query}"

Retrieved claims by source:

FROM DOCUMENTS:
{document_claims}

FROM WEBSITES:
{website_claims}

FROM PAPERS:
{paper_claims}

Synthesize these into a structured research response.

Respond with JSON only:
{{
  "response": "full synthesized answer with [Document], [Website], [Paper] citations inline",
  "by_source": {{
    "document": "summary of what documents say (2-3 sentences)",
    "website": "summary of what websites say (2-3 sentences)",
    "paper": "summary of what papers say (2-3 sentences)"
  }},
  "key_findings": ["finding 1", "finding 2", "finding 3", "finding 4"],
  "conflicts": ["contradiction 1 if any"],
  "has_conflicts": false
}}"""

    def __init__(self):
        if not HUGGINGFACE_API_TOKEN:
            raise ValueError(
                "HUGGINGFACE_API_TOKEN not set!\n"
                "Add it to your .env file: HUGGINGFACE_API_TOKEN=hf_xxxx"
            )

        self._model = "meta-llama/Llama-3.1-8B-Instruct:cerebras"
        self.client = OpenAI(
            base_url = "https://router.huggingface.co/v1",
            api_key  = HUGGINGFACE_API_TOKEN,
        )
        logger.info("AnalysisAgent initialised (model=%s)", self._model)

    # ── Public Methods ────────────────────────────────────────

    def analyse(
        self,
        query            : str,
        retrieval_result : RetrievalResult,
    ) -> AnalysisResult:
        """
        Main method — synthesizes retrieved chunks into a structured response.

        Args:
            query:            The original research query
            retrieval_result: Result from RetrievalAgent.retrieve()

        Returns:
            AnalysisResult with response, citations, and key findings
        """
        if not retrieval_result.chunks:
            logger.warning("No chunks to analyse — returning empty result")
            return self._empty_result(query)

        logger.info("Analysing %d chunks for query: '%s'",
                    len(retrieval_result.chunks), query[:60])

        # ── MAP phase ─────────────────────────────────────────
        # Extract claims from each chunk individually
        claims_by_source = self._map_phase(query, retrieval_result.chunks)
        logger.info("MAP complete — extracted claims from %d sources",
                    len(claims_by_source))

        # ── REDUCE phase ──────────────────────────────────────
        # Synthesize all claims into a coherent response
        result = self._reduce_phase(query, claims_by_source)

        # ── Build citations ───────────────────────────────────
        citations = self._build_citations(retrieval_result.chunks)
        result.citations  = citations
        result.model_used = self._model

        logger.info("✓ Analysis complete — %d key findings, %d citations, conflicts=%s",
                    len(result.key_findings), len(citations), result.has_conflicts)
        return result

    # ── MAP Phase ─────────────────────────────────────────────

    def _map_phase(
        self,
        query  : str,
        chunks : List[RetrievedChunk],
    ) -> Dict[str, List[str]]:
        """
        MAP: Extract key claims from each chunk individually.
        Groups claims by source tab.
        """
        claims_by_source = {
            "document": [],
            "website" : [],
            "paper"   : [],
        }

        for chunk in chunks:
            try:
                claims = self._extract_claims(query, chunk)
                source_tab = chunk.source_tab
                claims_by_source[source_tab].extend(claims)
                logger.debug("Extracted %d claims from [%s]", len(claims), source_tab)

            except Exception as e:
                logger.warning("Claim extraction failed for chunk: %s", str(e))
                # Fallback — use chunk text directly as a claim
                fallback = chunk.document.page_content[:200].strip()
                claims_by_source[chunk.source_tab].append(fallback)

        return claims_by_source

    def _extract_claims(
        self,
        query : str,
        chunk : RetrievedChunk,
    ) -> List[str]:
        """Extract key claims from a single chunk using LLM."""
        title = (
            chunk.document.metadata.get("title") or
            chunk.document.metadata.get("filename") or
            chunk.document.metadata.get("domain") or
            chunk.source_tab
        )

        user_message = self.MAP_USER_TEMPLATE.format(
            source_tab = chunk.source_tab,
            title      = title,
            text       = chunk.document.page_content[:800],
            query      = query,
        )

        response = self.client.chat.completions.create(
            model       = self._model,
            messages    = [
                {"role": "system", "content": self.MAP_SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            max_tokens  = 300,
            temperature = 0.1,
        )

        raw = response.choices[0].message.content.strip()
        data = self._parse_json(raw)
        return data.get("claims", [])

    # ── REDUCE Phase ──────────────────────────────────────────

    def _reduce_phase(
        self,
        query            : str,
        claims_by_source : Dict[str, List[str]],
    ) -> AnalysisResult:
        """
        REDUCE: Synthesize all extracted claims into a coherent response.
        Falls back to rule-based synthesis if LLM fails.
        """
        try:
            return self._llm_reduce(query, claims_by_source)
        except Exception as e:
            logger.warning("LLM reduce failed: %s — using rule-based synthesis", str(e))
            return self._rule_based_reduce(query, claims_by_source)

    def _llm_reduce(
        self,
        query            : str,
        claims_by_source : Dict[str, List[str]],
    ) -> AnalysisResult:
        """Synthesize claims using LLM."""

        # Format claims per source
        def format_claims(claims: List[str]) -> str:
            if not claims:
                return "No information retrieved from this source."
            return "\n".join(f"- {c}" for c in claims[:6])  # max 6 claims per source

        user_message = self.REDUCE_USER_TEMPLATE.format(
            query           = query,
            document_claims = format_claims(claims_by_source.get("document", [])),
            website_claims  = format_claims(claims_by_source.get("website",  [])),
            paper_claims    = format_claims(claims_by_source.get("paper",    [])),
        )

        response = self.client.chat.completions.create(
            model       = self._model,
            messages    = [
                {"role": "system", "content": self.REDUCE_SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            max_tokens  = 1000,
            temperature = LLM_TEMPERATURE,
        )

        raw  = response.choices[0].message.content.strip()
        data = self._parse_json(raw)

        return AnalysisResult(
            query         = query,
            response      = data.get("response", ""),
            by_source     = data.get("by_source", {}),
            citations     = [],
            conflicts     = data.get("conflicts", []),
            key_findings  = data.get("key_findings", []),
            has_conflicts = data.get("has_conflicts", False),
        )

    def _rule_based_reduce(
        self,
        query            : str,
        claims_by_source : Dict[str, List[str]],
    ) -> AnalysisResult:
        """
        Fallback synthesis without LLM.
        Structures claims into a readable response.
        """
        sections = []
        by_source = {}
        all_findings = []

        source_labels = {
            "document": "📄 From Documents",
            "website" : "🌐 From Websites",
            "paper"   : "🔬 From Research Papers",
        }

        for source_tab, label in source_labels.items():
            claims = claims_by_source.get(source_tab, [])
            if claims:
                section = f"{label}:\n" + "\n".join(f"• {c}" for c in claims[:4])
                sections.append(section)
                by_source[source_tab] = " ".join(claims[:2])
                all_findings.extend(claims[:2])

        response = f"Research Summary for: {query}\n\n" + "\n\n".join(sections)

        return AnalysisResult(
            query         = query,
            response      = response,
            by_source     = by_source,
            citations     = [],
            conflicts     = [],
            key_findings  = all_findings[:5],
            has_conflicts = False,
        )

    # ── Citation Builder ──────────────────────────────────────

    def _build_citations(self, chunks: List[RetrievedChunk]) -> List[Citation]:
        """Build citation list from retrieved chunks."""
        citations = []
        seen      = set()

        for chunk in chunks:
            meta  = chunk.document.metadata
            title = (
                meta.get("title") or
                meta.get("filename") or
                meta.get("page_title") or
                meta.get("domain") or
                f"{chunk.source_tab} source"
            )

            # Deduplicate citations by title
            if title in seen:
                continue
            seen.add(title)

            citations.append(Citation(
                source_tab = chunk.source_tab,
                title      = title,
                chunk_id   = meta.get("chunk_id", 0),
                year       = str(meta.get("year", "")),
                url        = meta.get("url", meta.get("source", "")),
            ))

        return citations

    # ── Helpers ───────────────────────────────────────────────

    def _parse_json(self, text: str) -> Dict:
        """Parse JSON from LLM response — handles markdown code blocks."""
        # Strip markdown code fences
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            text = match.group(1)

        # Find JSON boundaries
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON found in response: {text[:150]}")

        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}\nRaw: {text[start:end][:200]}")

    def _empty_result(self, query: str) -> AnalysisResult:
        """Return empty result when no chunks are available."""
        return AnalysisResult(
            query        = query,
            response     = (
                "No relevant information was found in the indexed sources. "
                "Please add documents, websites, or research papers first, "
                "then try your query again."
            ),
            by_source    = {},
            citations    = [],
            conflicts    = [],
            key_findings = [],
            has_conflicts= False,
        )


# ─────────────────────────────────────────────────────────────
#  QUICK TEST
#  Command: python insighthub/agents/analysis_agent.py
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from langchain.schema import Document

    print("\n" + "="*55)
    print("  InsightHub — Analysis Agent Test")
    print("="*55)

    # ── Set up sample retrieval result ────────────────────────
    from insighthub.agents.retrieval_agent import RetrievalResult, RetrievedChunk

    sample_chunks = [
        RetrievedChunk(
            document   = Document(
                page_content = "RAG (Retrieval-Augmented Generation) combines a retrieval component with a generative language model. It retrieves relevant documents and uses them as context, significantly reducing hallucinations by grounding responses in evidence.",
                metadata     = {"source_tab": "paper", "title": "RAG: Retrieval-Augmented Generation for NLP", "year": "2020", "chunk_id": 0}
            ),
            source_tab = "paper",
            score      = 0.95,
            rank       = 1,
        ),
        RetrievedChunk(
            document   = Document(
                page_content = "Self-RAG improves upon standard RAG by introducing special reflection tokens that allow the model to decide when retrieval is needed, critique its outputs, and select the most grounded responses adaptively.",
                metadata     = {"source_tab": "paper", "title": "Self-RAG: Learning to Retrieve, Generate, and Critique", "year": "2023", "chunk_id": 1}
            ),
            source_tab = "paper",
            score      = 0.88,
            rank       = 2,
        ),
        RetrievedChunk(
            document   = Document(
                page_content = "LangChain makes it easy to build RAG applications with built-in document loaders, vector store integrations, and chain abstractions. It supports FAISS, Chroma, and Pinecone out of the box.",
                metadata     = {"source_tab": "website", "domain": "python.langchain.com", "page_title": "LangChain RAG Tutorial", "scraped_date": "2024-01-10", "chunk_id": 2}
            ),
            source_tab = "website",
            score      = 0.72,
            rank       = 3,
        ),
        RetrievedChunk(
            document   = Document(
                page_content = "InsightHub architecture uses four specialized agents: Planner, Retrieval, Analysis, and Critic. The Planner decomposes queries, Retrieval fetches evidence, Analysis synthesizes, and Critic evaluates response quality.",
                metadata     = {"source_tab": "document", "filename": "insighthub_architecture.pdf", "year": "2024", "chunk_id": 3}
            ),
            source_tab = "document",
            score      = 0.65,
            rank       = 4,
        ),
    ]

    retrieval_result = RetrievalResult(
        query           = "How does RAG work and what are its key benefits?",
        chunks          = sample_chunks,
        by_source       = {
            "paper"   : [sample_chunks[0], sample_chunks[1]],
            "website" : [sample_chunks[2]],
            "document": [sample_chunks[3]],
        },
        total_retrieved = len(sample_chunks),
    )

    # ── Run analysis ──────────────────────────────────────────
    print("\n[1/1] Running Analysis Agent...")
    print(f"  Query: '{retrieval_result.query}'")
    print(f"  Input chunks: {len(sample_chunks)} ({len(set(c.source_tab for c in sample_chunks))} sources)")
    print("-"*55)

    try:
        analyser = AnalysisAgent()
        result   = analyser.analyse(retrieval_result.query, retrieval_result)

        print(f"\n  ✓ Model used     : {result.model_used}")
        print(f"  ✓ Has conflicts  : {result.has_conflicts}")
        print(f"  ✓ Citations      : {len(result.citations)}")
        print(f"  ✓ Key findings   : {len(result.key_findings)}")

        print(f"\n  📝 Full Response:")
        print("  " + "-"*45)
        # Print response wrapped at 70 chars
        for line in result.response.split("\n"):
            print(f"  {line}")

        print(f"\n  📌 Key Findings:")
        for i, finding in enumerate(result.key_findings, 1):
            print(f"    {i}. {finding}")

        print(f"\n  📚 Citations:")
        for c in result.citations:
            print(f"    [{c.source_tab.upper()}] {c.title} {c.year}")

        if result.by_source:
            print(f"\n  🗂️  By Source:")
            for source, summary in result.by_source.items():
                print(f"    [{source.upper()}] {summary[:100]}...")

        if result.conflicts:
            print(f"\n  ⚠️  Conflicts Detected:")
            for conflict in result.conflicts:
                print(f"    • {conflict}")

    except Exception as e:
        print(f"  ✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*55)
    print("  Analysis Agent Test Complete!")
    print("="*55 + "\n")