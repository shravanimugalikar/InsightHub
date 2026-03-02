# ============================================================
#  InsightHub — insighthub/agents/critic_agent.py
#  Evaluates analysis quality on 4 dimensions and triggers
#  refinement if scores fall below thresholds
# ============================================================

import sys
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from openai import OpenAI

from insighthub.config.settings import (
    HUGGINGFACE_API_TOKEN,
    LLM_TEMPERATURE,
    CRITIC_RELEVANCE_THRESHOLD,
    CRITIC_COHERENCE_THRESHOLD,
    CRITIC_GROUNDING_THRESHOLD,
    CRITIC_COMPLETENESS_THRESHOLD,
    CRITIC_MAX_RETRIES,
)
from insighthub.agents.analysis_agent import AnalysisAgent, AnalysisResult
from insighthub.agents.retrieval_agent import RetrievalResult

# ── Logging setup ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class CriticScore:
    """Scores for a single evaluation criterion."""
    criterion : str
    score     : float       # 0.0 - 10.0
    threshold : float       # minimum passing score
    feedback  : str = ""    # specific feedback for improvement
    passed    : bool = False

    def __post_init__(self):
        self.passed = self.score >= self.threshold


@dataclass
class CriticResult:
    """
    Full evaluation result from the Critic Agent.
    Contains scores, feedback, and final verdict.
    """
    query          : str
    scores         : List[CriticScore]
    overall_score  : float      # weighted average
    approved       : bool       # True if all criteria pass
    feedback       : str        # consolidated improvement feedback
    retry_needed   : bool       # True if refinement loop needed
    retry_count    : int = 0    # how many retries have been attempted
    final_response : str = ""   # the approved response (after retries)


# ─────────────────────────────────────────────────────────────
#  CRITIC AGENT CLASS
# ─────────────────────────────────────────────────────────────

class CriticAgent:
    """
    Evaluates the quality of the Analysis Agent's response using
    GPT-4o-as-judge style evaluation on 4 dimensions.

    Scoring Dimensions:
        Relevance    (30%) — Does the response answer the query?
        Coherence    (25%) — Is the reasoning logical and fluent?
        Grounding    (30%) — Are claims backed by retrieved sources?
        Completeness (15%) — Are all aspects of the query covered?

    If any score falls below threshold, the Critic triggers a
    refinement loop — sends specific feedback to the Analysis Agent
    to improve the response. Maximum CRITIC_MAX_RETRIES retries.

    Usage:
        critic   = CriticAgent()
        analyser = AnalysisAgent()

        analysis = analyser.analyse(query, retrieval_result)
        result   = critic.evaluate(query, analysis, retrieval_result)

        if result.approved:
            print("APPROVED:", result.final_response)
        else:
            print("REJECTED:", result.feedback)
    """

    # ── Criterion Weights ─────────────────────────────────────
    WEIGHTS = {
        "relevance"   : 0.30,
        "coherence"   : 0.25,
        "grounding"   : 0.30,
        "completeness": 0.15,
    }

    # ── Thresholds ────────────────────────────────────────────
    THRESHOLDS = {
        "relevance"   : CRITIC_RELEVANCE_THRESHOLD,
        "coherence"   : CRITIC_COHERENCE_THRESHOLD,
        "grounding"   : CRITIC_GROUNDING_THRESHOLD,
        "completeness": CRITIC_COMPLETENESS_THRESHOLD,
    }

    # ── Prompt Templates ──────────────────────────────────────

    EVAL_SYSTEM_PROMPT = """You are a strict research quality evaluator for an AI research assistant.
Evaluate the given response on 4 dimensions using scores from 0.0 to 10.0.
Be critical and accurate — do not inflate scores.
Always respond with valid JSON only."""

    EVAL_USER_TEMPLATE = """Research Query: "{query}"

Retrieved Source Chunks:
{source_chunks}

Generated Response:
{response}

Evaluate this response on 4 dimensions. Score each from 0.0 to 10.0:

1. RELEVANCE (Does the response directly answer the query?):
   - 9-10: Perfectly answers the query
   - 7-8:  Mostly answers with minor gaps
   - 5-6:  Partially answers
   - 0-4:  Off-topic or missing key aspects

2. COHERENCE (Is the response logically structured and fluent?):
   - 9-10: Excellent flow, clear reasoning, no contradictions
   - 7-8:  Good structure with minor issues
   - 5-6:  Somewhat coherent but disorganized
   - 0-4:  Incoherent, contradictory, or confusing

3. GROUNDING (Are claims supported by the retrieved source chunks?):
   - 9-10: All claims directly traceable to provided sources
   - 7-8:  Most claims grounded, few unsupported
   - 5-6:  Half the claims are grounded
   - 0-4:  Many hallucinated or unsupported claims

4. COMPLETENESS (Does the response cover all aspects of the query?):
   - 9-10: Comprehensive coverage of all query aspects
   - 7-8:  Covers most aspects
   - 5-6:  Covers main points but misses details
   - 0-4:  Incomplete, major gaps

Respond with JSON only:
{{
  "relevance":    {{"score": 0.0, "feedback": "specific feedback"}},
  "coherence":    {{"score": 0.0, "feedback": "specific feedback"}},
  "grounding":    {{"score": 0.0, "feedback": "specific feedback"}},
  "completeness": {{"score": 0.0, "feedback": "specific feedback"}},
  "overall_feedback": "consolidated feedback for improvement"
}}"""

    REFINE_SYSTEM_PROMPT = """You are a research analyst improving a response based on critic feedback.
Rewrite the response addressing all the critic's concerns.
Keep the same source citations but improve quality.
Respond with JSON only."""

    REFINE_USER_TEMPLATE = """Original Query: "{query}"

Previous Response:
{previous_response}

Critic Feedback:
{feedback}

Failed Criteria:
{failed_criteria}

Retrieved Sources (use these for grounding):
{source_chunks}

Rewrite the response addressing all feedback points.
Respond with JSON only:
{{
  "response": "improved response with [Document], [Website], [Paper] citations",
  "key_findings": ["finding 1", "finding 2", "finding 3"],
  "improvements_made": ["improvement 1", "improvement 2"]
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
        logger.info("CriticAgent initialised (model=%s)", self._model)

    # ── Public Methods ────────────────────────────────────────

    def evaluate(
        self,
        query            : str,
        analysis_result  : AnalysisResult,
        retrieval_result : RetrievalResult,
        analyser         : Optional[AnalysisAgent] = None,
    ) -> CriticResult:
        """
        Main method — evaluates the analysis and triggers refinement if needed.

        Args:
            query:            The original research query
            analysis_result:  Result from AnalysisAgent.analyse()
            retrieval_result: Result from RetrievalAgent.retrieve()
            analyser:         AnalysisAgent instance for refinement loop
                             (if None, refinement is skipped)

        Returns:
            CriticResult with scores, feedback, and final response
        """
        logger.info("Evaluating response for query: '%s'", query[:60])

        # ── Step 1: Score the response ────────────────────────
        scores = self._score_response(query, analysis_result, retrieval_result)
        overall_score = self._compute_overall(scores)
        approved      = all(s.passed for s in scores)

        # Build feedback from failed criteria
        failed   = [s for s in scores if not s.passed]
        feedback = self._build_feedback(scores)

        logger.info("Scores — Relevance:%.1f Coherence:%.1f Grounding:%.1f Completeness:%.1f → Overall:%.1f (%s)",
                    *[s.score for s in scores], overall_score,
                    "APPROVED ✓" if approved else "NEEDS REFINEMENT")

        result = CriticResult(
            query          = query,
            scores         = scores,
            overall_score  = overall_score,
            approved       = approved,
            feedback       = feedback,
            retry_needed   = not approved and analyser is not None,
            retry_count    = 0,
            final_response = analysis_result.response,
        )

        # ── Step 2: Refinement loop ───────────────────────────
        if not approved and analyser is not None:
            result = self._refinement_loop(
                query, analysis_result, retrieval_result, result, analyser
            )

        return result

    def score_only(
        self,
        query            : str,
        analysis_result  : AnalysisResult,
        retrieval_result : RetrievalResult,
    ) -> CriticResult:
        """
        Score without triggering refinement loop.
        Useful for evaluation and testing.
        """
        return self.evaluate(
            query, analysis_result, retrieval_result, analyser=None
        )

    # ── Scoring ───────────────────────────────────────────────

    def _score_response(
        self,
        query            : str,
        analysis_result  : AnalysisResult,
        retrieval_result : RetrievalResult,
    ) -> List[CriticScore]:
        """Score the response using LLM-as-judge."""
        try:
            return self._llm_score(query, analysis_result, retrieval_result)
        except Exception as e:
            logger.warning("LLM scoring failed: %s — using heuristic scoring", str(e))
            return self._heuristic_score(query, analysis_result, retrieval_result)

    def _llm_score(
        self,
        query            : str,
        analysis_result  : AnalysisResult,
        retrieval_result : RetrievalResult,
    ) -> List[CriticScore]:
        """Score using LLM-as-judge."""

        # Format source chunks for context
        source_text = self._format_sources(retrieval_result)

        user_message = self.EVAL_USER_TEMPLATE.format(
            query         = query,
            source_chunks = source_text[:2000],   # limit context length
            response      = analysis_result.response[:1500],
        )

        response = self.client.chat.completions.create(
            model       = self._model,
            messages    = [
                {"role": "system", "content": self.EVAL_SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            max_tokens  = 600,
            temperature = 0.1,
        )

        raw  = response.choices[0].message.content.strip()
        data = self._parse_json(raw)

        scores = []
        for criterion in ["relevance", "coherence", "grounding", "completeness"]:
            crit_data = data.get(criterion, {})
            score     = float(crit_data.get("score", 5.0))
            score     = max(0.0, min(10.0, score))   # clamp to 0-10

            scores.append(CriticScore(
                criterion = criterion,
                score     = score,
                threshold = self.THRESHOLDS[criterion],
                feedback  = crit_data.get("feedback", ""),
            ))

        return scores

    def _heuristic_score(
        self,
        query            : str,
        analysis_result  : AnalysisResult,
        retrieval_result : RetrievalResult,
    ) -> List[CriticScore]:
        """
        Fallback heuristic scoring — no LLM needed.
        Uses simple rule-based checks for each dimension.
        """
        response = analysis_result.response
        chunks   = retrieval_result.chunks

        # Relevance — check query keywords appear in response
        query_words  = set(query.lower().split())
        resp_words   = set(response.lower().split())
        keyword_overlap = len(query_words & resp_words) / max(len(query_words), 1)
        relevance_score = min(10.0, keyword_overlap * 15)

        # Coherence — check response length and structure
        sentences     = response.split(". ")
        coherence_score = min(10.0, len(sentences) * 1.5) if len(sentences) > 2 else 5.0

        # Grounding — check citation tags present
        has_doc_cite  = "[Document]" in response or "[document]" in response
        has_web_cite  = "[Website]"  in response or "[website]"  in response
        has_paper_cite= "[Paper]"    in response or "[paper]"    in response
        citation_count= sum([has_doc_cite, has_web_cite, has_paper_cite])
        grounding_score = min(10.0, 6.0 + citation_count * 1.2)

        # Completeness — check response covers multiple source types
        source_tabs_covered = len(analysis_result.by_source)
        completeness_score  = min(10.0, 5.0 + source_tabs_covered * 1.5)

        return [
            CriticScore("relevance",    relevance_score,    self.THRESHOLDS["relevance"],    "Heuristic score based on keyword overlap"),
            CriticScore("coherence",    coherence_score,    self.THRESHOLDS["coherence"],    "Heuristic score based on response structure"),
            CriticScore("grounding",    grounding_score,    self.THRESHOLDS["grounding"],    "Heuristic score based on citation presence"),
            CriticScore("completeness", completeness_score, self.THRESHOLDS["completeness"], "Heuristic score based on source coverage"),
        ]

    def _compute_overall(self, scores: List[CriticScore]) -> float:
        """Compute weighted overall score."""
        total = sum(
            self.WEIGHTS.get(s.criterion, 0.25) * s.score
            for s in scores
        )
        return round(total, 2)

    # ── Refinement Loop ───────────────────────────────────────

    def _refinement_loop(
        self,
        query            : str,
        analysis_result  : AnalysisResult,
        retrieval_result : RetrievalResult,
        critic_result    : CriticResult,
        analyser         : AnalysisAgent,
    ) -> CriticResult:
        """
        Refinement loop — sends feedback to Analysis Agent and re-evaluates.
        Runs up to CRITIC_MAX_RETRIES times.
        """
        current_response = analysis_result.response
        retry_count      = 0

        while retry_count < CRITIC_MAX_RETRIES:
            retry_count += 1
            logger.info("Refinement loop — attempt %d/%d", retry_count, CRITIC_MAX_RETRIES)

            # Get failed criteria
            failed = [s for s in critic_result.scores if not s.passed]
            failed_text = "\n".join(
                f"- {s.criterion.upper()} (score: {s.score:.1f}/{s.threshold}): {s.feedback}"
                for s in failed
            )

            # Ask LLM to improve the response
            try:
                improved_response = self._refine_response(
                    query, current_response, critic_result.feedback,
                    failed_text, retrieval_result
                )

                # Create updated analysis result
                updated_analysis = AnalysisResult(
                    query         = query,
                    response      = improved_response,
                    by_source     = analysis_result.by_source,
                    citations     = analysis_result.citations,
                    conflicts     = analysis_result.conflicts,
                    key_findings  = analysis_result.key_findings,
                    has_conflicts = analysis_result.has_conflicts,
                )

                # Re-evaluate
                new_scores   = self._score_response(query, updated_analysis, retrieval_result)
                new_overall  = self._compute_overall(new_scores)
                new_approved = all(s.passed for s in new_scores)

                logger.info("Retry %d scores — Overall: %.1f (%s)",
                            retry_count, new_overall,
                            "APPROVED ✓" if new_approved else "still failing")

                critic_result = CriticResult(
                    query          = query,
                    scores         = new_scores,
                    overall_score  = new_overall,
                    approved       = new_approved,
                    feedback       = self._build_feedback(new_scores),
                    retry_needed   = False,
                    retry_count    = retry_count,
                    final_response = improved_response,
                )

                current_response = improved_response

                if new_approved:
                    logger.info("✓ Response approved after %d refinement(s)", retry_count)
                    break

            except Exception as e:
                logger.warning("Refinement failed on attempt %d: %s", retry_count, str(e))
                break

        if not critic_result.approved:
            logger.warning("Response NOT approved after %d retries — returning best attempt",
                           CRITIC_MAX_RETRIES)
            critic_result.final_response = current_response

        return critic_result

    def _refine_response(
        self,
        query            : str,
        previous_response: str,
        feedback         : str,
        failed_criteria  : str,
        retrieval_result : RetrievalResult,
    ) -> str:
        """Ask LLM to improve the response based on critic feedback."""

        source_text = self._format_sources(retrieval_result)

        user_message = self.REFINE_USER_TEMPLATE.format(
            query             = query,
            previous_response = previous_response[:1000],
            feedback          = feedback,
            failed_criteria   = failed_criteria,
            source_chunks     = source_text[:2000],
        )

        response = self.client.chat.completions.create(
            model       = self._model,
            messages    = [
                {"role": "system", "content": self.REFINE_SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            max_tokens  = 800,
            temperature = 0.2,
        )

        raw  = response.choices[0].message.content.strip()
        data = self._parse_json(raw)
        return data.get("response", previous_response)

    # ── Helper Methods ────────────────────────────────────────

    def _build_feedback(self, scores: List[CriticScore]) -> str:
        """Build consolidated feedback string from all scores."""
        failed   = [s for s in scores if not s.passed]
        passed   = [s for s in scores if s.passed]

        parts = []
        if failed:
            parts.append("Issues to fix:")
            for s in failed:
                parts.append(
                    f"  • {s.criterion.upper()} ({s.score:.1f}/{s.threshold}): {s.feedback}"
                )
        if passed:
            parts.append("Passing criteria:")
            for s in passed:
                parts.append(f"  ✓ {s.criterion.upper()} ({s.score:.1f}/{s.threshold})")

        return "\n".join(parts)

    def _format_sources(self, retrieval_result: RetrievalResult) -> str:
        """Format retrieved chunks as context for evaluation."""
        lines = []
        for chunk in retrieval_result.chunks[:5]:   # max 5 chunks
            title = (
                chunk.document.metadata.get("title") or
                chunk.document.metadata.get("filename") or
                chunk.source_tab
            )
            lines.append(f"[{chunk.source_tab.upper()}] {title}:")
            lines.append(chunk.document.page_content[:300])
            lines.append("")
        return "\n".join(lines)

    def _parse_json(self, text: str) -> Dict:
        """Parse JSON from LLM response."""
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            text = match.group(1)

        start = text.find("{")
        end   = text.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError(f"No JSON in response: {text[:150]}")

        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}\nRaw: {text[start:end][:200]}")


# ─────────────────────────────────────────────────────────────
#  QUICK TEST
#  Command: python insighthub/agents/critic_agent.py
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from langchain.schema import Document
    from insighthub.agents.analysis_agent import AnalysisAgent, AnalysisResult, Citation
    from insighthub.agents.retrieval_agent import RetrievalResult, RetrievedChunk

    print("\n" + "="*55)
    print("  InsightHub — Critic Agent Test")
    print("="*55)

    # ── Sample data ───────────────────────────────────────────
    query = "How does RAG reduce hallucinations in language models?"

    sample_chunks = [
        RetrievedChunk(
            document   = Document(
                page_content = "RAG reduces hallucinations by grounding language model responses in retrieved documents. Instead of relying solely on parametric memory, RAG retrieves relevant evidence from a knowledge base and conditions generation on that evidence.",
                metadata     = {"source_tab": "paper", "title": "RAG Paper 2020", "year": "2020", "chunk_id": 0}
            ),
            source_tab = "paper", score = 0.95, rank = 1,
        ),
        RetrievedChunk(
            document   = Document(
                page_content = "LangChain supports building RAG pipelines with FAISS and Chroma vector stores. It provides retrieval chains that automatically fetch relevant documents before generating a response.",
                metadata     = {"source_tab": "website", "domain": "langchain.com", "page_title": "LangChain RAG", "chunk_id": 1}
            ),
            source_tab = "website", score = 0.72, rank = 2,
        ),
        RetrievedChunk(
            document   = Document(
                page_content = "The Retrieval Agent in InsightHub fetches evidence from Documents, Websites, and Research Papers. The Analysis Agent then synthesizes this evidence into a grounded response verified by the Critic Agent.",
                metadata     = {"source_tab": "document", "filename": "insighthub_design.pdf", "year": "2024", "chunk_id": 2}
            ),
            source_tab = "document", score = 0.65, rank = 3,
        ),
    ]

    retrieval_result = RetrievalResult(
        query           = query,
        chunks          = sample_chunks,
        by_source       = {"paper": [sample_chunks[0]], "website": [sample_chunks[1]], "document": [sample_chunks[2]]},
        total_retrieved = 3,
    )

    analysis_result = AnalysisResult(
        query         = query,
        response      = "RAG (Retrieval-Augmented Generation) reduces hallucinations by grounding responses in retrieved evidence [Paper]. Instead of relying on the model's parametric memory, it retrieves relevant documents from a knowledge base [Document]. LangChain supports this through FAISS and Chroma integrations [Website].",
        by_source     = {
            "paper"   : "RAG grounds responses in retrieved documents, reducing hallucinations.",
            "website" : "LangChain supports RAG with FAISS and Chroma vector stores.",
            "document": "InsightHub uses retrieval and analysis agents to produce grounded responses.",
        },
        citations     = [
            Citation(source_tab="paper",    title="RAG Paper 2020",   year="2020"),
            Citation(source_tab="website",  title="LangChain RAG",    year="2024"),
            Citation(source_tab="document", title="insighthub_design.pdf", year="2024"),
        ],
        conflicts     = [],
        key_findings  = [
            "RAG grounds responses in retrieved documents",
            "Reduces hallucinations vs pure parametric models",
            "LangChain simplifies RAG implementation",
        ],
        has_conflicts = False,
        model_used    = "meta-llama/Llama-3.1-8B-Instruct:cerebras",
    )

    # ── Run critic ────────────────────────────────────────────
    print(f"\n  Query  : '{query}'")
    print(f"  Chunks : {len(sample_chunks)} from 3 sources")
    print("-"*55)

    try:
        critic = CriticAgent()
        result = critic.score_only(query, analysis_result, retrieval_result)

        print(f"\n  📊 Scores:")
        for score in result.scores:
            status = "✓ PASS" if score.passed else "✗ FAIL"
            bar    = "█" * int(score.score) + "░" * (10 - int(score.score))
            print(f"    {score.criterion.upper():<14} {bar}  {score.score:.1f}/10  {status}")
            if score.feedback:
                print(f"    {'':14}  → {score.feedback[:70]}")

        print(f"\n  📈 Overall Score : {result.overall_score:.1f}/10")
        print(f"  ✅ Approved      : {result.approved}")
        print(f"  🔄 Retries needed: {result.retry_needed}")

        print(f"\n  📋 Feedback:")
        for line in result.feedback.split("\n"):
            print(f"    {line}")

        # ── Test with refinement loop ─────────────────────────
        if not result.approved:
            print(f"\n  🔄 Testing refinement loop...")
            analyser       = AnalysisAgent()
            result_refined = critic.evaluate(
                query, analysis_result, retrieval_result, analyser=analyser
            )
            print(f"  ✓ After refinement — Overall: {result_refined.overall_score:.1f}/10")
            print(f"  ✓ Approved: {result_refined.approved}")
            print(f"  ✓ Retry count: {result_refined.retry_count}")

    except Exception as e:
        print(f"  ✗ Critic test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*55)
    print("  Critic Agent Test Complete!")
    print("="*55 + "\n")