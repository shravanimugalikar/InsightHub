# 🔭 InsightHub
### Multi-Agent RAG System for Research Assistance

InsightHub is a multi-agent Retrieval-Augmented Generation (RAG) system
that helps researchers query, synthesize, and evaluate information from
Documents, Websites, and Research Papers through a unified AI pipeline.

## Architecture
- **Planner Agent** — Query decomposition and routing
- **Retrieval Agent** — Hybrid search across all sources
- **Analysis Agent** — MapReduce synthesis with citations
- **Critic Agent** — Factual grounding evaluation

## Stack
- LangChain + LangGraph (orchestration)
- HuggingFace (free LLM + embeddings)
- FAISS (vector store)
- Streamlit (UI)

## Setup
```bash
# Clone and setup
git clone https://github.com/shravanimugalikar/InsightHub.git
cd InsightHub
setup.bat
```