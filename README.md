# 🔬 InsightHub

A multi-agent RAG research assistant built with LangGraph, Groq, and Streamlit.

## What it does

Ask a research question and get a structured report sourced from:
- **Global** — arXiv papers
- **Local** — your uploaded PDF / DOCX / TXT files
- **Web** — live web search via Serper

Reports stream token-by-token and support follow-up Q&A.

---

## System Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│  Planner Agent  │  Breaks query into a research plan + sub-questions
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Retrieval Agent │  Fetches docs from arXiv / Chroma / Serper
└────────┬────────┘
         │
         ▼
┌──────────────────┐
│   Synthesizer    │  Writes structured report (streams token-by-token)
│     Agent        │
└────────┬─────────┘
         │
         ▼
    Streamlit UI
    (live stream → report → follow-up Q&A)
```

The three agents are connected using a **LangGraph StateGraph**. After the Planner runs, a conditional edge decides whether retrieval is needed — simple follow-ups skip straight to the Synthesizer.

---

## Agents

### 🧭 Planner Agent
- Takes the user query
- Produces a research plan and a list of sub-questions
- Sets a `needs_retrieval` flag — if `False`, retrieval is skipped

### 📚 Retrieval Agent
- Receives the sub-questions from the Planner
- Routes to the correct backend based on the selected source:

| Source | Backend |
|--------|---------|
| Global Insight | arXiv API |
| Local Insight | Chroma vectorstore |
| Web Insight | Serper (Google Search) |

- Returns ranked document chunks and citation metadata

### ✍️ Synthesizer Agent
- Takes all retrieved documents
- Generates a structured markdown report:
  ```
  ## Executive Summary
  ## Key Findings
  ## Analysis
  ## Conclusion
  ```
- Supports both blocking (follow-ups) and streaming (initial queries) modes

---

## Setup

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
SERPER_API_KEY=your_serper_api_key
```

Run the app:

```bash
streamlit run main.py
```

---

## Stack

- **LangGraph** — agent orchestration
- **Groq** — LLM inference (Llama 3)
- **Chroma** — local vector store
- **HuggingFace** — embeddings
- **Streamlit** — frontend

## Author

Shravani Mugalikar