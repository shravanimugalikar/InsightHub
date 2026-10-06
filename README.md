# 🔬 InsightHub

## Multi-Agent RAG Research Assistant

InsightHub is an AI-powered, multi-source research assistant built using an **Agentic Retrieval-Augmented Generation (RAG)** architecture. It combines academic research papers, user-uploaded documents, and real-time web information in a unified platform to help users retrieve, analyze, and synthesize research information.

The system uses specialized agents for **query planning, information retrieval, and research synthesis**. It also provides semantic document search, context-aware follow-up Q&A, session history, citation tracking, and downloadable PDF research reports.

---

## 🌐 Live Demo

### 🚀 Streamlit Application

**Live App:**  
https://insightapp-fpwidbgqscfat2maxn22xl.streamlit.app/

---


## ✨ Key Features

- 🤖 **Agentic RAG Workflow** for intelligent research planning and synthesis
- 🧠 **Planner Agent** that decomposes complex queries into focused sub-questions
- 📚 **Global Insights** using academic research papers from arXiv
- 📄 **Local Insights** using uploaded PDF, DOCX, TXT, and PPTX documents
- 🌐 **Web Insights** using real-time web search through the Google Serper API
- 🔎 **Semantic Search** using Hugging Face embeddings and ChromaDB
- 📝 **Structured Research Reports** with summaries, findings, analysis, and references
- 💬 **Context-Aware Follow-up Q&A** using session context
- 🗂️ **Session History** for revisiting previous research sessions
- 🔗 **Citation and Reference Tracking** for source traceability
- 📥 **PDF Report Export** using ReportLab
- ⚡ **Fast LLM Inference** using Groq

---

## 🏗️ System Architecture

```text
                         ┌─────────────────────┐
                         │      User / UI      │
                         │     Streamlit       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    Planner Agent    │
                         │ Query Analysis &    │
                         │ Query Decomposition │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Retrieval Layer   │
                         └──────────┬──────────┘
                                    │
             ┌──────────────────────┼──────────────────────┐
             ▼                      ▼                      ▼
      ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
      │ Global      │       │ Local       │       │ Web         │
      │ Insights    │       │ Insights    │       │ Insights    │
      │ arXiv       │       │ ChromaDB    │       │ Serper API  │
      └─────────────┘       └─────────────┘       └─────────────┘
             │                      │                      │
             └──────────────────────┼──────────────────────┘
                                    ▼
                         ┌─────────────────────┐
                         │  Relevant Context   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Research Synthesizer│
                         │      Agent          │
                         │  Groq LLM / LLaMA   │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┼────────────────┐
                    ▼               ▼                ▼
             ┌───────────┐   ┌─────────────┐  ┌─────────────┐
             │ Structured│   │ Follow-up   │  │ PDF Report  │
             │ Report    │   │ Q&A         │  │ Generation  │
             └───────────┘   └─────────────┘  └─────────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │ Session History │
                           └─────────────────┘
```

---

## 🤖 Multi-Agent Workflow

### 1. Planner Agent

The Planner Agent analyzes the user's research query and:

- Understands the research intent
- Decomposes complex queries into focused sub-questions
- Generates a structured research strategy
- Determines whether fresh retrieval is required
- Identifies follow-up questions using previous session context

### 2. Retrieval Agent / Retrieval Layer

The Retrieval Layer obtains relevant information from the selected knowledge source:

- **Global Insights:** arXiv research papers
- **Local Insights:** ChromaDB vector store containing uploaded document chunks
- **Web Insights:** Real-time search results through Google Serper API

### 3. Research Synthesizer Agent

The Research Synthesizer uses the retrieved context with the Groq-powered LLM to generate:

- Executive summaries
- Key findings
- Explanations
- Comparative analysis
- References and source links
- Context-aware responses

---

## 🔄 Agentic RAG Pipeline

```text
User Query
    ↓
Query Analysis
    ↓
Query Decomposition
    ↓
Retrieval Routing
    ↓
 ┌───────────────┬────────────────┐
 ↓               ↓                ↓
arXiv         ChromaDB        Web Search
 ↓               ↓                ↓
 └───────────────┴────────────────┘
                 ↓
        Context Aggregation
                 ↓
          LLM-based Synthesis
                 ↓
       Structured Research Report
                 ↓
      ┌──────────┴──────────┐
      ↓                     ↓
 Follow-up Q&A          PDF Export
```

For local documents, the retrieval pipeline additionally performs:

```text
Uploaded Document
      ↓
Text Extraction
      ↓
Preprocessing
      ↓
Recursive Chunking
      ↓
Embedding Generation
      ↓
ChromaDB Vector Store
      ↓
Semantic Similarity Search
      ↓
Relevant Context
```

---

## 🌍 Insight Modules

### 📚 Global Insights

Global Insights retrieves academic research papers dynamically from the **arXiv repository**.

Features include:

- Academic paper retrieval
- Year From / Year Until filtering
- Sorting by relevance or latest publications
- Research-oriented synthesis
- Citation and source tracking

### 📄 Local Insights

Local Insights allows users to work with their own documents.

Supported formats documented for the project include:

- PDF
- DOCX
- TXT
- PPTX

The uploaded documents are processed, chunked, embedded, and indexed in ChromaDB for semantic retrieval.

### 🌐 Web Insights

Web Insights retrieves current web information through the **Google Serper API**.

It is intended for:

- Latest technology information
- Current trends
- Web-based research
- Recent information and updates

---

## 🧠 Local Document RAG

**Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`

**Vector Database:** ChromaDB

**Embedding Dimension:** 384

Uploaded document chunks are converted into semantic embeddings and stored in the local `./vectorstore/` directory. During a query, semantic similarity search retrieves the most relevant chunks for the generation process.

---

## 📝 Structured Research Reports

InsightHub converts retrieved contextual information into structured Markdown reports containing:

- Executive Summary
- Key Findings
- Analysis
- Explanations
- Comparative insights
- References
- Source URLs

The generated Markdown report can be converted into a downloadable PDF using **ReportLab**.

---

## 💬 Follow-up Q&A

InsightHub supports context-aware conversational interaction after an initial research report.

The system classifies incoming questions as either:

- A new research request
- A follow-up question

For follow-up questions, previous session information, retrieved context, and generated reports can be used to produce a contextual response without requiring the user to repeat the original research query.

---

## 🗂️ Session History

InsightHub maintains local JSON-based session history.

Stored session information includes:

- Original query
- Insight type
- Generated report
- Report summary
- Timestamp
- Starred status
- Research tags
- Research plan
- Sub-questions
- Citations
- Retrieved documents
- Follow-up interactions

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python 3.11 |
| Frontend / UI | Streamlit |
| LLM | `llama-3.3-70b-versatile` via Groq |
| AI Frameworks | LangChain, LangGraph |
| Vector Database | ChromaDB |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Academic Source | arXiv API |
| Web Search | Google Serper API |
| PDF Generation | ReportLab |
| Environment Management | python-dotenv |
| Deployment | Local Server / Streamlit Cloud |


---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/shravanimugalikar/InsightHub
cd InsightHub
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
SERPER_API_KEY=your_serper_api_key
```

### 5. Run the Application

```bash
streamlit run app.py
```

---

## 👩‍💻 Author

**Shravani Mugalikar**
AI/ML Enthusiast | Generative AI Developer