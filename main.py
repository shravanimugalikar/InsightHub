"""
InsightHub — Main Application
══════════════════════════════════════════════════════════════════════════════
Multi-Agent RAG Research Assistant
  • Sources: Global (arXiv) | Local (File Upload) | Web (Real-time)
  • Follow-up Q&A with context memory per tab
  • Structured report generation + PDF download
══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import os
import time
from dotenv import load_dotenv

load_dotenv()

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InsightHub — Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:            #F5F4F1;
    --bg2:           #e8e7e2;
    --glass:         rgba(0,0,0,0.04);
    --glass2:        rgba(0,0,0,0.07);
    --border:        rgba(0,0,0,0.08);
    --border2:       rgba(110,86,255,0.40);
    --p:             #6e56ff;
    --p2:            #4a3aff;
    --cyan:          #0891b2;
    --green:         #16a34a;
    --orange:        #ea580c;
    --t1:            #1a1a2e;
    --t2:            #4a4a5e;
    --t3:            #888899;
    --r:             16px;
    --rp:            100px;
}

html, body, .stApp {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--t1);
}
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="stSidebarNav"] { display: none; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(110,86,255,0.35); border-radius: 10px; }

/* ══ SIDEBAR ══ */
[data-testid="stSidebar"] {
    background: #e8e7e2 !important;
    border-right: 1px solid var(--border) !important;
    min-width: 250px !important;
    max-width: 250px !important;
}
[data-testid="stSidebar"] > div { padding: 0 !important; }

/* Hide default radio bullets */
[data-testid="stSidebar"] .stRadio > div { gap: 0 !important; }
[data-testid="stSidebar"] .stRadio input[type="radio"] { display: none !important; }

/* Insight radio buttons */
[data-testid="stSidebar"] .stRadio label {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    width: 100% !important;
    padding: 11px 16px !important;
    margin: 2px 0 !important;
    border-radius: 10px !important;
    border: 1px solid transparent !important;
    cursor: pointer !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: var(--t2) !important;
    transition: all 0.22s ease !important;
    background: transparent !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: var(--glass) !important;
    color: var(--t1) !important;
    border-color: var(--border) !important;
}
[data-testid="stSidebar"] .stRadio label[data-checked="true"],
[data-testid="stSidebar"] .stRadio [aria-checked="true"] ~ label,
div[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(135deg, rgba(110,86,255,0.22), rgba(167,139,250,0.12)) !important;
    border-color: rgba(110,86,255,0.45) !important;
    color: var(--t1) !important;
    font-weight: 600 !important;
}

/* Sidebar action buttons */
[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    background: var(--glass) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--t2) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 10px 16px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    transition: all 0.25s ease !important;
    box-shadow: none !important;
    letter-spacing: 0 !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--glass2) !important;
    border-color: var(--border2) !important;
    color: var(--t1) !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ══ MAIN AREA ══ */
.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Main content wrapper ── */
.main-wrap {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
    padding: 64px 60px 40px;
    max-width: 860px;
    margin: 0 auto;
}

/* ── Page heading ── */
.page-heading {
    font-family: 'DM Serif Display', serif;
    font-size: 52px;
    font-weight: 700;
    color: var(--t1);
    letter-spacing: -1px;
    margin-bottom: 48px;
    line-height: 1.25;
    text-align: center;
}
.page-heading span {
    background: linear-gradient(135deg, #6e56ff, #a78bfa, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ── Upload zone ── */
.upload-zone-wrap {
    margin-bottom: 16px;
}
[data-testid="stFileUploader"] {
    background: var(--glass) !important;
    border: 1.5px dashed rgba(110,86,255,0.35) !important;
    border-radius: 14px !important;
    padding: 6px 12px !important;
    transition: border-color 0.25s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(110,86,255,0.6) !important;
}
[data-testid="stFileUploader"] label {
    color: var(--t2) !important;
    font-size: 13px !important;
}
[data-testid="stFileDropzoneInstructions"] {
    color: var(--t2) !important;
}
[data-testid="stFileDropzoneInstructions"] svg { display: none !important; }

/* ── Search pill form ── */
.search-pill-wrap {
    position: relative;
    width: 100%;
}
/* Style the form so it appears inline */
.search-pill-wrap .stTextInput > div > div {
    border-radius: 100px !important;
}
.search-pill-wrap .stTextInput > div > div > input {
    background: rgba(255,255,255,0.92) !important;
    border: 1.5px solid rgba(0,0,0,0.12) !important;
    border-radius: 100px !important;
    color: var(--t1) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 16px !important;
    padding: 18px 72px 18px 28px !important;
    transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
    height: 62px !important;
}
.search-pill-wrap .stTextInput > div > div > input::placeholder {
    color: var(--t3) !important;
    font-size: 15px !important;
}
.search-pill-wrap .stTextInput > div > div > input:focus {
    border-color: rgba(110,86,255,0.55) !important;
    box-shadow: 0 0 0 4px rgba(110,86,255,0.12), 0 8px 40px rgba(110,86,255,0.15) !important;
    outline: none !important;
}
.search-pill-wrap .stTextInput label { display: none !important; }

/* Arrow submit button inside/beside the pill */
.search-pill-wrap .stButton > button {
    background: linear-gradient(135deg, #6e56ff, #a78bfa) !important;
    border: none !important;
    border-radius: 50% !important;
    width: 44px !important;
    height: 44px !important;
    padding: 0 !important;
    font-size: 18px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: 0 4px 18px rgba(110,86,255,0.4) !important;
    transition: all 0.25s ease !important;
    position: absolute !important;
    right: 10px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    min-height: unset !important;
    letter-spacing: 0 !important;
    color: white !important;
}
.search-pill-wrap .stButton > button:hover {
    box-shadow: 0 6px 28px rgba(110,86,255,0.6) !important;
    transform: translateY(-50%) scale(1.07) !important;
}

/* ── Results area ── */
.results-area {
    margin-top: 40px;
}

/* Plan card */
.plan-card {
    background: linear-gradient(135deg, rgba(245,158,11,0.07), rgba(251,191,36,0.03));
    border: 1px solid rgba(245,158,11,0.22);
    border-left: 3px solid #f59e0b;
    border-radius: var(--r);
    padding: 20px 24px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06);
}
.plan-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(110,86,255,0.45), transparent);
}
.plan-label {
    font-size: 10px; font-weight: 800; letter-spacing: 2px;
    text-transform: uppercase; color: #f59e0b; margin-bottom: 10px;
}
.plan-text { font-size: 13.5px; color: var(--t2); line-height: 1.7; }
.sub-list { margin: 12px 0 0; padding: 0; list-style: none; display: flex; flex-direction: column; gap: 5px; }
.sub-list li {
    font-size: 12.5px; color: var(--t2);
    padding: 7px 12px;
    background: rgba(245,158,11,0.05);
    border: 1px solid rgba(245,158,11,0.15);
    border-radius: 8px;
    display: flex; align-items: flex-start; gap: 8px;
}
.sub-list li::before { content: '›'; color: #f59e0b; font-weight: 700; flex-shrink: 0; }

/* Stat row */
.stat-row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 18px; }
.stat-chip {
    background: var(--glass); border: 1px solid var(--border);
    border-radius: 10px; padding: 7px 14px;
    font-size: 12px; color: var(--t2);
    display: flex; align-items: center; gap: 6px;
    letter-spacing: 0.3px;
}
.stat-chip b { color: var(--t1); font-weight: 700; }
.badge-g { background: rgba(74,222,128,0.10); color: #4ade80; border: 1px solid rgba(74,222,128,0.28); border-radius: var(--rp); padding: 5px 14px; font-size: 11px; font-weight: 700; }
.badge-l { background: rgba(251,146,60,0.10); color: #fb923c; border: 1px solid rgba(251,146,60,0.28); border-radius: var(--rp); padding: 5px 14px; font-size: 11px; font-weight: 700; }
.badge-w { background: rgba(34,211,238,0.10); color: #22d3ee; border: 1px solid rgba(34,211,238,0.28); border-radius: var(--rp); padding: 5px 14px; font-size: 11px; font-weight: 700; }

/* Report */
.report-wrap {
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(8px);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 32px 36px;
    margin-bottom: 20px;
    line-height: 1.78;
    position: relative; overflow: hidden;
    box-shadow: 0 4px 24px rgba(0,0,0,0.06);
}
.report-wrap::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(110,86,255,0.4), rgba(34,211,238,0.25), transparent);
}
.report-wrap h2 {
    color: var(--p2) !important; font-size: 15px !important; font-weight: 700 !important;
    margin-top: 26px !important; margin-bottom: 8px !important;
    padding-bottom: 7px; border-bottom: 1px solid rgba(110,86,255,0.14);
    font-family: 'DM Serif Display', serif !important;
}

/* Citations */
.cit-wrap {
    background: var(--glass); border: 1px solid var(--border);
    border-radius: var(--r); padding: 18px 20px; margin-bottom: 20px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06);
}
.cit-label { font-size: 10px; font-weight: 800; letter-spacing: 2px; text-transform: uppercase; color: var(--t3); margin-bottom: 10px; }
.cit-chip {
    display: inline-block; background: rgba(110,86,255,0.09);
    border: 1px solid rgba(110,86,255,0.2); border-radius: 7px;
    padding: 3px 11px; font-size: 10px; color: var(--p2);
    margin: 3px 3px 3px 0;
    font-family: 'JetBrains Mono', monospace;
    transition: background 0.2s ease !important;
}
.cit-chip:hover { background: rgba(110,86,255,0.16) !important; cursor: pointer; }

/* Chat */
.chat-u {
    background: linear-gradient(135deg, rgba(110,86,255,0.18), rgba(167,139,250,0.1));
    border: 1px solid rgba(110,86,255,0.26); border-radius: 14px 14px 4px 14px;
    padding: 11px 17px; margin: 8px 0; font-size: 13.5px;
    max-width: 76%; margin-left: auto;
}
.chat-a {
    background: rgba(255,255,255,0.8); border: 1px solid var(--border);
    border-left: 3px solid var(--border);
    border-radius: 14px 14px 14px 4px; padding: 13px 17px;
    margin: 8px 0; font-size: 13.5px; color: var(--t2);
    line-height: 1.65; max-width: 86%;
}

/* Follow-up section */
.followup-pill .stTextInput > div > div > input {
    border-radius: 12px !important; padding: 13px 18px !important; height: auto !important;
}
.followup-pill .stTextInput > div > div > input:focus {
    border-color: rgba(110,86,255,0.55) !important;
    box-shadow: 0 0 0 4px rgba(110,86,255,0.12) !important;
}
.followup-pill .stButton > button {
    background: linear-gradient(135deg, #6e56ff, #a78bfa) !important;
    border-radius: 10px !important; border: none !important;
    font-weight: 600 !important; padding: 10px 22px !important;
    position: static !important; transform: none !important;
    width: 100% !important; height: auto !important;
    border-radius: 10px !important; color: white !important;
    box-shadow: 0 4px 14px rgba(110,86,255,0.3) !important;
}
.followup-pill .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 22px rgba(110,86,255,0.45) !important;
}

/* Download button */
.stDownloadButton > button {
    background: linear-gradient(135deg, rgba(74,222,128,0.16), rgba(34,211,238,0.10)) !important;
    color: var(--green) !important; border: 1px solid rgba(74,222,128,0.32) !important;
    border-radius: 10px !important; font-weight: 700 !important;
    font-size: 13px !important; padding: 10px 22px !important;
    transition: all 0.25s ease !important;
}
.stDownloadButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(74,222,128,0.22) !important;
}

/* Spinner / status */
.stSpinner > div { border-top-color: var(--p) !important; }
div[data-testid="stStatusWidget"] {
    background: var(--glass) !important; border: 1px solid var(--border) !important; border-radius: 12px !important;
}
.stAlert { border-radius: 12px !important; }

/* Section divider */
.div {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 28px 0;
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fi  { animation: fadeIn 0.38s ease forwards; }
.fi2 { animation: fadeIn 0.38s 0.08s ease both; }
.fi3 { animation: fadeIn 0.38s 0.16s ease both; }

@keyframes pulse { 0%,100%{opacity:1}50%{opacity:0.45} }

/* ══ SESSION HISTORY MODERN STYLES ══ */
.hist-header { margin-bottom: 32px; }
.hist-title {
    font-family: 'DM Serif Display', serif;
    font-size: 36px;
    font-weight: 800;
    color: var(--t1);
    letter-spacing: -0.8px;
    margin-bottom: 8px;
    text-align: left;
}
.hist-title span {
    background: linear-gradient(135deg, #6e56ff, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hist-subtitle { font-size: 14px; color: var(--t3); font-weight: 400; }
.header-divider {
    height: 1px;
    background: linear-gradient(90deg, #6e56ff33, #22d3ee33, transparent);
    margin: 20px 0 32px;
}

/* Search bar */
.search-container { position: relative; width: 100%; margin-bottom: 24px; }
.search-container::before {
    content: '🔍'; position: absolute; left: 18px; top: 50%;
    transform: translateY(-50%); z-index: 5; font-size: 15px; opacity: 0.5;
}
.search-container .stTextInput input {
    border-radius: 100px !important;
    padding-left: 48px !important;
    background: white !important;
    border: 1px solid var(--border) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03) !important;
    transition: all 0.25s ease !important;
    height: 52px !important;
}
.search-container .stTextInput input:focus {
    border-color: var(--p) !important;
    box-shadow: 0 0 0 4px rgba(110,86,255,0.12) !important;
}

/* Chips & Buttons */
.filter-chip-container { display: flex; gap: 8px; margin-bottom: 24px; }
.filter-chip button {
    border-radius: 100px !important;
    padding: 6px 18px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    background: var(--glass) !important;
    border: 1px solid var(--border) !important;
    color: var(--t2) !important;
    transition: all 0.2s ease !important;
}
.filter-chip button:hover { background: var(--glass2) !important; border-color: var(--border2) !important; }

.active-g button { background: rgba(22,163,74,0.06) !important; border-color: var(--green) !important; color: var(--green) !important; }
.active-l button { background: rgba(234,88,12,0.06) !important; border-color: var(--orange) !important; color: var(--orange) !important; }
.active-w button { background: rgba(8,145,178,0.06) !important; border-color: var(--cyan) !important; color: var(--cyan) !important; }

.sort-pill button {
    border-radius: 100px !important; font-size: 12px !important; width: 100%;
    background: var(--glass) !important; border: 1px solid var(--border) !important;
}
.clear-pill button {
    border-radius: 100px !important; font-size: 12px !important; width: 100%;
    background: rgba(239,68,68,0.06) !important; border: 1px solid rgba(239,68,68,0.2) !important;
    color: #ef4444 !important;
}
.clear-pill button:hover { background: rgba(239,68,68,0.12) !important; border-color: #ef4444 !important; }

/* Elevated Card */
.hist-card-modern {
    background: rgba(255,255,255,0.8) !important;
    backdrop-filter: blur(8px);
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 18px 24px !important;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06) !important;
    transition: all 0.28s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer;
    margin-bottom: 12px;
    position: relative;
    text-align: left;
}
.hist-card-modern:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.08) !important;
    border-color: rgba(110,86,255,0.25) !important;
}
.b-g { border-left: 4px solid var(--green) !important; }
.b-l { border-left: 4px solid var(--orange) !important; }
.b-w { border-left: 4px solid var(--cyan) !important; }

.hist-q-modern { font-size: 15px; font-weight: 700; color: var(--t1); margin-bottom: 6px; }
.hist-meta-modern { font-size: 12px; color: var(--t3); display: flex; align-items: center; gap: 8px; }

.badge-pill { padding: 2px 10px; border-radius: 100px; font-size: 10px; font-weight: 700; text-transform: uppercase; }
.bp-g { background: rgba(22,163,74,0.1); color: var(--green); }
.bp-l { background: rgba(234,88,12,0.1); color: var(--orange); }
.bp-w { background: rgba(8,145,178,0.1); color: var(--cyan); }

/* Trash icon-only button */
.trash-modern button {
    width: 32px !important; height: 32px !important; min-height: 32px !important;
    border-radius: 50% !important; border: none !important; padding: 0 !important;
    background: transparent !important; color: var(--t3) !important;
    transition: all 0.2s ease !important;
}
.trash-modern button:hover { background: rgba(239,68,68,0.1) !important; color: #ef4444 !important; }

.empty-state-modern {
    border: 2px dashed var(--border) !important;
    border-radius: 16px !important;
    padding: 64px 20px !important;
    text-align: center;
}

.ghost-back button {
    background: transparent !important; border: 1px solid var(--border) !important;
    border-radius: 10px !important; color: var(--t3) !important; font-size: 13px !important;
}
.ghost-back button:hover { background: rgba(110,86,255,0.06) !important; border-color: var(--p) !important; color: var(--p) !important; }

</style>
""", unsafe_allow_html=True)


# ─── Imports ─────────────────────────────────────────────────────────────────
from app.ingestion.loader import load_document
from app.ingestion.chunker import chunk_documents
from app.retrieval.vectorstore import create_vectorstore
from app.agents.workflow import run_workflow
from app.utils.memory import init_memory, add_to_memory, get_memory, clear_memory, get_last_report
from app.utils.pdf_generator import generate_pdf
from app.utils.history_store import load_history, append_entry, clear_history, remove_entry
import datetime


# ─── Session State ────────────────────────────────────────────────────────────
defaults = {
    "uploaded_db": None,
    "active_agent": None,
    "last_plan": {},
    "view": "research",          # "research" | "history"
    "history": load_history(),   # list of {source, query, result, ts}
    "insight": "Global Insight",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

for tab in ["Global Insight", "Local Insight", "Web Insight"]:
    init_memory(tab)

INSIGHT_SOURCE_MAP = {
    "Global Insight": "Global Insight",
    "Local Insight":  "Local Insight",
    "Web Insight":    "Web Insight",
}


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Brand
    st.markdown("""
    <div style='padding:28px 16px 20px; text-align:center;'>
        <div style='font-size:36px; margin-bottom:10px;
                    filter:drop-shadow(0 0 14px rgba(110,86,255,0.5));'>🔬</div>
        <div style='font-family:Space Grotesk,sans-serif; font-size:20px; font-weight:800;
                    background:linear-gradient(135deg,#6e56ff,#22d3ee);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    letter-spacing:-0.3px;'>InsightHub</div>
    </div>
    """, unsafe_allow_html=True)

    # Insight selector (radio as styled buttons)
    st.markdown(
        "<div style='font-size:10px;font-weight:800;letter-spacing:2px;text-transform:uppercase;"
        "color:var(--t3);padding:0 8px;margin-bottom:8px;'>Insights</div>",
        unsafe_allow_html=True,
    )
    
    # We use a container to precisely target these buttons with CSS
    with st.container():
        # Inject dynamic CSS to highlight the active button
        source_colors = {
            "Global Insight": ("22,163,74",   "#16a34a"),
            "Local Insight":  ("234,88,12",   "#ea580c"),
            "Web Insight":    ("8,145,178",   "#0891b2"),
        }
        rgb, hex_color = source_colors[st.session_state.insight]
        
        # Calculate index for CSS targeting
        all_sources = ["Global Insight", "Local Insight", "Web Insight"]
        active_idx = all_sources.index(st.session_state.insight)
        
        # This CSS targets buttons inside this specific container
        st.markdown(f"""
        <style>
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] > div:nth-child({active_idx + 2}) button {{
            background: rgba({rgb}, 0.08) !important;
            border-color: {hex_color} !important;
            color: {hex_color} !important;
            font-weight: 600 !important;
        }}
        </style>
        """, unsafe_allow_html=True)

        if st.button("🌍  Global Insight", key="btn_global", use_container_width=True):
            st.session_state.insight = "Global Insight"
            st.rerun()
        if st.button("📂  Local Insight", key="btn_local", use_container_width=True):
            st.session_state.insight = "Local Insight"
            st.rerun()
        if st.button("🌐  Web Insight", key="btn_web", use_container_width=True):
            st.session_state.insight = "Web Insight"
            st.rerun()
    
    insight = st.session_state.insight

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.07),transparent);"
        "margin:0 0 14px;'></div>",
        unsafe_allow_html=True,
    )

    # Action buttons
    st.markdown(
        "<div style='font-size:10px;font-weight:800;letter-spacing:2px;text-transform:uppercase;"
        "color:var(--t3);padding:0 8px;margin-bottom:8px;'>Actions</div>",
        unsafe_allow_html=True,
    )

    if st.button("✦  New Research", key="btn_new", use_container_width=True):
        st.session_state.view = "research"
        st.session_state.uploaded_db = None
        st.session_state.active_agent = None
        st.session_state.last_plan = {}
        clear_memory(INSIGHT_SOURCE_MAP.get(insight, "Global Insight"))
        st.rerun()

    if st.button("📋  Session History", key="btn_hist", use_container_width=True):
        st.session_state.view = "history"
        st.rerun()



# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _md_to_html(text: str, source: str = "") -> str:
    import html, re
    h2_color = {
        "Global Insight": "#16a34a",
        "Local Insight":  "#ea580c",
        "Web Insight":    "#0891b2",
    }.get(source, "#4a3aff")
    
    lines = text.split("\n")
    out = []
    for line in lines:
        s = line.strip()
        if s.startswith("## "):
            out.append(f"<h2 style='color:{h2_color}'>{html.escape(s[3:])}</h2>")
        elif s.startswith("# "):
            out.append(f"<h2 style='color:{h2_color}'>{html.escape(s[2:])}</h2>")
        elif s.startswith("- ") or s.startswith("* "):
            out.append(f"<p style='margin:5px 0 5px 16px;color:#8888aa;'>• {html.escape(s[2:])}</p>")
        elif s == "":
            out.append("<br>")
        else:
            e = html.escape(s)
            e = re.sub(r'\*\*(.*?)\*\*', r'<strong style="color:#eeeeff">\1</strong>', e)
            out.append(f"<p style='margin:6px 0;color:#9898bb;'>{e}</p>")
    return "\n".join(out)


def run_with_progress(query, source, uploaded_db=None, chat_history=None):
    result = {}
    with st.status("Researching...", expanded=True) as status:
        st.write("🔍 Analyzing query…")
        st.session_state.active_agent = "Planner"
        time.sleep(0.2)
        try:
            result = run_workflow(
                query=query, source=source,
                uploaded_db=uploaded_db,
                chat_history=chat_history or [],
            )
            st.write("📚 Retrieving documents…")
            st.session_state.active_agent = "Retrieval"
            time.sleep(0.15)
            st.write("✍️ Synthesizing report…")
            st.session_state.active_agent = "Synthesizer"
            time.sleep(0.15)
            status.update(label="✅ Done!", state="complete", expanded=False)
            st.session_state.active_agent = None
        except Exception as e:
            status.update(label=f"❌ {e}", state="error")
            st.session_state.active_agent = None
            st.error(str(e))
            return None
    return result


def render_plan(result):
    plan = result.get("plan", "")
    sub_qs = result.get("sub_questions", [])
    if not plan:
        return
    sub_html = "".join(f"<li>{q}</li>" for q in sub_qs)
    st.markdown(f"""
    <div class='plan-card fi'>
        <div class='plan-label'>🧭 Research Plan</div>
        <div class='plan-text'>{plan}</div>
        {"<ul class='sub-list'>" + sub_html + "</ul>" if sub_qs else ""}
    </div>""", unsafe_allow_html=True)


def render_report(result, badge_cls, badge_label, source):
    report    = result.get("report", "")
    citations = result.get("citations", [])
    docs      = result.get("retrieved_docs", [])
    if not report:
        st.warning("No report generated.")
        return
    st.markdown(f"""
    <div class='stat-row fi'>
        <div class='stat-chip'>📄 <b>{len(docs)}</b> docs</div>
        <div class='stat-chip'>🔗 <b>{len(citations)}</b> citations</div>
        <div class='badge-{badge_cls}'>● {badge_label}</div>
    </div>""", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='report-wrap fi2'>
    {_md_to_html(report, source=source)}
    </div>""", unsafe_allow_html=True)
    if citations:
        chips = "".join(
            f"<span class='cit-chip'>[{i+1}] {c[:70]}{'…' if len(c)>70 else ''}</span>"
            for i, c in enumerate(citations[:15])
        )
        st.markdown(f"""
        <div class='cit-wrap fi3'>
            <div class='cit-label'>🔗 Sources</div>
            {chips}
        </div>""", unsafe_allow_html=True)


def render_followup(tab_name, source, uploaded_db=None):
    st.markdown("<div class='div'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:12px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;"
                "color:#44445a;margin-bottom:14px;'>💬 Follow-up</div>", unsafe_allow_html=True)

    history = get_memory(tab_name)
    # Identify the last report and the user query that produced it to skip them in the chat transcript
    last_assistant_idx = -1
    for i in range(len(history)-1, -1, -1):
        if history[i]["role"] == "assistant":
            last_assistant_idx = i
            break

    skip_indices = set()
    if last_assistant_idx != -1:
        skip_indices.add(last_assistant_idx)
        if last_assistant_idx > 0 and history[last_assistant_idx-1]["role"] == "user":
            skip_indices.add(last_assistant_idx-1)

    for i, msg in enumerate(history):
        if i in skip_indices:
            continue
        cls = "chat-u" if msg["role"] == "user" else "chat-a"
        st.markdown(f"<div class='{cls}'>{msg['content']}</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='followup-pill'>", unsafe_allow_html=True)
        with st.form(key=f"fu_{tab_name}", clear_on_submit=True):
            c1, c2 = st.columns([5, 1])
            with c1:
                q = st.text_input("followup", placeholder="Ask a follow-up…", label_visibility="collapsed")
            with c2:
                sent = st.form_submit_button("↑ Send", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if sent and q.strip():
        add_to_memory(tab_name, "user", q.strip())
        res = run_with_progress(q.strip(), source, uploaded_db, get_memory(tab_name))
        if res and res.get("report"):
            add_to_memory(tab_name, "assistant", res["report"])
            st.session_state.last_plan[insight] = res
            st.rerun()


BADGE = {
    "Global Insight": ("g", "arXiv Global"),
    "Local Insight":  ("l", "Local Document"),
    "Web Insight":    ("w", "Live Web"),
}


# ══════════════════════════════════════════════════════════════════════════════
# SESSION HISTORY VIEW
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.view == "history":
    # ── Initialize History View State ──
    if "hist_search" not in st.session_state: st.session_state.hist_search = ""
    if "hist_filters" not in st.session_state: st.session_state.hist_filters = ["Global Insight", "Local Insight", "Web Insight"]
    if "hist_sort_newest" not in st.session_state: st.session_state.hist_sort_newest = True
    if "hist_confirm_clear" not in st.session_state: st.session_state.hist_confirm_clear = False

    # Card & Controls Style for Session History
    st.markdown("""
    <style>
    /* Target buttons in the list to act as cards */
    div[data-testid="stVerticalBlock"] > div:has(.hist-card-wrapper) div.stButton button {
        background: rgba(255,255,255,0.8) !important;
        backdrop-filter: blur(8px) !important;
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
        padding: 20px 24px 44px 24px !important; /* Space for meta */
        width: 100% !important;
        text-align: left !important;
        display: block !important;
        transition: all 0.28s cubic-bezier(0.4, 0, 0.2, 1) !important;
        min-height: 100px !important;
        color: var(--t1) !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04) !important;
    }
    div[data-testid="stVerticalBlock"] > div:has(.hist-card-wrapper) div.stButton button:hover {
        background: white !important;
        border-color: rgba(110,86,255,0.3) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.08) !important;
    }
    
    /* Meta info layering */
    .hist-card-wrapper { position: relative; pointer-events: none; margin-top: -40px; margin-left: 24px; z-index: 5; }
    
    /* Source borders (applied to the button via helper class in container) */
    div[data-testid="stVerticalBlock"] > div:has(.cc-g) div.stButton button { border-left: 4px solid var(--green) !important; }
    div[data-testid="stVerticalBlock"] > div:has(.cc-l) div.stButton button { border-left: 4px solid var(--orange) !important; }
    div[data-testid="stVerticalBlock"] > div:has(.cc-w) div.stButton button { border-left: 4px solid var(--cyan) !important; }

    /* Trash button positioning */
    .trash-pos { display: flex; align-items: center; justify-content: center; width: 100%; height: 60px; }
    </style>
    """, unsafe_allow_html=True)


    # 1. Page Header
    st.markdown("""
    <div style='padding:64px 60px 0;'>
        <div class='hist-header fi'>
            <div class='hist-title'>Session <span>History</span></div>
            <div class='hist-subtitle'>Review and reload your past research insights</div>
            <div class='header-divider'></div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='padding:0 60px;'>", unsafe_allow_html=True)
    
    # 2. Search bar — FIXED: icon in placeholder
    st.session_state.hist_search = st.text_input("Search", value=st.session_state.hist_search,
                                                placeholder="🔍  Search by query text...", 
                                                label_visibility="collapsed")
    
    # 3. Filter chips & 6. Newest first toggle & 7. Clear history
    sources = [("Global", "Global Insight", "g"), ("Local", "Local Insight", "l"), ("Web", "Web Insight", "w")]
    f_cols = st.columns([1, 1, 1, 1.4, 1.4])
    
    for i, (label, val, c_key) in enumerate(sources):
        is_active = val in st.session_state.hist_filters
        active_cls = f"active-{c_key}" if is_active else ""
        with f_cols[i]:
            st.markdown(f"<div class='filter-chip {active_cls}'>", unsafe_allow_html=True)
            if f_cols[i].button(f"{'●' if is_active else '○'} {label}", key=f"filter_{label}", use_container_width=True):
                if val in st.session_state.hist_filters:
                    if len(st.session_state.hist_filters) > 1:
                        st.session_state.hist_filters.remove(val)
                else:
                    st.session_state.hist_filters.append(val)
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    with f_cols[3]:
        st.markdown("<div class='sort-pill'>", unsafe_allow_html=True)
        # BUG FIXED: Plan string label for sort toggle
        sort_lbl = "Newest first ↓" if st.session_state.hist_sort_newest else "Oldest first ↑"
        if st.button(sort_lbl, use_container_width=True, key="sort_toggle"):
            st.session_state.hist_sort_newest = not st.session_state.hist_sort_newest
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with f_cols[4]:
        st.markdown("<div class='clear-pill'>", unsafe_allow_html=True)
        if not st.session_state.hist_confirm_clear:
            if st.button("🗑 Clear history", use_container_width=True):
                st.session_state.hist_confirm_clear = True
                st.rerun()
        else:
            cc1, cc2 = st.columns(2)
            if cc1.button("✅ Yes", use_container_width=True):
                st.session_state.history = []
                clear_history()
                st.session_state.hist_confirm_clear = False
                st.rerun()
            if cc2.button("❌ No", use_container_width=True):
                st.session_state.hist_confirm_clear = False
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)


    # ── Filtering & Display ──
    filtered = [
        (idx, item) for idx, item in enumerate(st.session_state.history)
        if st.session_state.hist_search.lower() in item["query"].lower()
        and item["insight"] in st.session_state.hist_filters
    ]
    display_list = list(reversed(filtered)) if st.session_state.hist_sort_newest else filtered

    # 8. Empty state
    if not st.session_state.history:
        st.markdown("""
        <div class='empty-state-modern'>
            <div style='font-size:48px;margin-bottom:16px;'>🗂️</div>
            <div style='color:var(--t2);font-size:18px;font-weight:700;margin-bottom:4px;'>Nothing here yet</div>
            <div style='color:var(--t3);font-size:14px;'>Your research queries will appear here</div>
        </div>""", unsafe_allow_html=True)
    elif not display_list:
        st.markdown("<div class='empty-state-modern'><div style='color:var(--t3);font-size:14px;'>No results for this search.</div></div>", unsafe_allow_html=True)
    else:
        # 4. History cards — FIXED: robust layout
        source_styles = {
            "Global Insight": ("#16a34a", "rgba(22,163,74,0.1)"),
            "Local Insight":  ("#ea580c", "rgba(234,88,12,0.1)"),
            "Web Insight":    ("#0891b2", "rgba(8,145,178,0.1)"),
        }
        
        for i, (orig_idx, item) in enumerate(display_list):
            b_cls, b_lbl = BADGE.get(item["insight"], ("g", "Global"))
            q_trunc = item["query"][:85] + "..." if len(item["query"]) > 85 else item["query"]
            source_hex, source_bg = source_styles.get(item["insight"], ("#6e56ff", "rgba(110,86,255,0.1)"))
            
            with st.container():
                col_btn, col_del = st.columns([10, 1])
                
                with col_btn:
                    st.markdown(f"""
                    <div style='border-left: 4px solid {source_hex};
                                background: rgba(255,255,255,0.85);
                                border-radius: 14px;
                                padding: 18px 24px;
                                box-shadow: 0 2px 12px rgba(0,0,0,0.05);
                                cursor: pointer;
                                margin-bottom: 4px;'>
                        <div style='font-size:15px;font-weight:700;color:var(--t1);margin-bottom:8px;'>{q_trunc}</div>
                        <div style='font-size:12px;color:var(--t3);display:flex;align-items:center;gap:8px;'>
                            {item["insight"]} · {item["ts"]}
                            <span style='background:{source_bg};color:{source_hex};padding:2px 10px;
                                         border-radius:100px;font-size:10px;font-weight:700;'>{b_lbl}</span>
                        </div>
                    </div>""", unsafe_allow_html=True)
                    
                    if st.button("Open", key=f"hist_reload_{orig_idx}", use_container_width=True):
                        st.session_state.insight = item["insight"]
                        st.session_state.view    = "research"
                        st.session_state.restore_query = item["query"]
                        st.session_state.last_plan[item["insight"]] = {
                            "report": item.get("report", ""),
                            "restored": True
                        }
                        st.rerun()

                # 5. Trash icon button
                with col_del:
                    st.markdown("<div class='trash-pos trash-modern'>", unsafe_allow_html=True)
                    if st.button("🗑", key=f"hist_del_{orig_idx}", help="Delete this session"):
                        st.session_state.history.pop(orig_idx)
                        remove_entry(orig_idx)
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)

    # 9. Back to Research button
    st.markdown("<div style='margin-top:40px;'>", unsafe_allow_html=True)
    st.markdown("<div class='ghost-back'>", unsafe_allow_html=True)
    if st.button("← Back to Research", key="back_btn"):
        st.session_state.view = "research"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════════════════════
# RESEARCH VIEW
# ══════════════════════════════════════════════════════════════════════════════
else:
    # ── Initialize research query context ──
    query_to_run = ""
    restore_active = False
    
    if "restore_query" in st.session_state:
        query_to_run = st.session_state.pop("restore_query")
        restore_active = True

    source_mem  = INSIGHT_SOURCE_MAP[insight]
    badge_cls, badge_lbl = BADGE[insight][:1][0], BADGE[insight][1]

    # ── Heading ──────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='padding:64px 60px 0;'>
        <p class='page-heading fi'>
            Uncover insights.<br>
            <span>What's your question?</span>
        </p>
    </div>""", unsafe_allow_html=True)

    # ── Main input area ───────────────────────────────────────────────────────
    with st.container():
            st.markdown("<div style='padding:0 60px;'>", unsafe_allow_html=True)

            # ── LOCAL: file upload ────────────────────────────────────────
            if insight == "Local Insight":
                st.markdown("<div class='upload-zone-wrap fi2'>", unsafe_allow_html=True)
                uploaded_file = st.file_uploader(
                    "Drop your file here",
                    type=["pdf", "docx", "txt"],
                    label_visibility="visible",
                )
                st.markdown("</div>", unsafe_allow_html=True)

                if uploaded_file:
                    path = os.path.join(os.getcwd(), "data", "local")
                    os.makedirs(path, exist_ok=True)
                    fp = os.path.join(path, uploaded_file.name)
                    with open(fp, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    with st.spinner("Indexing document…"):
                        docs   = load_document(path)
                        chunks = chunk_documents(docs)
                        st.session_state.uploaded_db = create_vectorstore(chunks)
                    st.success(f"✅ **{uploaded_file.name}** — {len(chunks)} chunks ready.")

            # ── Search pill ──────────────────────────────────────────────
            st.markdown("<div class='search-pill-wrap fi3'>", unsafe_allow_html=True)
            with st.form("query_form", clear_on_submit=True):
                col_input, col_btn = st.columns([12, 1])
                with col_input:
                    # If we just restored, the input value is pre-populated
                    user_input = st.text_input(
                        "query",
                        placeholder="Dive into a topic…",
                        label_visibility="collapsed",
                        value=query_to_run if restore_active else "",
                    )
                with col_btn:
                    submitted = st.form_submit_button("↑")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

    # Determine what query to actually execute
    final_q = ""
    if submitted and user_input.strip():
        final_q = user_input.strip()
    
    # ── Run pipeline ─────────────────────────────────────────────────────────
    # We only auto-run if there's no report yet or if the user explicitly submitted
    if final_q:
        if insight == "Local Insight" and st.session_state.uploaded_db is None:
            st.markdown("<div style='padding:0 60px;'>", unsafe_allow_html=True)
            st.warning("⚠️ Please upload a document first.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            add_to_memory(source_mem, "user", final_q)
            with st.container():
                st.markdown("<div style='padding:0 60px;'>", unsafe_allow_html=True)
                res = run_with_progress(
                    final_q, source_mem,
                    uploaded_db=st.session_state.uploaded_db if insight == "Local Insight" else None,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            if res:
                st.session_state.last_plan[insight] = res
                # Save to history & persistence
                new_entry = {
                    "insight": insight,
                    "query":   final_q,
                    "ts":      datetime.datetime.now().strftime("%H:%M"),
                    "report":  res.get("report", ""),
                }
                st.session_state.history.append(new_entry)
                append_entry(new_entry)
                add_to_memory(source_mem, "assistant", res.get("report", ""))

    # ── Show last result ──────────────────────────────────────────────────────
    if insight in st.session_state.last_plan:
        result = st.session_state.last_plan[insight]
        st.markdown("<div style='padding:0 60px;margin-top:32px;'>", unsafe_allow_html=True)

        render_plan(result)
        render_report(result, badge_cls, badge_lbl, insight)

        pdf = generate_pdf(result.get("report", ""), f"InsightHub")
        fn_map = {"Global Insight": "global", "Local Insight": "local", "Web Insight": "web"}
        st.download_button(
            label="📄 Download Report as PDF",
            data=pdf,
            file_name=f"insighthub_{fn_map[insight]}_report.pdf",
            mime="application/pdf",
        )

        # Follow-up Q&A
        last_report = get_last_report(source_mem)
        if last_report:
            render_followup(
                source_mem, source_mem,
                uploaded_db=st.session_state.uploaded_db if insight == "Local Insight" else None,
            )

        st.markdown("</div>", unsafe_allow_html=True)