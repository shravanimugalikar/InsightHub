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
import datetime
import re
from dotenv import load_dotenv

load_dotenv()
import base64

def get_base64_img(path):
    if not os.path.exists(path): return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InsightHub — Research Assistant",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:            #F8F9FA; /* SaaS Light Gray */
    --bg2:           #f1f3f5;
    --glass:         rgba(0,0,0,0.03);
    --glass2:        rgba(0,0,0,0.06);
    --border:        rgba(0,0,0,0.08);
    --border2:       rgba(110,86,255,0.25);
    --p:             #5e48ff;
    --p2:            #4433ff;
    --cyan:          #0891b2;
    --green:         #10b981;
    --orange:        #f59e0b;
    --t1:            #111827;
    --t2:            #4b5563;
    --t3:            #9ca3af;
    --r:             12px;
    --rp:            100px;
    --shadow:        0 1px 3px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.1);
    --shadow-hover:  0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05);
    --sb-bg:         linear-gradient(180deg, #eeecea 0%, #e4e2dc 100%);
    --card-bg:       #ffffff;
    --report-bg:     rgba(255,255,255,0.85);
    --input-bg:      rgba(255,255,255,0.92);
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
    background: var(--sb-bg) !important;
    border-right: 1px solid var(--border) !important;
    min-width: 250px !important;
    max-width: 250px !important;
}
[data-testid="stSidebar"] > div { padding: 0 !important; }

/* Sidebar section labels */
.sb-label {
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #a09c94;
    padding: 0 14px;
    margin-bottom: 6px;
    display: block;
}

/* Sidebar divider */
.sb-divider {
    height: 1px;
    margin: 10px 14px 14px;
    background: linear-gradient(90deg, transparent, rgba(0,0,0,0.10), transparent);
}

/* ── All sidebar buttons base ── */
[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 10px !important;
    color: #5a5650 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 10px 14px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    transition: all 0.2s ease !important;
    box-shadow: none !important;
    letter-spacing: 0 !important;
    margin-bottom: 2px !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(0,0,0,0.05) !important;
    border-color: rgba(0,0,0,0.08) !important;
    color: #1a1816 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Insight buttons: color per source ── */
/* Global — green */
[data-testid="stSidebar"] .btn-global button {
    border-left: 3px solid transparent !important;
}
[data-testid="stSidebar"] .btn-global button:hover {
    border-left-color: #16a34a !important;
    color: #16a34a !important;
    background: rgba(22,163,74,0.07) !important;
}
[data-testid="stSidebar"] .btn-global.active button {
    background: rgba(22,163,74,0.10) !important;
    border-left-color: #16a34a !important;
    border-color: rgba(22,163,74,0.30) !important;
    color: #16a34a !important;
    font-weight: 700 !important;
}

/* Local — orange */
[data-testid="stSidebar"] .btn-local button {
    border-left: 3px solid transparent !important;
}
[data-testid="stSidebar"] .btn-local button:hover {
    border-left-color: #ea580c !important;
    color: #ea580c !important;
    background: rgba(234,88,12,0.07) !important;
}
[data-testid="stSidebar"] .btn-local.active button {
    background: rgba(234,88,12,0.10) !important;
    border-left-color: #ea580c !important;
    border-color: rgba(234,88,12,0.30) !important;
    color: #ea580c !important;
    font-weight: 700 !important;
}

/* Web — cyan */
[data-testid="stSidebar"] .btn-web button {
    border-left: 3px solid transparent !important;
}
[data-testid="stSidebar"] .btn-web button:hover {
    border-left-color: #0891b2 !important;
    color: #0891b2 !important;
    background: rgba(8,145,178,0.07) !important;
}
[data-testid="stSidebar"] .btn-web.active button {
    background: rgba(8,145,178,0.10) !important;
    border-left-color: #0891b2 !important;
    border-color: rgba(8,145,178,0.30) !important;
    color: #0891b2 !important;
    font-weight: 700 !important;
}

/* ── Action buttons (New Research / Session History) ── */
[data-testid="stSidebar"] .sb-action button {
    background: rgba(110,86,255,0.07) !important;
    border: 1px solid rgba(110,86,255,0.18) !important;
    color: #5e48ff !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
}
[data-testid="stSidebar"] .sb-action button:hover {
    background: rgba(110,86,255,0.14) !important;
    border-color: rgba(110,86,255,0.35) !important;
    color: #4433ff !important;
}

/* Active view highlight for action buttons */
[data-testid="stSidebar"] .sb-action.active button {
    background: rgba(110,86,255,0.15) !important;
    border-color: rgba(110,86,255,0.40) !important;
    color: #4433ff !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 8px rgba(110,86,255,0.15) !important;
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
    font-family: 'DM Serif Display', serif !important;
    font-size: 42px !important;
    font-weight: 700 !important;
    color: var(--t1) !important;
    letter-spacing: -2px !important;
    margin-bottom: 45px !important;
    line-height: 1.2 !important;
    text-align: center !important;
}
.page-heading span {
    background: linear-gradient(135deg, #6e56ff, #a78bfa, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    display: inline-block;
    padding-top: 10px;
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
    background: var(--input-bg) !important;
    border: 1.5px solid var(--border) !important;
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
    background: var(--report-bg);
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
.hist-page { padding: 0px 5px 10px; max-width: 860px; }
.hist-header { margin-bottom: 16px; }


/* ── Header ── */
.hist-title {
    font-family: 'DM Serif Display', serif;
    font-size: 40px; font-weight: 800; color: var(--t1);
    letter-spacing: -1px; margin-bottom: 6px;
}
.hist-title span {
    background: linear-gradient(135deg, #6e56ff, #22d3ee);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hist-subtitle { font-size: 15px; color: var(--t3); margin-bottom: 0; }
.header-divider {
    height: 1px; margin: 16px 0 28px;
    background: linear-gradient(90deg, rgba(110,86,255,0.25), rgba(34,211,238,0.15), transparent);
}

/* ── Search bar ── */
.hist-page .stTextInput > div > div > input {
    border-radius: 10px !important;
    background: var(--input-bg) !important;
    border: 1px solid var(--border) !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
    height: 46px !important;
    font-size: 14px !important;
    padding: 0 16px !important;
    transition: all 0.2s ease !important;
}
.hist-page .stTextInput > div > div > input:focus {
    border-color: var(--p) !important;
    box-shadow: 0 0 0 3px rgba(110,86,255,0.10) !important;
}
.hist-page .stTextInput label { display: none !important; }

/* ── Filter row buttons ── */
.filter-chip button {
    border-radius: 8px !important; padding: 6px 16px !important;
    font-size: 12px !important; font-weight: 600 !important;
    background: var(--card-bg) !important; border: 1px solid var(--border) !important;
    color: var(--t2) !important; transition: all 0.2s ease !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}
.filter-chip button:hover { border-color: var(--p) !important; color: var(--p) !important; }
.active-g button { background: rgba(16,185,129,0.08) !important; border-color: #10b981 !important; color: #10b981 !important; }
.active-l button { background: rgba(245,158,11,0.08) !important; border-color: #f59e0b !important; color: #f59e0b !important; }
.active-w button { background: rgba(8,145,178,0.08) !important; border-color: #0891b2 !important; color: #0891b2 !important; }

.sort-pill button {
    border-radius: 8px !important; font-size: 12px !important; width: 100% !important;
    background: var(--card-bg) !important; border: 1px solid var(--border) !important;
    color: var(--t2) !important; box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}
.clear-pill button {
    border-radius: 8px !important; font-size: 12px !important; width: 100% !important;
    background: rgba(239,68,68,0.05) !important; border: 1px solid rgba(239,68,68,0.18) !important;
    color: #ef4444 !important;
}
.clear-pill button:hover { background: rgba(239,68,68,0.10) !important; }

/* ── Date group label ── */
.hist-date-group {
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.2px; color: var(--t3);
    margin: 28px 0 12px; padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}

/* ── History card (static) ── */
.hist-card-row {
    display: flex;
    align-items: center;
    gap: 12px;
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0 12px;
    height: 40px !important;
    min-height: 40px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    transition: all 0.2s ease;
}

.hist-card-row:hover { box-shadow: 0 4px 14px rgba(0,0,0,0.08); }
.hist-card-badge {
    flex-shrink: 0;
    padding: 3px 9px; border-radius: 5px;
    font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.4px;
    white-space: nowrap;
}
.hist-card-text {
    flex: 1;
    font-size: 14px; font-weight: 500;
    color: var(--t1); line-height: 1.4;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
/* Open button — symbol only */
.hist-open-btn button {
    background: rgba(110,86,255,0.07) !important;
    border: 1px solid rgba(110,86,255,0.2) !important;
    border-radius: 8px !important; color: var(--p) !important;
    font-size: 15px !important; font-weight: 600 !important;
    padding: 0 !important;
    min-height: 40px !important; height: 40px !important;
    width: 40px !important; min-width: 40px !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    line-height: 1 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03) !important;
}
.hist-open-btn { display: flex; align-items: center; justify-content: center; width: 100%; height: 100%; }






.hist-open-btn button:hover { background: rgba(110,86,255,0.16) !important; }
/* Delete button — symbol only */
.hist-del-btn button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important; color: var(--t3) !important;
    font-size: 15px !important; font-weight: 600 !important;
    padding: 0 !important;
    min-height: 40px !important; height: 40px !important;
    width: 40px !important; min-width: 40px !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    line-height: 1 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03) !important;
}
.hist-del-btn { display: flex; align-items: center; justify-content: center; width: 100%; height: 100%; }
.hist-del-btn button:hover {
    border-color: #ef4444 !important;
    color: #ef4444 !important;
    background: rgba(239,68,68,0.05) !important;
}
/* Remove Streamlit default column gap for the card row */
.hist-row-wrap {
    margin-bottom: 2px !important;
}
/* The outer horizontal flex container */
.hist-row-wrap [data-testid="stHorizontalBlock"] {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    gap: 6px !important;
    flex-wrap: nowrap !important;
}
/* Each column: fixed 40px height, fully centered */
.hist-row-wrap [data-testid="column"] {
    padding: 0 !important;
    height: 40px !important;
    min-height: 40px !important;
    max-height: 40px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    overflow: visible !important;
}
/* Collapse Streamlit's inner stVerticalBlock wrappers */
.hist-row-wrap [data-testid="column"] [data-testid="stVerticalBlock"] {
    gap: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    width: 100% !important;
    height: 40px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
/* Collapse any intermediate div wrappers inside columns */
.hist-row-wrap [data-testid="column"] > div,
.hist-row-wrap [data-testid="column"] > div > div {
    width: 100% !important;
    height: 40px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 0 !important;
    margin: 0 !important;
}
/* Button elements themselves */
.hist-row-wrap [data-testid="column"] .stButton {
    width: 100% !important;
    height: 40px !important;
    margin: 0 !important;
    padding: 0 !important;
}
.hist-row-wrap [data-testid="column"] .stButton > button {
    height: 40px !important;
    min-height: 40px !important;
    width: 40px !important;
    min-width: 40px !important;
    margin: 0 !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}




/*.hist-row-wrap [data-testid="column"]:first-child {
    align-items: stretch !important;
}*/




/* ── Empty state ── */
.hist-empty {
    border: 2px dashed var(--border); border-radius: 14px;
    padding: 72px 20px; text-align: center; margin-top: 20px;
}

/* ── Back button ── */
.ghost-back button {
    background: transparent !important; border: 1px solid var(--border) !important;
    border-radius: 8px !important; color: var(--t3) !important;
    font-size: 13px !important; transition: all 0.22s ease !important;
}
.ghost-back button:hover { border-color: var(--p) !important; color: var(--p) !important; }

/* Open button */
.hist-action-btn button {
    background: var(--card-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--t2) !important;
    font-size: 15px !important;
    width: 100% !important;
    height: 50px !important;
    min-height: 50px !important;
    align-self: stretch !important;
    padding: 0 !important;
    transition: all 0.18s ease !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}
.hist-action-btn button:hover {
    border-color: var(--p) !important;
    color: var(--p) !important;
    background: rgba(110,86,255,0.05) !important;
}

/* Delete button */
.hist-del-btn button {
    background: white !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--t3) !important;
    font-size: 15px !important;
    width: 100% !important;
    height: 50px !important;
    min-height: 50px !important;
    align-self: stretch !important;
    padding: 0 !important;
    transition: all 0.18s ease !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}
.hist-del-btn button:hover {
    border-color: #ef4444 !important;
    color: #ef4444 !important;
    background: rgba(239,68,68,0.05) !important;
}

/* ── Card row spacing: collapse all Streamlit vertical rhythm ── */

/* The outer stVerticalBlock that holds all the card rows */
[data-testid="stVerticalBlock"] {
    gap: 4px !important;
}

/* Border wrapper added by newer Streamlit versions */
[data-testid="stVerticalBlockBorderWrapper"] {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}

/* Element container wrapping each st.columns() call */
.element-container {
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}

/* Fix card row horizontal layout */
[data-testid="stHorizontalBlock"] {
    gap: 8px !important;
    margin-bottom: 0 !important;
    margin-top: 0 !important;
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    align-items: stretch !important;
}

/* Zero out the column's own vertical padding */
[data-testid="stHorizontalBlock"] > [data-testid="column"] {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}

/* Inner vertical block inside each column */
[data-testid="stHorizontalBlock"] > [data-testid="column"] > [data-testid="stVerticalBlock"] {
    gap: 0 !important;
}


</style>
""", unsafe_allow_html=True)


# ─── Imports ─────────────────────────────────────────────────────────────────
from app.ingestion.loader import load_document
from app.ingestion.chunker import chunk_documents
from app.retrieval.vectorstore import create_vectorstore
from app.agents.workflow import run_workflow
from app.utils.memory import init_memory, add_to_memory, get_memory, clear_memory, get_last_report
from app.utils.pdf_generator import generate_pdf
from app.utils.history_store import load_history, save_history, append_entry, clear_history, remove_entry
import datetime


# ─── Session State ────────────────────────────────────────────────────────────
defaults = {
    "uploaded_db": None,
    "active_agent": None,
    "last_plan": {},
    "followup_results": {},   # { tab_name: [list of result dicts] }
    "view": "research",          # "research" | "history"
    "history": load_history(),   # list of {source, query, result, ts}
    "insight": "Global Insight",
    "hist_search": "",
    "hist_filters": ["Global Insight", "Local Insight", "Web Insight"],
    "hist_sort_newest": True,
    "hist_confirm_clear": False,
    "came_from_history": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ──────────────────────────────────────────────────────────────────────────────

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
    logo_b64 = get_base64_img("insighthub-logo.png")
    st.markdown(f"""
        <div style='padding: 14px 20px 16px; text-align: center;'>
            <div style='display: flex; flex-direction: column; align-items: center; gap: 8px;'>
                <div style='width:64px; height:64px; border-radius:50%;
                            background: linear-gradient(135deg,rgba(110,86,255,0.15),rgba(34,211,238,0.12));
                            border: 1.5px solid rgba(110,86,255,0.20);
                            display:flex; align-items:center; justify-content:center;
                            box-shadow: 0 4px 16px rgba(110,86,255,0.10);'>
                    <img src='data:image/png;base64,{logo_b64}' width='44'
                         style='filter: drop-shadow(0 2px 8px rgba(110,86,255,0.18));'>
                </div>
                <div style='font-family: "DM Serif Display", serif; font-size: 26px; font-weight: 700;
                            background: linear-gradient(120deg, #3b82f6, #6e56ff);
                            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                            letter-spacing: -1px; line-height: 1.1;'>InsightHub</div>
                <div style='font-size:10px; color:#a09c94; letter-spacing:0.5px;
                            font-weight:600; text-transform:uppercase;'>Research Assistant</div>
            </div>
        </div>
        <div style='height:1px; margin:0 14px 14px;
                    background:linear-gradient(90deg,transparent,rgba(0,0,0,0.10),transparent);'></div>
    """, unsafe_allow_html=True)

    # Insight selector
    st.markdown("<span class='sb-label'>Insights</span>", unsafe_allow_html=True)

    insight_cfg = {
        "Global Insight": "btn-global",
        "Local Insight":  "btn-local",
        "Web Insight":    "btn-web",
    }

    with st.container():
        for src, cls in insight_cfg.items():
            active_cls = "active" if st.session_state.insight == src else ""
            st.markdown(f"<div class='{cls} {active_cls}'>", unsafe_allow_html=True)
            if st.button(src, key=f"btn_{cls}", use_container_width=True):
                st.session_state.insight = src
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    insight = st.session_state.insight

    st.markdown("<div class='sb-divider'></div>", unsafe_allow_html=True)

    # Action buttons
    st.markdown("<span class='sb-label'>Actions</span>", unsafe_allow_html=True)

    new_active = "active" if st.session_state.view == "research" else ""
    st.markdown(f"<div class='sb-action {new_active}'>", unsafe_allow_html=True)
    if st.button("New Research", key="btn_new", use_container_width=True):
        st.session_state.view = "research"
        st.session_state.uploaded_db = None
        st.session_state.active_agent = None
        st.session_state.last_plan = {}
        st.session_state.followup_results = {}   # ← ADD THIS
        clear_memory(INSIGHT_SOURCE_MAP.get(insight, "Global Insight"))
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    hist_active = "active" if st.session_state.view == "history" else ""
    st.markdown(f"<div class='sb-action {hist_active}'>", unsafe_allow_html=True)
    if st.button("Session History", key="btn_hist", use_container_width=True):
        st.session_state.view = "history"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)



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
            out.append(f"<p style='margin:5px 0 5px 16px;color:var(--t2);'>• {html.escape(s[2:])}</p>")
        elif s == "":
            out.append("<br>")
        else:
            e = html.escape(s)
            e = re.sub(r'\*\*(.*?)\*\*', r'<strong style="color:var(--t1)">\1</strong>', e)
            out.append(f"<p style='margin:6px 0;color:var(--t2);'>{e}</p>")
    return "\n".join(out)


def run_with_progress(query, source, uploaded_db=None, chat_history=None):
    result = {}
    with st.status("Researching...", expanded=True) as status:
        st.write("Analyzing query…")
        st.session_state.active_agent = "Planner"
        time.sleep(0.2)
        try:
            result = run_workflow(
                query=query, source=source,
                uploaded_db=uploaded_db,
                chat_history=chat_history or [],
            )
            st.write("Retrieving documents…")
            st.session_state.active_agent = "Retrieval"
            time.sleep(0.15)
            st.write("Synthesizing report…")
            st.session_state.active_agent = "Synthesizer"
            time.sleep(0.15)
            status.update(label="Done!", state="complete", expanded=False)
            st.session_state.active_agent = None
        except Exception as e:
            status.update(label=f"Error: {e}", state="error")
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
        <div class='plan-label'>Research Plan</div>
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
        <div class='stat-chip'><b>{len(docs)}</b> docs</div>
        <div class='stat-chip'><b>{len(citations)}</b> citations</div>
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
            <div class='cit-label'>Sources</div>
            {chips}
        </div>""", unsafe_allow_html=True)


def render_followup(tab_name, source, uploaded_db=None):
    st.markdown("<div class='div'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:12px;font-weight:700;letter-spacing:1.5px;"
        "text-transform:uppercase;color:var(--t3);margin-bottom:20px;'>💬 Follow-up</div>",
        unsafe_allow_html=True,
    )

    # Render all previous follow-up exchanges
    followup_list = st.session_state.followup_results.get(tab_name, [])
    for i, fu in enumerate(followup_list):
        # User question bubble
        st.markdown(
            f"<div class='chat-u'>{fu['query']}</div>",
            unsafe_allow_html=True,
        )
        # Context pill — shows what topic the answer is grounded in
        original_q = st.session_state.last_plan.get(insight, {}).get("query", "")
        if original_q:
            st.markdown(f"""
            <div style='font-size:11px; color:var(--t3); margin:6px 0 10px;
                        display:flex; align-items:center; gap:6px;'>
                <span style='background:rgba(110,86,255,0.08); color:var(--p);
                             padding:2px 10px; border-radius:100px;
                             font-weight:600; font-size:10px;'>
                    In context of: {original_q[:50]}{'…' if len(original_q)>50 else ''}
                </span>
            </div>""", unsafe_allow_html=True)
        # Assistant answer as styled report card
        st.markdown(
            f"<div class='report-wrap fi'>{_md_to_html(fu['report'], source=source)}</div>",
            unsafe_allow_html=True,
        )
        # Citations if present
        if fu.get("citations"):
            chips = "".join(
                f"<span class='cit-chip'>[{j+1}] {c[:70]}{'…' if len(c)>70 else ''}</span>"
                for j, c in enumerate(fu["citations"][:15])
            )
            st.markdown(f"""
            <div class='cit-wrap'>
                <div class='cit-label'>🔗 Sources</div>
                {chips}
            </div>""", unsafe_allow_html=True)
        # Download button for this follow-up
        fu_pdf = generate_pdf(fu["report"], "InsightHub", query=fu["query"])
        st.download_button(
            label="📄 Download Follow-up Report",
            data=fu_pdf,
            file_name=f"insighthub_followup_{i+1}.pdf",
            mime="application/pdf",
            key=f"dl_fu_{tab_name}_{i}",
        )
        st.markdown("<div class='div'></div>", unsafe_allow_html=True)

    # Follow-up input form
    st.markdown("<div class='followup-pill'>", unsafe_allow_html=True)
    with st.form(key=f"fu_{tab_name}", clear_on_submit=True):
        c1, c2 = st.columns([5, 1])
        with c1:
            q = st.text_input(
                "followup",
                placeholder="Ask a follow-up…",
                label_visibility="collapsed",
            )
        with c2:
            sent = st.form_submit_button("↑ Send", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if sent and q.strip():
        # Build a concise chat history for the follow-up
        initial_report = st.session_state.last_plan.get(insight, {}).get("report", "")
        initial_query  = st.session_state.last_plan.get(insight, {}).get("query", "")

        # Summarise the initial report to first 400 chars as context
        report_summary = initial_report[:400].strip() + "…" if len(initial_report) > 400 else initial_report

        focused_history = []

        # Add the original query as user turn
        if initial_query:
            focused_history.append({
                "role":    "user",
                "content": initial_query,
            })
            # Add a short summary of the initial report as assistant turn
            if report_summary:
                focused_history.append({
                    "role":    "assistant",
                    "content": f"Research summary: {report_summary}",
                })

        # Add any previous follow-up exchanges (last 2 only to keep context tight)
        prev_followups = st.session_state.followup_results.get(tab_name, [])
        for fu in prev_followups[-2:]:
            focused_history.append({"role": "user",      "content": fu["query"]})
            focused_history.append({"role": "assistant",  "content": fu["report"][:300] + "…"})

        # Add the current follow-up question
        focused_history.append({"role": "user", "content": q.strip()})

        # Rewrite follow-up to be self-contained with topic context
        topic = initial_query or tab_name
        if topic and topic.lower() not in q.strip().lower():
            # Query doesn't mention the topic — prepend it
            contextual_query = f"Regarding {topic}: {q.strip()}"
        else:
            contextual_query = q.strip()

        add_to_memory(tab_name, "user", q.strip())
        res = run_with_progress(
            contextual_query, source, uploaded_db, focused_history
        )
        if res and res.get("report"):
            add_to_memory(tab_name, "assistant", res["report"])
            
            if tab_name not in st.session_state.followup_results:
                st.session_state.followup_results[tab_name] = []
                
            st.session_state.followup_results[tab_name].append({
                "query":          q.strip(),          # show original to user
                "full_query":     contextual_query,   # what was actually sent
                "report":         res.get("report", ""),
                "citations":      res.get("citations", []),
                "retrieved_docs": res.get("retrieved_docs", []),
            })

            # ── Persist follow-ups to history ──
            lp = st.session_state.last_plan.get(source, {})
            target_ts = lp.get("iso_ts")
            target_q  = lp.get("query")
            
            # Find and update the history entry
            for item in st.session_state.history:
                # Match by iso_ts if available, else by query + ts
                match = (target_ts and item.get("iso_ts") == target_ts) or \
                        (not target_ts and item.get("query") == target_q)
                
                if match:
                    item["followups"] = st.session_state.followup_results[tab_name]
                    save_history(st.session_state.history)
                    break

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
    st.markdown("<div class='hist-page'>", unsafe_allow_html=True)

    # Header
    st.markdown("""
<div class='hist-header'>
    <div class='hist-title'>Session <span>History</span></div>
    <div class='hist-subtitle'>Review and reload your past research insights</div>
</div>
<div class='header-divider'></div>
""", unsafe_allow_html=True)


    # Search
    st.session_state.hist_search = st.text_input(
        "Search", value=st.session_state.hist_search,
        placeholder="🔍  Search by query text...",
        label_visibility="collapsed",
        key="hist_search_input",
    )

    # Filter row
    sources = [("Global", "Global Insight", "g"), ("Local", "Local Insight", "l"), ("Web", "Web Insight", "w")]
    f_cols = st.columns([1, 1, 1, 1.5, 1.5])

    for i, (label, val, c_key) in enumerate(sources):
        is_active = val in st.session_state.hist_filters
        active_cls = f"active-{c_key}" if is_active else ""
        with f_cols[i]:
            st.markdown(f"<div class='filter-chip {active_cls}'>", unsafe_allow_html=True)
            if st.button(f"{'●' if is_active else '○'} {label}", key=f"filter_{label}", use_container_width=True):
                if val in st.session_state.hist_filters:
                    if len(st.session_state.hist_filters) > 1:
                        st.session_state.hist_filters.remove(val)
                else:
                    st.session_state.hist_filters.append(val)
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    with f_cols[3]:
        st.markdown("<div class='sort-pill'>", unsafe_allow_html=True)
        sort_lbl = "Newest first ↓" if st.session_state.hist_sort_newest else "Oldest first ↑"
        if st.button(sort_lbl, use_container_width=True, key="sort_toggle"):
            st.session_state.hist_sort_newest = not st.session_state.hist_sort_newest
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with f_cols[4]:
        st.markdown("<div class='clear-pill'>", unsafe_allow_html=True)
        if not st.session_state.hist_confirm_clear:
            if st.button("🗑 Clear history", use_container_width=True, key="clear_hist"):
                st.session_state.hist_confirm_clear = True
                st.rerun()
        else:
            cc1, cc2 = st.columns(2)
            if cc1.button("✅ Yes", use_container_width=True, key="clear_yes"):
                st.session_state.history = []
                clear_history()
                st.session_state.hist_confirm_clear = False
                st.rerun()
            if cc2.button("❌ No", use_container_width=True, key="clear_no"):
                st.session_state.hist_confirm_clear = False
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Filtering ──────────────────────────────────────────────────────────────
    filtered = [
        (idx, item) for idx, item in enumerate(st.session_state.history)
        if st.session_state.hist_search.lower() in item["query"].lower()
        and item["insight"] in st.session_state.hist_filters
    ]
    display_list = list(reversed(filtered)) if st.session_state.hist_sort_newest else filtered

    # ── Source style map ───────────────────────────────────────────────────────
    SOURCE_STYLES = {
        "Global Insight": ("g", "arXiv Global"),
        "Local Insight":  ("l", "Local Document"),
        "Web Insight":    ("w", "Live Web"),
    }

    def _rel_time(ts_str: str) -> str:
        """Convert HH:MM string or ISO string to relative label."""
        try:
            now = datetime.datetime.now()
            # Try ISO first
            try:
                dt = datetime.datetime.fromisoformat(ts_str)
            except Exception:
                # Fall back to today + HH:MM
                dt = datetime.datetime.strptime(ts_str, "%H:%M").replace(
                    year=now.year, month=now.month, day=now.day
                )
            diff = now - dt
            if diff.total_seconds() < 60:   return "just now"
            if diff.total_seconds() < 3600: return f"{int(diff.total_seconds()//60)}m ago"
            if diff.days == 0:              return f"{int(diff.total_seconds()//3600)}h ago"
            if diff.days == 1:              return "Yesterday"
            if diff.days < 7:               return f"{diff.days}d ago"
            return dt.strftime("%b %d")
        except Exception:
            return ts_str  # fallback: show raw ts

    # ── Date grouping ──────────────────────────────────────────────────────────
    def _group(entries):
        now   = datetime.datetime.now()
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        yest  = today - datetime.timedelta(days=1)
        week  = today - datetime.timedelta(days=7)
        groups = {"Today": [], "Yesterday": [], "Last 7 Days": [], "Earlier": []}
        for idx, item in entries:
            ts = item.get("iso_ts") or item.get("ts", "")
            try:
                try:   dt = datetime.datetime.fromisoformat(ts)
                except: dt = datetime.datetime.strptime(ts, "%H:%M").replace(year=now.year, month=now.month, day=now.day)
            except: dt = today
            if dt >= today:    groups["Today"].append((idx, item))
            elif dt >= yest:   groups["Yesterday"].append((idx, item))
            elif dt >= week:   groups["Last 7 Days"].append((idx, item))
            else:              groups["Earlier"].append((idx, item))
        return groups

    # ── Render ─────────────────────────────────────────────────────────────────
    if not st.session_state.history:
        st.markdown("""
<div class='hist-empty'>
    <div style='font-size:40px;margin-bottom:12px;'>🗂️</div>
    <div style='font-size:17px;font-weight:700;color:var(--t2);margin-bottom:4px;'>Nothing here yet</div>
    <div style='font-size:14px;color:var(--t3);'>Your research queries will appear here</div>
</div>""", unsafe_allow_html=True)

    elif not display_list:
        st.markdown("""
<div class='hist-empty'>
    <div style='font-size:14px;color:var(--t3);'>No results match your search.</div>
</div>""", unsafe_allow_html=True)

    else:
        SOURCE_COLORS = {
            "Global Insight": ("#28A745", "rgba(40,167,69,0.1)"),   # Green
            "Local Insight":  ("#FF9800", "rgba(255,152,0,0.1)"),    # Orange
            "Web Insight":    ("#00BCD4", "rgba(0,188,212,0.1)"),    # Cyan
        }

        for orig_idx, item in display_list:
            b_cls = SOURCE_STYLES.get(item["insight"], ("g", "Global"))[0]
            b_lbl = SOURCE_STYLES.get(item["insight"], ("g", "Global"))[1]
            q_trunc = item["query"][:65] + "…" if len(item["query"]) > 65 else item["query"]

            # Single unified row: [card] [↗] [🗑]
            st.markdown("<div class='hist-row-wrap'>", unsafe_allow_html=True)
            col_query, col_open, col_del = st.columns([11, 0.7, 0.7])

            with col_query:
                source_colors_map = {
                    "g": ("#10b981", "rgba(16,185,129,0.10)"),
                    "l": ("#f59e0b", "rgba(245,158,11,0.10)"),
                    "w": ("#0891b2", "rgba(8,145,178,0.10)"),
                }
                hex_c, bg_c = source_colors_map.get(b_cls, ("#6e56ff", "rgba(110,86,255,0.1)"))
                st.markdown(f"""
                <div style='display:flex; align-items:center; gap:12px;
                            background:var(--card-bg); border:1px solid var(--border);
                            border-left: 3px solid {hex_c};
                            border-radius:10px; padding:14px 18px;
                            box-shadow:0 1px 3px rgba(0,0,0,0.04); transition: all 0.3s ease-in-out;'>
                    <span style='background:{bg_c}; color:{hex_c}; font-size:10px;
                                 font-weight:700; text-transform:uppercase;
                                 letter-spacing:0.5px; padding:3px 9px;
                                 border-radius:5px; white-space:nowrap;'>{b_lbl}</span>
                    <span style='font-size:14px; font-weight:500;
                                 color:var(--t1);'>{q_trunc}</span>
                </div>""", unsafe_allow_html=True)

            with col_open:
                st.markdown("<div class='hist-action-btn'>", unsafe_allow_html=True)
                if st.button("↗", key=f"open_{orig_idx}", use_container_width=True, help="Open report"):
                    st.session_state.insight = item["insight"]
                    st.session_state.view = "research"
                    st.session_state.came_from_history = True
                    st.session_state.restore_query = item["query"]
                    st.session_state.last_plan[item["insight"]] = {
                        "report": item.get("report", ""),
                        "query":  item.get("query", ""),
                        "ts":     item.get("ts", ""),
                        "iso_ts": item.get("iso_ts", ""),
                        "restored": True,
                    }
                    # Restore follow-up results if they exist for this session
                    source_key = INSIGHT_SOURCE_MAP.get(item["insight"])
                    if "followups" in item:
                        st.session_state.followup_results[source_key] = item["followups"]
                    else:
                        st.session_state.followup_results[source_key] = []
                    
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

            with col_del:
                st.markdown("<div class='hist-del-btn'>", unsafe_allow_html=True)
                if st.button("🗑", key=f"del_{orig_idx}", use_container_width=True, help="Delete"):
                    remove_entry(orig_idx)
                    st.session_state.history.pop(orig_idx)
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)



    st.markdown("</div>", unsafe_allow_html=True)  # .hist-page



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
    st.markdown("""
    <div style='padding:64px 60px 0;'>
        <div class='page-heading fi'>
            Uncover insights.<br>
            <span>What's your question?</span>
        </div>
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
                    st.success(f"**{uploaded_file.name}** — {len(chunks)} chunks ready.")

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
            st.warning("Please upload a document first.")
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
                res["query"] = final_q   # ← ADD THIS LINE before storing
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
                
                # Sync metadata back to the result so follow-ups can find the entry
                res["iso_ts"] = new_entry.get("iso_ts")
                res["ts"] = new_entry.get("ts")
                res["query"] = final_q

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
            label="Download Report as PDF",
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