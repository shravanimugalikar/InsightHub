import os
import json
import datetime

HISTORY_FILE = "data/history/session_history.json"
OLD_HISTORY_FILE = "data/session_history.json"

# Move old file if it exists
if os.path.exists(OLD_HISTORY_FILE) and not os.path.exists(HISTORY_FILE):
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    os.rename(OLD_HISTORY_FILE, HISTORY_FILE)

def load_history() -> list:
    """Reads JSON file, returns [] if missing or corrupted."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []

def save_history(history: list):
    """Writes full list to JSON file."""
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def append_entry(entry: dict):
    """Loads, appends one entry with summary and tag generation, then saves."""
    history = load_history()
    
    # Generate summary if report exists and summary is missing
    if "report" in entry and ("summary" not in entry or not entry["summary"]):
        import re
        # Remove markdown headers and strong tags etc. for a clean snippet
        clean = re.sub(r'#+\s+', '', entry["report"])
        clean = re.sub(r'\*\*(.*?)\*\*', r'\1', clean)
        # Take first 150 chars
        summary = (clean[:147] + '...') if len(clean) > 150 else clean
        entry["summary"] = summary.strip()
    
    # Extract tags (LLM, RAG, Agents etc.)
    if "tags" not in entry:
        tags = []
        rep = entry.get("report", "").upper()
        if "RAG" in rep or "RETRIEV" in rep: tags.append("RAG")
        if "AGENT" in rep or "AUTONOM" in rep: tags.append("Agents")
        if "LLM" in rep or "GPT" in rep or "MODEL" in rep: tags.append("LLM")
        if "RESEARCH" in rep: tags.append("Research")
        entry["tags"] = list(set(tags))[:3] # Max 3 tags

    # Ensure default fields
    if "is_starred" not in entry:
        entry["is_starred"] = False
        
    if "followups" not in entry:
        entry["followups"] = []
        
    # Guarantee unique ISO timestamp for robust grouping
    if not entry.get("iso_ts"):
        entry["iso_ts"] = datetime.datetime.now().isoformat()
        
    history.append(entry)
    save_history(history)

def update_entry_followups(iso_ts: str, followups: list, query: str = None):
    """Updates the follow-ups list for an entry identified by iso_ts (or query fallback)."""
    history = load_history()
    updated = False
    
    # Pass 1: Try to match by ISO Timestamp (recommended)
    if iso_ts:
        for item in history:
            if item.get("iso_ts") == iso_ts:
                item["followups"] = followups
                updated = True
                break
    
    # Pass 2: Fallback to matching by Query text (case-insensitive)
    if not updated and query:
        for item in history:
            item_q = item.get("query", "").strip().lower()
            if item_q == query.strip().lower():
                item["followups"] = followups
                # Repair iso_ts if missing to prevent future fallback needs
                if not item.get("iso_ts") and iso_ts:
                    item["iso_ts"] = iso_ts
                updated = True
                break
    
    if updated:
        save_history(history)
    return updated

def toggle_star(index: int):
    """Flips is_starred state for an entry."""
    history = load_history()
    if 0 <= index < len(history):
        history[index]["is_starred"] = not history[index].get("is_starred", False)
        save_history(history)

def clear_history():
    """Writes [] to the file."""
    save_history([])

def remove_entry(index: int):
    """Loads, removes entry at index, saves."""
    history = load_history()
    if 0 <= index < len(history):
        history.pop(index)
        save_history(history)
