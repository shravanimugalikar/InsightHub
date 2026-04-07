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
    """Loads, appends one entry, saves."""
    history = load_history()
    history.append(entry)
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
