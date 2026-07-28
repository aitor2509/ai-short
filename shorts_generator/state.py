"""JSON-based queue/state management for multi-video batch processing.

Stores the processing queue and per-video results as JSON files so
workflows (GitHub Actions or local) can pass state between runs without
a database.

State file structure (state.json):
{
  "queue": [
    {"url": "https://...", "account": "cuenta1", "status": "pending", "niche": "comedy"},
    {"url": "https://...", "account": "cuenta2", "status": "done", "clips": 45}
  ],
  "accounts": [
    {"name": "cuenta1", "facebook_page_id": "...", "fb_token": "...",
     "sources": ["@channel1", "channel2"], "niche": "comedy"}
  ],
  "stats": {"total_processed": 0, "total_clips": 0}
}
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Optional


STATE_FILE = "state.json"


def load_state(path: str = STATE_FILE) -> Dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"queue": [], "accounts": [], "stats": {"total_processed": 0, "total_clips": 0}}


def save_state(state: Dict, path: str = STATE_FILE):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def add_to_queue(url: str, account: str = "", niche: str = "", state: Optional[Dict] = None) -> Dict:
    if state is None:
        state = load_state()
    state["queue"].append({
        "url": url,
        "account": account,
        "niche": niche,
        "status": "pending",
        "added_at": datetime.utcnow().isoformat(),
    })
    save_state(state)
    return state


def next_pending(state: Optional[Dict] = None) -> Optional[Dict]:
    if state is None:
        state = load_state()
    for item in state["queue"]:
        if item["status"] == "pending":
            return item
    return None


def mark_done(url: str, clip_count: int = 0, state: Optional[Dict] = None):
    if state is None:
        state = load_state()
    for item in state["queue"]:
        if item["url"] == url and item["status"] == "pending":
            item["status"] = "done"
            item["clips"] = clip_count
            item["done_at"] = datetime.utcnow().isoformat()
            break
    state["stats"]["total_processed"] = sum(1 for q in state["queue"] if q["status"] == "done")
    state["stats"]["total_clips"] += clip_count
    save_state(state)


def mark_failed(url: str, error: str, state: Optional[Dict] = None):
    if state is None:
        state = load_state()
    for item in state["queue"]:
        if item["url"] == url and item["status"] == "pending":
            item["status"] = "failed"
            item["error"] = error
            break
    save_state(state)
