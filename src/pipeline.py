"""
pipeline.py - small helpers shared by friday.py (and, later, tuesday.py).

    refresh()      run the four data steps (fpl.py, fpl_fixtures.py, fpl_history.py, ingest.py), one after
                   another, each in its own process. A failed step never stops the run - it comes back
                   as ok=False so the report can say the data is stale.
    run_script()   run any script in src/ and capture the outcome.
    data_warnings() plain-English warnings if the data on disk looks old.
    gw_deadline()  the deadline of a gameweek from the FPL API (None if it can't be fetched).
    default_team() the FPL team ID used last (data/recommendations/teams.csv).
    llm_text()     one plain OpenRouter call (no web search) for the written summary.
"""

import os
import subprocess
import sys
import time
from datetime import date, datetime

import pandas as pd

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)

PROC = os.path.join(ROOT, "data", "processed")
REPORTS_DIR = os.path.join(ROOT, "data", "reports")
TEAMS_CSV = os.path.join(ROOT, "data", "recommendations", "teams.csv")
API = "https://fantasy.premierleague.com/api/"
HEADERS = {"User-Agent": "Mozilla/5.0 (fpl-agents project)"}

# (label, script under src/, extra args). Order matters: ingest reads what the scrapers write.
REFRESH_STEPS = [
    ("FPL players and prices", "scrapers/fpl.py", []),
    ("Fixtures", "scrapers/fpl_fixtures.py", []),
    ("Player history", "scrapers/fpl_history.py", []),
    ("Combine into player_gw", "ingest.py", []),
]


def run_script(rel, args=(), timeout=1200):
    """Run src/<rel> with the same Python as this process. Returns dict(step, ok, seconds, tail)."""
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, os.path.join(SRC, rel), *map(str, args)], cwd=ROOT,
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        ok = p.returncode == 0
        tail = "\n".join((p.stdout + "\n" + p.stderr).strip().splitlines()[-6:])
    except subprocess.TimeoutExpired:
        ok, tail = False, f"timed out after {timeout}s"
    except Exception as e:  # noqa: BLE001 - report it, don't crash the run
        ok, tail = False, f"{type(e).__name__}: {e}"
    return dict(step=rel, ok=ok, seconds=round(time.time() - t0, 1), tail=tail)


def refresh(log=print, steps=REFRESH_STEPS):
    results = []
    for label, rel, args in steps:
        log(f"  {label} ...", flush=True)
        r = run_script(rel, args)
        r["label"] = label
        results.append(r)
        log(f"    {'ok' if r['ok'] else 'FAILED'} ({r['seconds']}s)" + ("" if r["ok"] else "\n      " + r["tail"].replace("\n", "\n      ")))
    return results


def data_warnings(refresh_results=None, today=None):
    """Things that make today's recommendation less trustworthy. Empty list = data looks current."""
    today = today or date.today()
    out = [f"{r['label']} failed - using the data already on disk ({r['tail'].splitlines()[-1] if r['tail'] else 'no detail'})"
           for r in (refresh_results or []) if not r["ok"]]
    for fname, what in (("fpl_players.parquet", "FPL prices and injury flags"), ("player_gw.parquet", "the player history table")):
        path = os.path.join(PROC, fname)
        if not os.path.exists(path):
            out.append(f"{fname} is missing")
            continue
        d = datetime.fromtimestamp(os.path.getmtime(path)).date()
        if d != today:
            out.append(f"{what}: last refreshed on {d}")
    return out


def gw_deadline(gw):
    """Deadline of gameweek gw as an aware datetime, or None if the FPL API can't be reached."""
    try:
        import requests
        r = requests.get(API + "bootstrap-static/", headers=HEADERS, timeout=30)
        r.raise_for_status()
        for e in r.json()["events"]:
            if e["id"] == gw:
                return datetime.fromisoformat(e["deadline_time"].replace("Z", "+00:00"))
    except Exception:  # noqa: BLE001
        pass
    return None


def api_get(path):
    """GET an FPL API path and return the JSON (raises on failure)."""
    import requests
    r = requests.get(API + path, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def default_team():
    """(team_id, team_name) used most recently, or None."""
    if not os.path.exists(TEAMS_CSV):
        return None
    t = pd.read_csv(TEAMS_CSV)
    if not len(t):
        return None
    t = t.sort_values("last_used", ascending=False, kind="stable")   # ties keep the file order (newest first)
    return int(t.iloc[0].team_id), str(t.iloc[0].team_name)


def llm_text(system, user, model, max_tokens=400):
    """One OpenRouter chat call. Returns (text, cost) or (None, error message). Never raises."""
    try:
        from agents import news_agent
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            return None, "OPENROUTER_API_KEY not set"
        body = {"model": model, "temperature": 0, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if not model.startswith("anthropic/"):
            body["reasoning"] = {"effort": "low"}
        resp = news_agent.call_model(body, key, timeout=90)
        text = (resp["choices"][0]["message"].get("content") or "").strip()
        if not text:
            return None, "empty reply"
        return text, (resp.get("usage") or {}).get("cost") or 0.0
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"
