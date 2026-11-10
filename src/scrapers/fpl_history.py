"""
fpl_history.py

Pulls every player's gameweek-by-gameweek history from the official FPL API and builds
one table: a row per player per fixture (a double gameweek gives two rows).

Endpoints (free, no bot protection):
    bootstrap-static/        players, teams, gameweeks (1 call)
    element-summary/{id}/    one player's history (1 call per player, ~800 total)

Writes (relative to the repo root, two folders above this file):
    data/raw/fpl/history/<player_id>.json     raw response per player (the cache)
    data/raw/fpl/history/_manifest.json       last finished gameweek fetched per player
    data/processed/fpl_player_history.parquet the one table

HOW THE CACHE WORKS: a player is only re-fetched when the manifest says it was last
fetched before the most recent finished gameweek. The first run downloads everyone
(a few minutes); later runs only add the new gameweek. --refresh re-downloads everyone.

NOTES:
  - "history" in the API covers the CURRENT season only. Previous seasons come back as
    "history_past" and are saved in the raw JSON but not put in the table.
  - The team column is each player's CURRENT club, so a mid-season transfer will show
    the new club on old rows. Opponent names are correct for each row.

INSTALL: pip install requests pandas pyarrow

USAGE:
    python fpl_history.py                # fetch whatever is missing, then rebuild the table
    python fpl_history.py --refresh      # re-download every player
    python fpl_history.py --parse-only   # rebuild the table from saved JSON (one quick call
                                         # to bootstrap-static for names, none per player)
"""

import argparse
import json
import os
import time

import pandas as pd
import requests

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
SUMMARY_URL = "https://fantasy.premierleague.com/api/element-summary/{pid}/"
HEADERS = {"User-Agent": "Mozilla/5.0 (fpl-agents project)"}

REQUEST_DELAY_SECONDS = 0.25
MAX_RETRIES = 4

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "fpl", "history")
OUT_DIR = os.path.join(BASE_DIR, "data", "processed")
MANIFEST_PATH = os.path.join(RAW_DIR, "_manifest.json")

PRIORITY_COLUMNS = [
    "player_id", "web_name", "team", "position", "gameweek", "fixture_id",
    "opponent", "was_home", "minutes", "total_points",
]
TEXT_COLUMNS = ("web_name", "team", "position", "opponent", "kickoff_time")


def get_json(session, url):
    """GET with retry/backoff. Returns parsed JSON, or None if it never succeeded."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            print(f"  [warn] {url} -> HTTP {r.status_code} (attempt {attempt}/{MAX_RETRIES})")
        except requests.RequestException as e:
            print(f"  [warn] {url} -> {e} (attempt {attempt}/{MAX_RETRIES})")
        time.sleep(2 * attempt)  # back off; covers 429 rate limiting too
    print(f"  [error] giving up on {url}")
    return None


def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(manifest):
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f)


def latest_finished_gameweek(bootstrap):
    finished = [e["id"] for e in bootstrap.get("events", []) if e.get("finished")]
    return max(finished) if finished else 0


def fetch_histories(session, players, latest_gw, refresh):
    """Downloads raw history for every player that is missing or behind. Returns failures."""
    os.makedirs(RAW_DIR, exist_ok=True)
    manifest = load_manifest()

    todo = [
        p for p in players
        if refresh
        or manifest.get(str(p["id"]), -1) < latest_gw
        or not os.path.exists(os.path.join(RAW_DIR, f"{p['id']}.json"))
    ]
    print(f"{len(todo)} of {len(players)} players to fetch (latest finished gameweek: {latest_gw}).")

    failed = []
    try:
        for i, p in enumerate(todo, 1):
            data = get_json(session, SUMMARY_URL.format(pid=p["id"]))
            if data is None:
                failed.append(p["id"])
                continue
            with open(os.path.join(RAW_DIR, f"{p['id']}.json"), "w", encoding="utf-8") as f:
                json.dump(data, f)
            manifest[str(p["id"])] = latest_gw
            if i % 100 == 0:
                print(f"  fetched {i}/{len(todo)}")
            time.sleep(REQUEST_DELAY_SECONDS)
    finally:
        save_manifest(manifest)  # progress survives a crash or Ctrl+C

    if failed:
        print(f"{len(failed)} players failed - re-run to retry them: {failed[:10]}...")
    return failed


def build_table(bootstrap):
    teams = {t["id"]: t["name"] for t in bootstrap["teams"]}
    positions = {t["id"]: t["singular_name_short"] for t in bootstrap["element_types"]}
    info = {
        p["id"]: {
            "web_name": p["web_name"],
            "team": teams.get(p["team"]),
            "position": positions.get(p["element_type"]),
        }
        for p in bootstrap["elements"]
    }

    rows = []
    for pid, meta in info.items():
        path = os.path.join(RAW_DIR, f"{pid}.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for h in data.get("history", []):
            row = dict(h)
            row["player_id"] = pid
            row["gameweek"] = row.pop("round", None)
            row["fixture_id"] = row.pop("fixture", None)
            row["opponent"] = teams.get(row.pop("opponent_team", None))
            row["price"] = row["value"] / 10 if row.get("value") is not None else None
            row.pop("value", None)
            row.pop("element", None)
            row.update(meta)
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    for col in df.columns:
        if col in TEXT_COLUMNS:
            df[col] = df[col].astype("string")
            continue
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            df[col] = df[col].astype("string")  # mixed types would break parquet

    ordered = [c for c in PRIORITY_COLUMNS if c in df.columns]
    rest = sorted(set(df.columns) - set(ordered))
    return df[ordered + rest].sort_values(["gameweek", "player_id"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Fetch per-gameweek FPL player history.")
    parser.add_argument("--refresh", action="store_true", help="Re-download every player")
    parser.add_argument("--parse-only", action="store_true", help="Rebuild the table from saved JSON")
    args = parser.parse_args()

    session = requests.Session()
    bootstrap = get_json(session, BOOTSTRAP_URL)
    if bootstrap is None:
        raise SystemExit("Could not fetch bootstrap-static.")

    if not args.parse_only:
        fetch_histories(session, bootstrap["elements"], latest_finished_gameweek(bootstrap), args.refresh)

    df = build_table(bootstrap)
    if df.empty:
        print("No history rows to write yet.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "fpl_player_history.parquet")
    df.to_parquet(path, index=False)
    print(f"Wrote {len(df)} rows ({df['player_id'].nunique()} players, "
          f"gameweeks {int(df['gameweek'].min())}-{int(df['gameweek'].max())}) -> {path}")


if __name__ == "__main__":
    main()
