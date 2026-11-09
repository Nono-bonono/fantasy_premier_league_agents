"""
fpl.py - pulls every player's season stats from the official FPL API into ONE table.

Endpoint: https://fantasy.premierleague.com/api/bootstrap-static/
It's free, has no bot protection, and returns every player ("elements") plus lookup
tables for teams and positions. One request = the whole league.

USAGE:
    python fpl.py
Writes (relative to the repo root, two folders above this file):
    data/raw/fpl/bootstrap_<date>.json   raw response, for debugging
    data/processed/fpl_players.csv       the one table, latest snapshot
    data/processed/fpl_players_<date>.csv  dated copy, so you can track form over time

INSTALL: pip install requests pandas
"""

import json
import os
from datetime import date

import pandas as pd
import requests

URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
HEADERS = {"User-Agent": "Mozilla/5.0 (fpl-agents project)"}

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "fpl")
OUT_DIR = os.path.join(BASE_DIR, "data", "processed")

# Columns that matter for modelling, in a sensible order. Anything the API
# doesn't return is skipped, so a renamed field won't crash the script.
KEEP = [
    "id", "web_name", "first_name", "second_name", "team", "position",
    "price", "status", "chance_of_playing_next_round", "news",
    "minutes", "starts", "total_points", "points_per_game", "form", "ep_next",
    "goals_scored", "assists", "clean_sheets", "goals_conceded",
    "yellow_cards", "red_cards", "saves", "bonus", "bps",
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "influence", "creativity", "threat", "ict_index",
    "selected_by_percent", "transfers_in_event", "transfers_out_event",
]


def fetch():
    r = requests.get(URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def build_table(data):
    players = pd.DataFrame(data["elements"])
    teams = pd.DataFrame(data["teams"])[["id", "name"]].rename(columns={"id": "team", "name": "team_name"})
    positions = pd.DataFrame(data["element_types"])[["id", "singular_name_short"]].rename(
        columns={"id": "element_type", "singular_name_short": "position"}
    )

    df = players.merge(teams, on="team", how="left").merge(positions, on="element_type", how="left")
    df["team"] = df["team_name"]  # replace team id with the name
    df["price"] = df["now_cost"] / 10  # API gives price in tenths of £m

    df = df[[c for c in KEEP if c in df.columns]].copy()

    # API returns many numeric fields as strings
    text_cols = {"web_name", "first_name", "second_name", "team", "position", "status", "news"}
    for c in df.columns:
        if c not in text_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.sort_values("total_points", ascending=False).reset_index(drop=True)


def main():
    today = date.today().isoformat()
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    data = fetch()
    with open(os.path.join(RAW_DIR, f"bootstrap_{today}.json"), "w") as f:
        json.dump(data, f)

    df = build_table(data)
    df.to_csv(os.path.join(OUT_DIR, "fpl_players.csv"), index=False)
    df.to_csv(os.path.join(OUT_DIR, f"fpl_players_{today}.csv"), index=False)
    print(f"Wrote {len(df)} players x {df.shape[1]} columns -> {OUT_DIR}")
    print(df[["web_name", "team", "position", "price", "minutes", "total_points"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
