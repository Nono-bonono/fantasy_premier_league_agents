"""
fpl_fixtures.py - pulls the full season fixture list from the official FPL API.

Endpoints (free, no bot protection):
    bootstrap-static/   team names and gameweek info (1 call)
    fixtures/           every fixture of the season, played and upcoming (1 call)

Writes (relative to the repo root, two folders above this file):
    data/raw/fpl/fixtures_<date>.json            raw response, for debugging
    data/processed/fpl_fixtures.parquet          one row per fixture
    data/processed/fpl_team_fixtures.parquet     one row per team per fixture (home and away
                                                 views), the shape the models want:
                                                 "who does Arsenal play in GW11, home or away?"

Columns in fpl_team_fixtures.parquet:
    gameweek, fixture_id, team, opponent, was_home, kickoff_time, finished,
    difficulty (FPL's own 1-5 rating for that team), team_score, opponent_score
Fixtures with no gameweek yet (postponed / not scheduled) have a blank gameweek. A team with
two fixtures in a gameweek (double gameweek) gets two rows; a team with none is blank week.

Use it to predict a future gameweek: filter to that gameweek, join to players on team, and pass
player_id, opponent, was_home to models.baseline.predict().

INSTALL: pip install requests pandas pyarrow

USAGE:
    python fpl_fixtures.py
"""

import json
import os
from datetime import date

import pandas as pd
import requests

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
HEADERS = {"User-Agent": "Mozilla/5.0 (fpl-agents project)"}

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "fpl")
OUT_DIR = os.path.join(BASE_DIR, "data", "processed")


def fetch(url):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def build_fixtures(fixtures, teams):
    """One row per fixture, with team names instead of ids. teams = {id: name}."""
    df = pd.DataFrame(fixtures)
    if df.empty:
        return df
    out = pd.DataFrame({
        "fixture_id": df["id"],
        "gameweek": pd.to_numeric(df["event"], errors="coerce"),
        "kickoff_time": df["kickoff_time"].astype("string"),
        "home_team": df["team_h"].map(teams),
        "away_team": df["team_a"].map(teams),
        "finished": df["finished"].fillna(False).astype(bool),
        "home_score": pd.to_numeric(df["team_h_score"], errors="coerce"),
        "away_score": pd.to_numeric(df["team_a_score"], errors="coerce"),
        "home_difficulty": pd.to_numeric(df["team_h_difficulty"], errors="coerce"),
        "away_difficulty": pd.to_numeric(df["team_a_difficulty"], errors="coerce"),
    })
    return out.sort_values(["gameweek", "kickoff_time", "fixture_id"], na_position="last").reset_index(drop=True)


def build_team_fixtures(fx):
    """Splits each fixture into a home row and an away row (team, opponent, was_home)."""
    if fx.empty:
        return fx
    common = ["fixture_id", "gameweek", "kickoff_time", "finished"]
    home = fx[common].assign(
        team=fx["home_team"], opponent=fx["away_team"], was_home=True,
        difficulty=fx["home_difficulty"], team_score=fx["home_score"], opponent_score=fx["away_score"],
    )
    away = fx[common].assign(
        team=fx["away_team"], opponent=fx["home_team"], was_home=False,
        difficulty=fx["away_difficulty"], team_score=fx["away_score"], opponent_score=fx["home_score"],
    )
    cols = ["gameweek", "fixture_id", "team", "opponent", "was_home", "kickoff_time", "finished",
            "difficulty", "team_score", "opponent_score"]
    out = pd.concat([home, away], ignore_index=True)[cols]
    return out.sort_values(["gameweek", "team", "kickoff_time"], na_position="last").reset_index(drop=True)


def next_gameweek(bootstrap):
    """The gameweek FPL calls 'next' (the first one not yet finished or in progress)."""
    for e in bootstrap.get("events", []):
        if e.get("is_next"):
            return e["id"]
    return None


def main():
    today = date.today().isoformat()
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    bootstrap = fetch(BOOTSTRAP_URL)
    fixtures = fetch(FIXTURES_URL)
    with open(os.path.join(RAW_DIR, f"fixtures_{today}.json"), "w", encoding="utf-8") as f:
        json.dump(fixtures, f)

    teams = {t["id"]: t["name"] for t in bootstrap["teams"]}
    fx = build_fixtures(fixtures, teams)
    tf = build_team_fixtures(fx)
    fx.to_parquet(os.path.join(OUT_DIR, "fpl_fixtures.parquet"), index=False)
    tf.to_parquet(os.path.join(OUT_DIR, "fpl_team_fixtures.parquet"), index=False)

    unscheduled = int(fx["gameweek"].isna().sum())
    print(f"Wrote {len(fx)} fixtures ({int(fx['finished'].sum())} finished, {unscheduled} without a gameweek) "
          f"and {len(tf)} team-fixture rows -> {OUT_DIR}")

    nxt = next_gameweek(bootstrap)
    if nxt is not None:
        upcoming = fx[fx["gameweek"] == nxt]
        print(f"\nNext gameweek: {nxt} ({len(upcoming)} fixtures)")
        print(upcoming[["home_team", "away_team", "kickoff_time", "home_difficulty", "away_difficulty"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
