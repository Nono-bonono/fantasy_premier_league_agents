"""
fpl_last_season.py - last season's (2025-26) FPL gameweek data, in the same shape as player_gw.parquet.

The official FPL API only serves the CURRENT season gameweek by gameweek, so this reads the public
archive at github.com/vaastav/Fantasy-Premier-League (one row per player per fixture, all 38 GWs).

Downloads (cached; --refresh to re-download):
    data/raw/fpl/<season>/merged_gw.csv     player x fixture stats
    data/raw/fpl/<season>/players_raw.csv   player list (web_name, permanent `code`)
    data/raw/fpl/<season>/teams.csv         team id -> name for that season

Writes:
    data/processed/player_gw_<season>.parquet       one row per player per gameweek (double GWs summed)
    data/processed/team_fixtures_<season>.parquet   one row per team per fixture (for the team model)

HOW PLAYERS ARE LINKED TO THIS SEASON
  FPL player IDs are renumbered every season, but each player also has a permanent `code`.
  last-season id -> code (players_raw) -> this season's id (latest data/raw/fpl/bootstrap_*.json,
  written by fpl.py). Players who have since left the league get player_id = 100000 + last-season
  id, so they never clash with a current id. last_season_id keeps the original.

OTHER NOTES
  - Teams and opponents are names ("Man City", "Spurs"), as in this season's tables.
  - position "GK" is renamed "GKP" to match fpl.py.
  - price = value / 10 (FPL stores 5.5m as 55).
  - fpl_xP is the archive's xP column (FPL's expected points as captured by the archive). It is all
    zeros for most gameweeks, and where present it seems to have been captured after kickoff (it
    tracks actual points far too closely), so do not use it as a feature or a fair benchmark.
  - team is the club the player was at for that fixture.
  - The archive repeats a few player-fixture rows (e.g. Kroupi.Jr GW1-9); exact repeats are dropped
    so points aren't counted twice.
  - Double gameweeks: stats are summed, but last_season_id, team_h_score and team_a_score are IDs /
    match facts, so they keep the first fixture's value (summing them gave wrong IDs and scores).
  - Also writes data/processed/team_fixtures_<season>.parquet: one row per team per fixture, with
    both fixtures of a double gameweek (the player table can only hold one opponent per GW).

INSTALL: pip install requests pandas pyarrow

USAGE:
    python src/scrapers/fpl_last_season.py
    python src/scrapers/fpl_last_season.py --season 2024-25 --refresh
"""

import argparse
import glob
import json
import os
import sys

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))
from ingest import parquet_safe, to_gameweek  # noqa: E402

ARCHIVE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/{file}"
FILES = {"merged_gw.csv": "gws/merged_gw.csv", "players_raw.csv": "players_raw.csv", "teams.csv": "teams.csv"}
HEADERS = {"User-Agent": "Mozilla/5.0 (fpl-agents project)"}
UNMATCHED_OFFSET = 100000

# merged_gw column -> our column name (everything else keeps its archive name)
RENAME = {"element": "last_season_id", "GW": "gameweek", "fixture": "fixture_id", "xP": "fpl_xP"}
DROP = ["name", "round", "opponent_team", "value", "id", "code"]
FIRST = ["player_id", "last_season_id", "web_name", "team", "position", "gameweek", "fixture_id",
         "opponent", "was_home", "minutes", "total_points"]


def download(season, refresh):
    raw_dir = os.path.join(BASE_DIR, "data", "raw", "fpl", season)
    os.makedirs(raw_dir, exist_ok=True)
    for name, path in FILES.items():
        dest = os.path.join(raw_dir, name)
        if os.path.exists(dest) and not refresh:
            continue
        url = ARCHIVE.format(season=season, file=path)
        print(f"Downloading {url}")
        r = requests.get(url, headers=HEADERS, timeout=60)
        r.raise_for_status()
        with open(dest, "wb") as f:
            f.write(r.content)
    return raw_dir


def current_ids_by_code():
    files = sorted(glob.glob(os.path.join(BASE_DIR, "data", "raw", "fpl", "bootstrap_*.json")))
    if not files:
        raise SystemExit("No data/raw/fpl/bootstrap_*.json found - run src/scrapers/fpl.py first.")
    with open(files[-1], encoding="utf-8") as f:
        elements = json.load(f)["elements"]
    return {e["code"]: e["id"] for e in elements}


def build(raw_dir):
    gw = pd.read_csv(os.path.join(raw_dir, "merged_gw.csv"))
    n = len(gw)
    gw = gw.drop_duplicates(["element", "fixture"])
    if len(gw) < n:
        print(f"Dropped {n - len(gw)} repeated player-fixture rows from the archive.")
    players = pd.read_csv(os.path.join(raw_dir, "players_raw.csv"), usecols=["id", "code", "web_name"])
    teams = pd.read_csv(os.path.join(raw_dir, "teams.csv"), usecols=["id", "name"])

    gw = gw.merge(players, left_on="element", right_on="id", how="left")
    cur = current_ids_by_code()
    gw["player_id"] = gw["code"].map(cur)
    gw["player_id"] = gw["player_id"].fillna(UNMATCHED_OFFSET + gw["element"]).astype(int)

    gw["opponent"] = gw["opponent_team"].map(teams.set_index("id")["name"])
    gw["position"] = gw["position"].replace({"GK": "GKP"})
    gw["price"] = gw["value"] / 10
    gw["web_name"] = gw["web_name"].fillna(gw["name"])

    gw = gw.drop(columns=DROP, errors="ignore").rename(columns=RENAME)
    gw = gw[FIRST + [c for c in gw.columns if c not in FIRST]]
    gw = gw.sort_values(["gameweek", "kickoff_time", "fixture_id", "player_id"])
    gw = parquet_safe(gw, text_cols=("web_name", "team", "position", "opponent", "kickoff_time"))
    return gw


FIRST_FIXTURE_COLS = ["last_season_id", "team_h_score", "team_a_score"]


def keep_first_fixture(out, fixtures):
    """to_gameweek sums numeric columns in double GWs; IDs and match scores must not be summed."""
    key = ["player_id", "gameweek"]
    cols = [c for c in FIRST_FIXTURE_COLS if c in out.columns]
    firsts = fixtures.groupby(key, sort=False)[cols].first().reset_index()
    return out.drop(columns=cols).merge(firsts, on=key, how="left")[out.columns]


def team_fixtures(fixtures):
    """One row per team per fixture (both games of a double GW), same columns as fpl_team_fixtures."""
    d = fixtures.drop_duplicates(["fixture_id", "team"])
    h = d["was_home"].astype(bool)
    out = pd.DataFrame({
        "gameweek": d["gameweek"], "fixture_id": d["fixture_id"], "team": d["team"],
        "opponent": d["opponent"], "was_home": h, "kickoff_time": d["kickoff_time"], "finished": True,
        "team_score": d["team_h_score"].where(h, d["team_a_score"]).astype(float),
        "opponent_score": d["team_a_score"].where(h, d["team_h_score"]).astype(float)})
    return out.sort_values(["gameweek", "kickoff_time", "fixture_id", "team"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="2025-26", help="Archive season folder, e.g. 2025-26")
    ap.add_argument("--refresh", action="store_true", help="Re-download the archive files")
    args = ap.parse_args()

    raw_dir = download(args.season, args.refresh)
    fixtures = build(raw_dir)
    out = to_gameweek(fixtures)
    out = keep_first_fixture(out, fixtures)

    out_path = os.path.join(BASE_DIR, "data", "processed", f"player_gw_{args.season}.parquet")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_parquet(out_path, index=False)
    team_path = os.path.join(BASE_DIR, "data", "processed", f"team_fixtures_{args.season}.parquet")
    team_fixtures(fixtures).to_parquet(team_path, index=False)

    played = out[out["minutes"] > 0]
    linked = (played["player_id"] < UNMATCHED_OFFSET).mean()
    print(f"{args.season}: {len(fixtures)} player-fixture rows -> {len(out)} player-gameweek rows, "
          f"GW {int(out['gameweek'].min())}-{int(out['gameweek'].max())}, "
          f"{int((out['n_fixtures'] > 1).sum())} double-gameweek rows.")
    print(f"{linked:.0%} of played rows are players still in FPL this season (same player_id as now).")
    print(f"Wrote {out_path}")
    print(f"Wrote {team_path}")


if __name__ == "__main__":
    main()
