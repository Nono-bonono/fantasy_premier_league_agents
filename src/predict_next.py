"""
predict_next.py - expected points for every player in the next gameweek, saved next to FPL's
own ep_next so the two can be compared once the gameweek has been played.

INPUTS (run the scrapers/ingest first):
    data/processed/player_gw.parquet          ingest.py          history the model learns from
    data/processed/fpl_players.parquet        fpl.py             current team, ep_next, injury flags
    data/processed/fpl_team_fixtures.parquet  fpl_fixtures.py    who each team plays in each GW

OUTPUT:
    data/predictions/baseline_gw<N>_<date>.parquet and .csv   (gitignored)
    Columns: player, team, position, price, opponent, was_home, the model rungs, avail, xpts,
    ep_next. xpts = chosen model x avail. Keep these files: after the GW is played, join to the
    real points to see whether the model or FPL's ep_next did better.

HOW IT HANDLES THE AWKWARD CASES
  - Double gameweek: each fixture is predicted separately and the points are added up.
  - Blank gameweek (team has no fixture): the player is left out of the table.
  - Availability: FPL's chance_of_playing_next_round / 100 is multiplied in (blank = 100%).
    This is the hook the news agent will eventually feed.
  - No history yet (new signing): xpts is 0 until he has appeared.

MODEL CHOICE: --model defaults to m3 (chance of appearing x points per appearance). The fixture
rung m4 did NOT beat m3 in the backtest (7 GWs, too little data), so it is shown but not the
default. Switch with --model m4_with_fixture once the team-strength model replaces it.

USAGE:
    python src/predict_next.py                    # next GW = last played GW + 1
    python src/predict_next.py --gw 12 --top 30
"""

import argparse
import os
import sys
from datetime import date, datetime

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from models.baseline import MODELS, predict  # noqa: E402

BASE_DIR = os.path.abspath(os.path.join(HERE, ".."))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
GW_PATH = os.path.join(PROC_DIR, "player_gw.parquet")
PLAYERS_PATH = os.path.join(PROC_DIR, "fpl_players.parquet")
FIXTURES_PATH = os.path.join(PROC_DIR, "fpl_team_fixtures.parquet")
OUT_DIR = os.path.join(BASE_DIR, "data", "predictions")

DEFAULT_MODEL = "m3_availability_x_ppa"
SUM_COLS = ["m3_availability_x_ppa", "m4_with_fixture"]


def player_fixtures(players, team_fixtures, gw):
    """One row per player per fixture in gw: player_id, opponent, was_home."""
    tf = team_fixtures[team_fixtures["gameweek"] == gw][["team", "opponent", "was_home", "kickoff_time"]]
    tf = tf.sort_values(["team", "kickoff_time"])
    out = players[["id", "team"]].rename(columns={"id": "player_id"}).merge(tf, on="team", how="inner")
    out["fixture_no"] = out.groupby("player_id").cumcount()
    return out


def predict_gameweek(player_gw, players, team_fixtures, gw, model=DEFAULT_MODEL):
    fx = player_fixtures(players, team_fixtures, gw)
    if fx.empty:
        raise SystemExit(f"No fixtures found for gameweek {gw} - run fpl_fixtures.py (or check --gw).")

    # predict() takes one fixture per player, so run it once per fixture number and add up
    parts = []
    for n in sorted(fx["fixture_no"].unique()):
        p = predict(player_gw, gw, fx[fx["fixture_no"] == n][["player_id", "opponent", "was_home"]])
        parts.append(p)
    pred = pd.concat(parts, ignore_index=True)

    agg = pred.groupby("player_id").agg(
        **{c: (c, "sum") for c in SUM_COLS},
        opponent=("opponent", lambda s: " + ".join(s)),
        was_home=("was_home", lambda s: " + ".join("H" if h else "A" for h in s)),
        n_fixtures=("opponent", "size"),
        p_appear=("p_appear", "first"),
    ).reset_index()

    info = players.set_index("id")[["web_name", "team", "position", "price", "status",
                                    "chance_of_playing_next_round", "news", "ep_next"]]
    out = agg.join(info, on="player_id")
    out["avail"] = (out["chance_of_playing_next_round"].fillna(100) / 100).clip(0, 1)
    out["xpts"] = (out[model].fillna(0) * out["avail"]).round(3)
    out["gameweek"] = gw
    cols = ["gameweek", "player_id", "web_name", "team", "position", "price", "opponent", "was_home",
            "n_fixtures", "p_appear", "m3_availability_x_ppa", "m4_with_fixture", "avail", "xpts",
            "ep_next", "status", "news"]
    return out[cols].sort_values("xpts", ascending=False).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="Predict the next gameweek with the baseline model.")
    ap.add_argument("--gw", type=int, help="gameweek to predict (default: last played + 1)")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=MODELS[3:])
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--fixtures", default=FIXTURES_PATH)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    for path, hint in ((GW_PATH, "ingest.py"), (PLAYERS_PATH, "fpl.py"), (args.fixtures, "fpl_fixtures.py")):
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path} - run {hint} first.")

    player_gw = pd.read_parquet(GW_PATH)
    players = pd.read_parquet(PLAYERS_PATH)
    team_fixtures = pd.read_parquet(args.fixtures)
    gw = args.gw or int(player_gw["gameweek"].max()) + 1

    snap = datetime.fromtimestamp(os.path.getmtime(PLAYERS_PATH)).date()
    if snap != date.today():
        print(f"[warn] fpl_players.parquet is from {snap}, not today - ep_next and injury flags may be stale. "
              f"Run fpl.py first.")

    pred = predict_gameweek(player_gw, players, team_fixtures, gw, args.model)
    blank = players[~players["team"].isin(set(team_fixtures.loc[team_fixtures["gameweek"] == gw, "team"]))]
    if len(blank):
        print(f"{blank['team'].nunique()} team(s) have no fixture in GW{gw} and are left out: "
              f"{sorted(blank['team'].unique())}")

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, f"baseline_gw{gw}_{date.today().isoformat()}")
    pred.to_parquet(stem + ".parquet", index=False)
    pred.to_csv(stem + ".csv", index=False, encoding="utf-8")

    show = ["web_name", "team", "position", "price", "opponent", "was_home", "xpts", "ep_next", "avail"]
    pd.set_option("display.width", 200)
    print(f"\nGW{gw} - model {args.model}, saved to {stem}.parquet/.csv")
    print(f"\nTop {args.top} overall:")
    print(pred.head(args.top)[show].to_string(index=False))
    for pos in ("GKP", "DEF", "MID", "FWD"):
        print(f"\nTop 5 {pos}:")
        print(pred[pred["position"] == pos].head(5)[show].to_string(index=False))
    both = pred.dropna(subset=["ep_next"])
    print(f"\nRank correlation with FPL ep_next: {both['xpts'].rank().corr(both['ep_next'].rank()):.2f}")


if __name__ == "__main__":
    main()
