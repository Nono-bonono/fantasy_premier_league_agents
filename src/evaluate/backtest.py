"""
backtest.py - walk-forward test of the baseline ladder against the points players actually scored.

For each gameweek t (from MIN_HISTORY+1 on) the model sees ONLY gameweeks < t, predicts every
player who had a fixture in t, and is scored against their real total_points in t.

Metrics (what matters for picking players is ranking, not just average error):
  MAE / RMSE      average error over all players with a fixture
  Spearman        rank correlation between predicted and actual points, per GW then averaged,
                  among players with at least one prior appearance
  Top-N points    average actual points of the model's top N picks (N=15 overall, and top 3 per
                  position-slot style groups) vs the best possible N, and vs a random pick

NOT included yet: FPL's own ep_next. The API only exposes the CURRENT ep_next, so there is no
history to test against. predict_next.py (later) will log ep_next every week from now on.

LAST SEASON: --season 2025-26 runs the same test on data/processed/player_gw_2025-26.parquet
(from src/scrapers/fpl_last_season.py). That table also has fpl_xP, FPL's expected points as
captured by the archive, but only for some gameweeks; the models are compared with it on those
gameweeks only (a separate table at the end). CAUTION: fpl_xP looks like it was captured after the
gameweek started (it correlates 0.6 with actual points among players who played, far above what a
pre-match forecast manages), so it is NOT a fair benchmark. The weekly ep_next log is the fair one.

USAGE:
    python src/evaluate/backtest.py                     # this season
    python src/evaluate/backtest.py --season 2025-26    # last season
Writes data/backtest/backtest_by_gw[_<season>].csv and backtest_summary[_<season>].csv (gitignored).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from models.baseline import MODELS_WITH_TEAM as MODELS, predict  # noqa: E402
import models.team_strength as ts  # noqa: E402

BASE_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
OUT_DIR = os.path.join(BASE_DIR, "data", "backtest")

MIN_HISTORY = 3   # need this many prior gameweeks before the first prediction
TOP_N = 15
SEED = 0
FPL_XP = "fpl_xP"


def spearman(a, b):
    return pd.Series(a).rank().corr(pd.Series(b).rank())


def has_xp(gw_rows):
    """True if the archive captured fpl_xP for this gameweek (it is all zeros when it didn't)."""
    if FPL_XP not in gw_rows:
        return False
    played = gw_rows[gw_rows["minutes"] > 0]
    return len(played) > 0 and (played[FPL_XP] == 0).mean() < 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default=None, help="e.g. 2025-26 for last season; omit for this season")
    args = ap.parse_args()
    suffix = f"_{args.season}" if args.season else ""
    gw_path = os.path.join(PROC_DIR, f"player_gw{suffix}.parquet")
    if not os.path.exists(gw_path):
        raise SystemExit(f"Missing {gw_path} - run src/scrapers/fpl_last_season.py --season {args.season} first.")
    gw = pd.read_parquet(gw_path)
    last = int(gw["gameweek"].max())
    rng = np.random.default_rng(SEED)
    rows = []
    # team model: this season uses last season's ratings as a prior; last season has no earlier season, so no prior
    if args.season:
        tf, prior, carry = ts.last_season_team_fixtures(), None, 0.0
    else:
        tf, prior, carry = ts.load_team_fixtures(), ts.last_season_prior(), ts.CARRY

    for t in range(MIN_HISTORY + 1, last + 1):
        rows_t = gw[gw["gameweek"] == t]
        actual = rows_t[["player_id", "opponent", "was_home", "total_points", "minutes"]]
        tm = ts.fit(tf, t, prior=prior, carry=carry)
        team_preds = ts.predict_fixtures(tm, tf[tf.gameweek == t])
        pred = predict(gw, t, actual, team_preds=team_preds).merge(actual[["player_id", "total_points"]], on="player_id")
        models_t = list(MODELS)
        if has_xp(rows_t):
            pred = pred.merge(rows_t[["player_id", FPL_XP]], on="player_id", how="left")
            models_t.append(FPL_XP)
        elig = pred[pred["has_history"] & (pred["n_apps"] > 0)]
        best_n = elig.nlargest(TOP_N, "total_points")["total_points"].mean()
        rand_n = np.mean([elig.sample(TOP_N, random_state=int(rng.integers(1e9)))["total_points"].mean()
                          for _ in range(200)])
        for m in models_t:
            if m not in pred:
                continue
            err = pred[m] - pred["total_points"]
            top = elig.nlargest(TOP_N, m)["total_points"].mean()
            rows.append({
                "gameweek": t, "model": m, "n_players": len(pred),
                "mae": err.abs().mean(), "rmse": np.sqrt((err ** 2).mean()),
                "spearman": spearman(elig[m], elig["total_points"]),
                "top_n_actual": top, "top_n_best_possible": best_n, "top_n_random": rand_n,
            })

    by_gw = pd.DataFrame(rows)
    xp_gws = sorted(by_gw.loc[by_gw["model"] == FPL_XP, "gameweek"].unique())
    xp_table = None
    if xp_gws:
        xp_table = by_gw[by_gw["gameweek"].isin(xp_gws)].groupby("model")[
            ["mae", "rmse", "spearman", "top_n_actual"]].mean().reindex(MODELS + [FPL_XP])
        by_gw_models = by_gw[by_gw["model"] != FPL_XP]
    else:
        by_gw_models = by_gw
    summary = by_gw_models.groupby("model").agg(
        mae=("mae", "mean"), rmse=("rmse", "mean"), spearman=("spearman", "mean"),
        top_n_actual=("top_n_actual", "mean"), top_n_random=("top_n_random", "mean"),
        top_n_best_possible=("top_n_best_possible", "mean"),
    ).reindex(MODELS)
    # how many of the GWs did each model beat the previous rung on Spearman / top-N?
    wide_s = by_gw_models.pivot(index="gameweek", columns="model", values="spearman")[MODELS]
    wide_t = by_gw_models.pivot(index="gameweek", columns="model", values="top_n_actual")[MODELS]
    summary["gws_beating_prev_spearman"] = [np.nan] + [int((wide_s[b] > wide_s[a]).sum()) for a, b in zip(MODELS, MODELS[1:])]
    summary["gws_beating_prev_topn"] = [np.nan] + [int((wide_t[b] > wide_t[a]).sum()) for a, b in zip(MODELS, MODELS[1:])]

    os.makedirs(OUT_DIR, exist_ok=True)
    by_gw.to_csv(os.path.join(OUT_DIR, f"backtest_by_gw{suffix}.csv"), index=False)
    summary.to_csv(os.path.join(OUT_DIR, f"backtest_summary{suffix}.csv"))

    pd.set_option("display.width", 220)
    print(f"Season: {args.season or 'current'}")
    print(f"Walk-forward backtest, gameweeks {MIN_HISTORY + 1}-{last} ({last - MIN_HISTORY} GWs), top-{TOP_N} picks\n")
    print(summary.round(3).to_string())
    print("\nSpearman by gameweek:")
    print(wide_s.round(3).to_string())
    print("\nTop-N actual points by gameweek:")
    print(wide_t.round(2).to_string())
    if xp_table is not None:
        print(f"\nModels vs fpl_xP on the {len(xp_gws)} GWs where the archive has it: {[int(g) for g in xp_gws]}")
        print("CAUTION: fpl_xP seems to have been captured after kickoff, so it likely knows too much. Not a fair benchmark.")
        print(xp_table.round(3).to_string())


if __name__ == "__main__":
    main()
