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

USAGE:
    python src/evaluate/backtest.py
Writes data/backtest/backtest_by_gw.csv and backtest_summary.csv (both gitignored).
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from models.baseline import MODELS, predict  # noqa: E402

BASE_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
GW_PATH = os.path.join(BASE_DIR, "data", "processed", "player_gw.parquet")
OUT_DIR = os.path.join(BASE_DIR, "data", "backtest")

MIN_HISTORY = 3   # need this many prior gameweeks before the first prediction
TOP_N = 15
SEED = 0


def spearman(a, b):
    return pd.Series(a).rank().corr(pd.Series(b).rank())


def main():
    gw = pd.read_parquet(GW_PATH)
    last = int(gw["gameweek"].max())
    rng = np.random.default_rng(SEED)
    rows = []

    for t in range(MIN_HISTORY + 1, last + 1):
        actual = gw[gw["gameweek"] == t][["player_id", "opponent", "was_home", "total_points", "minutes"]]
        pred = predict(gw, t, actual).merge(actual[["player_id", "total_points"]], on="player_id")
        elig = pred[pred["has_history"] & (pred["n_apps"] > 0)]
        best_n = elig.nlargest(TOP_N, "total_points")["total_points"].mean()
        rand_n = np.mean([elig.sample(TOP_N, random_state=int(rng.integers(1e9)))["total_points"].mean()
                          for _ in range(200)])
        for m in MODELS:
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
    summary = by_gw.groupby("model").agg(
        mae=("mae", "mean"), rmse=("rmse", "mean"), spearman=("spearman", "mean"),
        top_n_actual=("top_n_actual", "mean"), top_n_random=("top_n_random", "mean"),
        top_n_best_possible=("top_n_best_possible", "mean"),
    ).reindex(MODELS)
    # how many of the GWs did each model beat the previous rung on Spearman / top-N?
    wide_s = by_gw.pivot(index="gameweek", columns="model", values="spearman")[MODELS]
    wide_t = by_gw.pivot(index="gameweek", columns="model", values="top_n_actual")[MODELS]
    summary["gws_beating_prev_spearman"] = [np.nan] + [int((wide_s[b] > wide_s[a]).sum()) for a, b in zip(MODELS, MODELS[1:])]
    summary["gws_beating_prev_topn"] = [np.nan] + [int((wide_t[b] > wide_t[a]).sum()) for a, b in zip(MODELS, MODELS[1:])]

    os.makedirs(OUT_DIR, exist_ok=True)
    by_gw.to_csv(os.path.join(OUT_DIR, "backtest_by_gw.csv"), index=False)
    summary.to_csv(os.path.join(OUT_DIR, "backtest_summary.csv"))

    pd.set_option("display.width", 220)
    print(f"Walk-forward backtest, gameweeks {MIN_HISTORY + 1}-{last} ({last - MIN_HISTORY} GWs), top-{TOP_N} picks\n")
    print(summary.round(3).to_string())
    print("\nSpearman by gameweek:")
    print(wide_s.round(3).to_string())
    print("\nTop-N actual points by gameweek:")
    print(wide_t.round(2).to_string())


if __name__ == "__main__":
    main()
