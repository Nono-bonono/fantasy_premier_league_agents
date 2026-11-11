"""
compare_ep_next.py - after a gameweek has been played, who predicted it better: our model or
FPL's own ep_next?

INPUTS:
    data/predictions/baseline_gw<N>_<date>.parquet   written by predict_next.py BEFORE the GW
    data/processed/player_gw.parquet                 ingest.py output, must now contain GW N

Run order after a gameweek finishes:  fpl_history.py -> ingest.py -> this script.

WHAT IT COMPARES (same players, same gameweek, against real total_points):
    xpts         our headline number (model x availability)
    m3 / m4      our model rungs without the availability adjustment
    ep_next      FPL's expected points
  Metrics: MAE and RMSE over all players with a fixture; Spearman and top-N average actual
  points among players our model thinks can play (p_appear > 0). Top-N is the one that matters
  for picking a team.

GUARDS:
  - A prediction file is only used if it was made on or before the day of the gameweek's first
    kickoff. Later files could contain ep_next that already knows the result; they are skipped
    unless --include-late is given.
  - If several files exist for one gameweek, the latest valid one is used.
  - Gameweeks not yet in player_gw.parquet are skipped.
  - One gameweek is mostly noise. The bootstrap interval on the error difference says whether
    a gap is bigger than that noise; with a few GWs it usually is not. Keep logging.

USAGE:
    python src/evaluate/compare_ep_next.py
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
GW_PATH = os.path.join(BASE_DIR, "data", "processed", "player_gw.parquet")
PRED_DIR = os.path.join(BASE_DIR, "data", "predictions")
OUT_DIR = os.path.join(BASE_DIR, "data", "backtest")

COMPARE = {"xpts": "xpts (ours)", "m3_availability_x_ppa": "m3", "m4_with_fixture": "m4", "ep_next": "ep_next (FPL)"}
TOP_N = 15
N_BOOT = 2000
SEED = 0
NAME_RE = re.compile(r"baseline_gw(\d+)_(\d{4}-\d{2}-\d{2})\.parquet$")


def find_prediction_files(pred_dir, gw_table, include_late):
    """Latest valid prediction file per gameweek -> {gw: (path, date)}."""
    kickoff = pd.to_datetime(gw_table["kickoff_time"], utc=True, errors="coerce")
    first_kickoff = kickoff.groupby(gw_table["gameweek"]).min().dt.date
    chosen = {}
    for path in sorted(glob.glob(os.path.join(pred_dir, "baseline_gw*_*.parquet"))):
        m = NAME_RE.search(path)
        if not m:
            continue
        gw, made = int(m.group(1)), pd.Timestamp(m.group(2)).date()
        if gw not in first_kickoff.index:
            print(f"GW{gw}: not in player_gw.parquet yet - skipped ({os.path.basename(path)})")
            continue
        if made > first_kickoff[gw] and not include_late:
            print(f"[skip] {os.path.basename(path)} was made after GW{gw} kicked off ({first_kickoff[gw]}); "
                  f"its ep_next may already know the result.")
            continue
        if gw not in chosen or made > chosen[gw][1]:
            chosen[gw] = (path, made)
    return chosen


def spearman(a, b):
    return pd.Series(a).rank().corr(pd.Series(b).rank())


def score_gameweek(pred, gw_table, gw):
    actual = gw_table[gw_table["gameweek"] == gw][["player_id", "total_points"]]
    df = pred.merge(actual, on="player_id", how="inner")
    elig = df[df["p_appear"] > 0]
    rows = []
    for col, label in COMPARE.items():
        if col not in df:
            continue
        d = df.dropna(subset=[col])
        err = d[col] - d["total_points"]
        e = elig.dropna(subset=[col])
        rows.append({
            "gameweek": gw, "model": label, "n_players": len(d),
            "mae": err.abs().mean(), "rmse": np.sqrt((err ** 2).mean()),
            "spearman": spearman(e[col], e["total_points"]),
            "top_n_actual": e.nlargest(TOP_N, col)["total_points"].mean(),
        })
    return pd.DataFrame(rows), df


def bootstrap_gap(frames, col_a="xpts", col_b="ep_next"):
    """Mean(|err_a| - |err_b|) pooled over gameweeks, with a 95% bootstrap interval over players.
    Negative = our model has the smaller error."""
    pooled = pd.concat(frames, ignore_index=True).dropna(subset=[col_a, col_b])
    diff = (pooled[col_a] - pooled["total_points"]).abs().to_numpy() - (pooled[col_b] - pooled["total_points"]).abs().to_numpy()
    rng = np.random.default_rng(SEED)
    boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(N_BOOT)]
    return diff.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5), len(diff)


def main():
    ap = argparse.ArgumentParser(description="Compare our predictions with FPL's ep_next on played gameweeks.")
    ap.add_argument("--pred-dir", default=PRED_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--include-late", action="store_true", help="use predictions made after kickoff (not trustworthy)")
    args = ap.parse_args()

    if not os.path.exists(GW_PATH):
        raise SystemExit(f"Missing {GW_PATH} - run ingest.py first.")
    gw_table = pd.read_parquet(GW_PATH)

    files = find_prediction_files(args.pred_dir, gw_table, args.include_late)
    if not files:
        raise SystemExit("Nothing to compare yet: need a baseline_gw<N>_<date>.parquet from predict_next.py "
                         "for a gameweek that is now in player_gw.parquet.")

    by_gw, frames = [], []
    for gw, (path, made) in sorted(files.items()):
        scores, df = score_gameweek(pd.read_parquet(path), gw_table, gw)
        by_gw.append(scores)
        frames.append(df)
        print(f"GW{gw}: {len(df)} players scored from {os.path.basename(path)}")
    by_gw = pd.concat(by_gw, ignore_index=True)

    order = list(COMPARE.values())
    summary = by_gw.groupby("model")[["mae", "rmse", "spearman", "top_n_actual"]].mean().reindex(order).dropna(how="all")

    os.makedirs(args.out_dir, exist_ok=True)
    by_gw.to_csv(os.path.join(args.out_dir, "ep_next_comparison_by_gw.csv"), index=False)
    summary.to_csv(os.path.join(args.out_dir, "ep_next_comparison_summary.csv"))

    pd.set_option("display.width", 200)
    print(f"\nGameweeks compared: {sorted(files)}   (top-{TOP_N} picks, mean over gameweeks)\n")
    print(summary.round(3).to_string())
    if len(by_gw["gameweek"].unique()) > 1:
        print("\nTop-N actual points by gameweek:")
        print(by_gw.pivot(index="gameweek", columns="model", values="top_n_actual")[summary.index].round(2).to_string())

    mean, lo, hi, n = bootstrap_gap(frames)
    verdict = "ours smaller" if hi < 0 else "ep_next smaller" if lo > 0 else "no clear difference"
    print(f"\nAbsolute-error gap, xpts minus ep_next: {mean:+.3f} pts per player "
          f"(95% interval {lo:+.3f} to {hi:+.3f}, {n} player-gameweeks) -> {verdict}")


if __name__ == "__main__":
    main()
