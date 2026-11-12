"""
backtest_transfers.py - replays a season week by week with the transfer optimiser in charge,
to tune its settings (noise margin, value of a banked free transfer, horizon).

HOW THE REPLAY WORKS
  GW4: build a £100m squad from scratch (weeks 1-3 are needed as model history).
  Every later GW t: the models see only GWs < t, the optimiser plans t .. t+H-1, and its week-1
  moves are made. Then the GW is scored with the points players really got:
  XI + captain x2 (vice if the captain didn't play), automatic subs from the bench in order
  (formation must stay valid), minus 4 per hit. Selling prices follow FPL's half-the-rise rule.

  Compared against "no transfers": the same GW4 squad kept all season (XI and captain still
  picked each week), which is what the transfers have to beat.

CAVEATS
  - No injury news in the archive, so the replay has no chance-of-playing flags: it will sometimes
    buy or start injured players. Absolute scores are lower than live use; comparing settings is fair.
  - Fixtures are taken as finally played (postponements already moved), a small hindsight edge.
  - One season = about 35 transfer decisions. Pick a setting from a flat region, not the single best.

USAGE:
    python src/evaluate/backtest_transfers.py --season 2025-26 --grid      # full grid (slow-ish)
    python src/evaluate/backtest_transfers.py --margin 1.5 --ft-value 1 --horizon 3   # one setting, this season
Writes data/backtest/transfers_grid[_<season>].csv and transfers_by_gw[_<season>].csv.
"""

import argparse
import itertools
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, "optimise"))
from models import team_strength as ts  # noqa: E402
import transfers as T  # noqa: E402

ROOT = os.path.dirname(SRC)
PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "data", "backtest")
START_GW = 4
H_MAX = 5
VALID_MIN = {"DEF": 3, "MID": 2, "FWD": 1}


def load(season):
    if season:
        gw = pd.read_parquet(os.path.join(PROC, f"player_gw_{season}.parquet"))
        gw = gw.rename(columns={"player_id": "player_id_now"}).rename(columns={"last_season_id": "player_id"})
        return gw, ts.last_season_team_fixtures(), None, 0.0
    gw = pd.read_parquet(os.path.join(PROC, "player_gw.parquet"))
    return gw, ts.load_team_fixtures(), ts.last_season_prior(), ts.CARRY


def precompute(season, model="m5"):
    """Per GW: player info (position, club, price), expected points for the next H_MAX GWs, actual points."""
    gw, tf, prior, carry = load(season)
    last = int(gw.gameweek.max())
    weeks = {}
    for t in range(START_GW, last + 1):
        upto = gw[gw.gameweek <= t].sort_values(["gameweek", "kickoff_time"])
        latest = upto.groupby("player_id").tail(1)
        roster = latest[["player_id", "position", "team", "price", "web_name"]].copy()
        active = set(latest.loc[latest.gameweek >= t - 3, "player_id"])
        m = ts.fit(tf, t, prior=prior, carry=carry)
        up = tf[tf.gameweek.between(t, t + H_MAX - 1)]
        tp = ts.predict_fixtures(m, up)
        pts = T.project_points(gw, roster, up, tp, t, H_MAX, model)
        pts.loc[~pts.index.isin(active)] = 0.0
        act = gw[gw.gameweek == t].groupby("player_id")[["total_points", "minutes"]].sum()
        meta = roster.set_index("player_id").rename(columns={"price": "buy"})
        weeks[t] = dict(pts=pts, meta=meta, actual=act)
    return weeks


def valid(eleven, pos):
    c = pd.Series([pos[p] for p in eleven]).value_counts()
    return all(c.get(k, 0) >= v for k, v in VALID_MIN.items()) and c.get("GKP", 0) == 1


def score_week(step, pos, actual):
    pts = actual["total_points"].to_dict()
    mins = actual["minutes"].to_dict()
    played = lambda p: mins.get(p, 0) > 0  # noqa: E731
    eleven = list(step["xi"])
    bench = step["bench"]
    gk = [p for p in eleven if pos[p] == "GKP"][0]
    bench_gk = [p for p in bench if pos[p] == "GKP"]
    if not played(gk) and bench_gk and played(bench_gk[0]):
        eleven = [bench_gk[0] if p == gk else p for p in eleven]
    for b in [p for p in bench if pos[p] != "GKP"]:
        if not played(b):
            continue
        for r in [p for p in eleven if pos[p] != "GKP" and not played(p)]:
            new = [b if p == r else p for p in eleven]
            if valid(new, pos):
                eleven = new
                break
    total = sum(pts.get(p, 0) for p in eleven)
    cap, vice = step["captain"], step["vice"]
    if played(cap):
        total += pts.get(cap, 0)
    elif vice is not None and played(vice) and vice in eleven:
        total += pts.get(vice, 0)
    return total


def replay(weeks, cfg, allow_transfers=True):
    squad, bank, ft, purchase = [], 100.0, 0, {}
    rows = []
    for t in sorted(weeks):
        w = weeks[t]
        cols = list(w["pts"].columns[:cfg["horizon"]])
        pts = w["pts"][cols]
        meta = w["meta"].copy()
        meta["sell"] = meta["buy"]
        for p in squad:
            meta.at[p, "sell"] = T.selling_price(purchase[p], meta.at[p, "buy"])
        pool = T.build_pool(pts, meta, squad, cfg["decay"])
        res = T.optimise(pts.loc[pool], meta, squad, bank, ft, cfg,
                         allow_transfers=allow_transfers or not squad, time_limit=30)
        step = res["plan"][0]
        fresh = not squad
        n_in = len(step["transfers_in"])
        for p in step["transfers_out"]:
            purchase.pop(p, None)
        for p in step["transfers_in"]:
            purchase[p] = meta.at[p, "buy"]
        squad, bank = step["squad"], step["bank"]
        ft = 1 if fresh else min(cfg["max_ft"], ft - (n_in - step["hits"]) + 1)
        pos = meta["position"].to_dict()
        pts_gw = score_week(step, pos, w["actual"])
        rows.append(dict(gameweek=t, points=pts_gw - 4 * step["hits"], raw_points=pts_gw, hits=step["hits"],
                         transfers=0 if fresh else n_in, predicted_xi=step["xi_points"], bank=bank))
    return pd.DataFrame(rows)


def run_cfg(args):
    weeks, cfg, allow = args
    df = replay(weeks, cfg, allow)
    return cfg, allow, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default=None, help="e.g. 2025-26 (default: this season)")
    ap.add_argument("--grid", action="store_true", help="margin x FT value x horizon grid")
    ap.add_argument("--margin", type=float, default=T.DEFAULTS["margin"])
    ap.add_argument("--ft-value", type=float, default=T.DEFAULTS["ft_end_value"])
    ap.add_argument("--horizon", type=int, default=T.DEFAULTS["horizon"])
    ap.add_argument("--model", choices=["m3", "m5"], default="m5")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    suffix = f"_{a.season}" if a.season else ""

    t0 = time.time()
    weeks = precompute(a.season, a.model)
    print(f"Projections for GW{min(weeks)}-{max(weeks)} built in {time.time() - t0:.0f}s")

    base = dict(T.DEFAULTS, model=a.model)
    if a.grid:
        cfgs = [dict(base, margin=m, ft_end_value=f, horizon=h)
                for h, m, f in itertools.product([1, 2, 3, 4, 5], [0, 1, 2, 3], [0, 1, 2])]
    else:
        cfgs = [dict(base, margin=a.margin, ft_end_value=a.ft_value, horizon=a.horizon)]
    jobs = [(weeks, c, True) for c in cfgs] + [(weeks, dict(base, horizon=3), False)]

    with Pool(a.workers) as pool:
        results = pool.map(run_cfg, jobs)

    summary, by_gw = [], []
    for cfg, allow, df in results:
        label = "no transfers" if not allow else f"H{cfg['horizon']} m{cfg['margin']:g} ft{cfg['ft_end_value']:g}"
        summary.append(dict(setting=label, horizon=cfg["horizon"] if allow else np.nan,
                            margin=cfg["margin"] if allow else np.nan, ft_value=cfg["ft_end_value"] if allow else np.nan,
                            total=df.points.sum(), per_gw=df.points.mean(), transfers=df.transfers.sum(), hits=df.hits.sum()))
        by_gw.append(df.assign(setting=label))
    s = pd.DataFrame(summary).sort_values("total", ascending=False)
    hold = s.loc[s.setting == "no transfers", "total"].iloc[0]
    s["vs_no_transfers"] = s.total - hold
    os.makedirs(OUT, exist_ok=True)
    s.to_csv(os.path.join(OUT, f"transfers_grid{suffix}.csv"), index=False)
    pd.concat(by_gw).to_csv(os.path.join(OUT, f"transfers_by_gw{suffix}.csv"), index=False)
    pd.set_option("display.width", 200)
    print(f"Season {a.season or 'current'}, GW{min(weeks)}-{max(weeks)}, {time.time() - t0:.0f}s\n")
    print(s.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
