"""
backtest_team_strength.py - walk-forward test of the Poisson team model.

For each GW t (from MIN_HISTORY+1) fit on GW < t, predict GW t goals. Compared with a no-skill
baseline (league-average goals with a home/away split, same data window).

Metrics (lower is better): Poisson log-loss per team-goal count, RPS on win/draw/loss,
Brier on clean sheet. Also a small grid over half-life and ridge.
USAGE: python src/evaluate/backtest_team_strength.py
"""
import os, sys, itertools
import numpy as np
import pandas as pd
from scipy.stats import poisson

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))
import team_strength as ts

MIN_HISTORY = 3


def rps(p, outcome):  # p = (win, draw, loss); outcome 0 win,1 draw,2 loss
    cp = np.cumsum(p)[:-1]
    co = np.cumsum(np.eye(3)[outcome])[:-1]
    return 0.5 * np.sum((cp - co) ** 2)


def run(tf, hl, ridge, baseline=False, prior=None, carry=0.0):
    out = []
    last = int(tf[tf.finished].gameweek.max())
    for t in range(MIN_HISTORY + 1, last + 1):
        m = ts.fit(tf, t, hl, ridge, prior=prior, carry=carry)
        if baseline:
            m = dict(m, attack={k: 0 for k in m["attack"]}, defence={k: 0 for k in m["defence"]})
            hist = tf[(tf.finished) & (tf.gameweek < t)]
            h, a = hist[hist.was_home].team_score.mean(), hist[~hist.was_home].team_score.mean()
            m["mu"], m["home_adv"] = np.log(a), np.log(h / a)
        g = tf[(tf.finished) & (tf.gameweek == t)]
        p = ts.predict_fixtures(m, g)
        p["goals"] = g.team_score.to_numpy(); p["conc"] = g.opponent_score.to_numpy()
        p["gw"] = t
        out.append(p)
    p = pd.concat(out)
    ll = -poisson.logpmf(p.goals, p.xg_for).mean()
    oc = np.where(p.goals > p.conc, 0, np.where(p.goals == p.conc, 1, 2))
    r = np.mean([rps((a, b, c), o) for a, b, c, o in zip(p.p_win, p.p_draw, p.p_loss, oc)])
    brier = np.mean((p.p_clean_sheet - (p.conc == 0)) ** 2)
    corr = np.corrcoef(p.xg_for, p.goals)[0, 1]
    return dict(logloss=ll, rps=r, cs_brier=brier, corr_xg_goals=corr, n=len(p), gws=f"{MIN_HISTORY+1}-{p.gw.max()}")


if __name__ == "__main__":
    tf = ts.load_team_fixtures()
    prior = ts.last_season_prior()
    rows = [dict(model="league avg + home/away", hl=np.inf, ridge=np.nan, carry=np.nan,
                 **run(tf, np.inf, 1e9, baseline=True))]
    for hl, ridge in itertools.product([12], [5, 10, 20]):
        rows.append(dict(model="poisson (no prior)", hl=hl, ridge=ridge, carry=0.0, **run(tf, hl, ridge)))
    for carry, ridge in itertools.product([0.25, 0.5, 0.75, 1.0], [3, 5, 10, 20]):
        rows.append(dict(model="poisson + last-season prior", hl=12, ridge=ridge, carry=carry,
                         **run(tf, 12, ridge, prior=prior, carry=carry)))
    res = pd.DataFrame(rows)
    print(res.round(4).to_string(index=False))
    os.makedirs(os.path.join(ts.ROOT, "data", "backtest"), exist_ok=True)
    res.to_csv(os.path.join(ts.ROOT, "data", "backtest", "team_strength_prior_grid.csv"), index=False)
