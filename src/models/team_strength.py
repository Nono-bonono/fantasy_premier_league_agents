"""
team_strength.py - Poisson attack/defence model for teams.

Each team-in-a-match scores goals ~ Poisson(lambda):

    log(lambda) = mu + home_adv * was_home + attack[team] + defence[opponent]

  attack[team]      > 0: scores more than average.
  defence[opponent] > 0: LEAKY (concedes more than average); < 0: tight.
  Both are ridge-shrunk towards 0 (few matches -> ratings stay near average), and
  matches are weighted by recency (half-life in gameweeks).

From the two lambdas of a fixture we get what FPL needs:
  xG for / against, P(clean sheet) = exp(-lambda_against), P(win/draw/loss) from a
  Poisson scoreline grid.

For a target gameweek t it uses ONLY finished fixtures with gameweek < t (no leakage).

USAGE:
    python src/models/team_strength.py            # fit on all finished GWs, write ratings + upcoming fixture predictions
    from models.team_strength import fit, predict_fixtures
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

HALF_LIFE = 12.0   # gameweeks; picked by backtest (see src/evaluate/backtest_team_strength.py)
RIDGE = 10.0       # shrinkage on attack/defence (bigger = closer to the prior)
CARRY = 0.75       # share of last season's rating used as the prior (0 = start every team at average)
MAX_GOALS = 10

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC = os.path.join(ROOT, "data", "processed")


def load_team_fixtures():
    return pd.read_parquet(os.path.join(PROC, "fpl_team_fixtures.parquet"))


def last_season_team_fixtures():
    """Team-level results for 2025-26, one row per team per fixture (both games of a double GW).

    Reads team_fixtures_2025-26.parquet (written by fpl_last_season.py). Falls back to rebuilding from
    the player table, which holds one opponent per GW and so misses the second match of a double GW.
    """
    path = os.path.join(PROC, "team_fixtures_2025-26.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    print("[warn] team_fixtures_2025-26.parquet not found - run src/scrapers/fpl_last_season.py. "
          "Falling back to the player table (double-GW second matches missing).")
    d = pd.read_parquet(os.path.join(PROC, "player_gw_2025-26.parquet"))
    d = d.drop_duplicates(["fixture_id", "team"])
    h = d.was_home.astype(bool)
    out = pd.DataFrame({
        "gameweek": d.gameweek, "fixture_id": d.fixture_id, "team": d.team, "opponent": d.opponent,
        "was_home": h, "finished": True,
        "team_score": np.where(h, d.team_h_score, d.team_a_score).astype(float),
        "opponent_score": np.where(h, d.team_a_score, d.team_h_score).astype(float)})
    return out.reset_index(drop=True)


def last_season_prior(ridge=RIDGE):
    """Full-season 2025-26 ratings, used as the starting point (prior) for this season's ratings."""
    lf = last_season_team_fixtures()
    m = fit(lf, int(lf.gameweek.max()) + 1, half_life=np.inf, ridge=ridge)
    return {"attack": m["attack"], "defence": m["defence"]}


def fit(team_fixtures, target_gw, half_life=HALF_LIFE, ridge=RIDGE, prior=None, carry=0.0):
    """Fit on finished rows with gameweek < target_gw. Returns a dict of parameters.

    prior/carry: ratings are shrunk towards carry * (last season's rating) instead of towards 0.
    Teams with no prior (newly promoted) are shrunk towards 0. carry=0 is the original model.
    """
    d = team_fixtures[(team_fixtures.finished) & (team_fixtures.gameweek < target_gw)].copy()
    teams = sorted(team_fixtures.team.unique())
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    a0 = np.array([carry * prior["attack"].get(t, 0.0) for t in teams]) if prior else np.zeros(n)
    d0 = np.array([carry * prior["defence"].get(t, 0.0) for t in teams]) if prior else np.zeros(n)
    ti = d.team.map(idx).to_numpy()
    oi = d.opponent.map(idx).to_numpy()
    home = d.was_home.astype(float).to_numpy()
    y = d.team_score.to_numpy()
    w = 0.5 ** ((target_gw - 1 - d.gameweek.to_numpy()) / half_life) if np.isfinite(half_life) else np.ones(len(d))

    def unpack(p):
        return p[0], p[1], p[2:2 + n], p[2 + n:]

    def nll(p):
        mu, ha, att, dfn = unpack(p)
        eta = mu + ha * home + att[ti] + dfn[oi]
        lam = np.exp(eta)
        val = np.sum(w * (lam - y * eta)) + ridge * (np.sum((att - a0) ** 2) + np.sum((dfn - d0) ** 2))
        r = w * (lam - y)
        g = np.zeros_like(p)
        g[0] = r.sum()
        g[1] = (r * home).sum()
        g[2:2 + n] = np.bincount(ti, r, n) + 2 * ridge * (att - a0)
        g[2 + n:] = np.bincount(oi, r, n) + 2 * ridge * (dfn - d0)
        return val, g

    p0 = np.zeros(2 + 2 * n)
    p0[0] = np.log(max(y.mean(), 0.1)) if len(y) else 0.3
    p0[2:2 + n], p0[2 + n:] = a0, d0
    res = minimize(nll, p0, jac=True, method="L-BFGS-B")
    mu, ha, att, dfn = unpack(res.x)
    return {"mu": mu, "home_adv": ha, "attack": dict(zip(teams, att)),
            "defence": dict(zip(teams, dfn)), "n_matches": len(d) // 2}


def lam(model, team, opponent, was_home):
    return float(np.exp(model["mu"] + model["home_adv"] * bool(was_home)
                        + model["attack"][team] + model["defence"][opponent]))


def scoreline_probs(lam_for, lam_against):
    """P(win), P(draw), P(loss) for 'team' from independent Poissons."""
    k = np.arange(MAX_GOALS + 1)
    grid = np.outer(poisson.pmf(k, lam_for), poisson.pmf(k, lam_against))
    return np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum()


def predict_fixtures(model, fixtures):
    """fixtures: DataFrame with team, opponent, was_home (one row per team per fixture)."""
    rows = []
    for r in fixtures.itertuples():
        lf = lam(model, r.team, r.opponent, r.was_home)
        la = lam(model, r.opponent, r.team, not r.was_home)
        pw, pd_, pl = scoreline_probs(lf, la)
        rows.append({"gameweek": r.gameweek, "team": r.team, "opponent": r.opponent,
                     "was_home": r.was_home, "xg_for": lf, "xg_against": la,
                     "p_clean_sheet": np.exp(-la), "p_win": pw, "p_draw": pd_, "p_loss": pl})
    return pd.DataFrame(rows)


def ratings_table(model):
    t = pd.DataFrame({"attack": model["attack"], "defence": model["defence"]})
    t["attack_mult"] = np.exp(t.attack)    # x league-average goals scored
    t["defence_mult"] = np.exp(t.defence)  # x league-average goals conceded (below 1 = tight)
    return t.sort_values("attack", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=5, help="future GWs to predict")
    a = ap.parse_args()
    tf = load_team_fixtures()
    next_gw = int(tf[~tf.finished].gameweek.min())
    prior = last_season_prior()
    m = fit(tf, next_gw, prior=prior, carry=CARRY)
    print(f"Fitted on {m['n_matches']} matches (GW1-{next_gw-1}); mu={m['mu']:.3f} "
          f"home_adv={m['home_adv']:.3f} (x{np.exp(m['home_adv']):.2f})")
    rt = ratings_table(m)
    print(rt.round(2).to_string())
    up = tf[(~tf.finished) & (tf.gameweek < next_gw + a.horizon)]
    preds = predict_fixtures(m, up)
    rt.to_csv(os.path.join(PROC, "team_strength_ratings.csv"))
    preds.to_parquet(os.path.join(PROC, "team_fixture_preds.parquet"), index=False)
    print(f"\nWrote team_strength_ratings.csv and team_fixture_preds.parquet (GW{next_gw}-{next_gw+a.horizon-1}, {len(preds)} rows)")


if __name__ == "__main__":
    main()
