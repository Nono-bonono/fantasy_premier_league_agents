"""
baseline.py - the simplest sensible expected-points model. Everything later has to beat this.

For a target gameweek t it uses ONLY rows from gameweeks < t (no leakage), and builds:

    expected points = P(appears) x points-per-appearance x fixture factor

  P(appears)         recency-weighted share of recent gameweeks the player got on the pitch
  points/appearance  recency-weighted, shrunk towards the position average
                     (empirical Bayes: few appearances -> trust the position average more,
                     which is the "5-minute sub who scored once" fix)
  fixture factor     opponent's points conceded to that position (shrunk to 1) x home/away

Also exposes the simpler rungs of the ladder so the backtest can show what each step adds.

USAGE (as a library):
    from models.baseline import predict
    preds = predict(player_gw, target_gw, fixtures)   # fixtures: player_id, opponent, was_home
"""

import numpy as np
import pandas as pd

HALF_LIFE = 4.0        # gameweeks; a result 4 GWs ago counts half as much as last week's
PPA_SHRINK_K = 4.0     # pseudo-appearances of the position average added to every player
OPP_SHRINK_K = 40.0    # pseudo-appearances for opponent-vs-position strength


def _decay(history, target_gw):
    return 0.5 ** ((target_gw - 1 - history["gameweek"]) / HALF_LIFE)


def _player_form(history, target_gw):
    """Per-player recency-weighted stats from gameweeks before target_gw."""
    h = history.copy()
    h["w"] = _decay(h, target_gw)
    h["app"] = (h["minutes"] > 0).astype(float)
    h["pts_app"] = h["total_points"].where(h["app"] == 1, 0.0)

    h["w_app"] = h["w"] * h["app"]
    h["w_pts"] = h["w"] * h["total_points"]
    h["w_pts_app"] = h["w"] * h["pts_app"]
    g = h.groupby("player_id")
    out = pd.DataFrame({
        "w_rows": g["w"].sum(),
        "w_apps": g["w_app"].sum(),
        "w_pts": g["w_pts"].sum(),
        "w_pts_app": g["w_pts_app"].sum(),
        "last_pts": h.sort_values("gameweek").groupby("player_id")["total_points"].last(),
    })
    out["n_apps"] = g["app"].sum()
    return out


def _position_means(history):
    played = history[history["minutes"] > 0]
    return played.groupby("position")["total_points"].mean()


def _opponent_factor(history):
    """Points per appearance conceded to each position by each opponent, relative to league."""
    played = history[history["minutes"] > 0]
    league = played.groupby("position")["total_points"].mean()
    g = played.groupby(["opponent", "position"])["total_points"].agg(["sum", "count"]).reset_index()
    g["league"] = g["position"].map(league)
    g["factor"] = (g["sum"] + OPP_SHRINK_K * g["league"]) / ((g["count"] + OPP_SHRINK_K) * g["league"])
    return g.set_index(["opponent", "position"])["factor"]


def _home_factor(history):
    played = history[history["minutes"] > 0]
    league = played.groupby("position")["total_points"].mean()
    ha = played.groupby(["position", "was_home"])["total_points"].mean().unstack()
    return {
        (pos, True): ha.loc[pos, True] / league[pos] for pos in ha.index
    } | {
        (pos, False): ha.loc[pos, False] / league[pos] for pos in ha.index
    }


def predict(player_gw, target_gw, fixtures, use_fixture=True):
    """
    player_gw : ingest output (player_id, position, gameweek, minutes, total_points, opponent, was_home)
    fixtures  : one row per player playing in target_gw: player_id, opponent, was_home
    Returns a DataFrame indexed like fixtures with every rung of the model ladder as a column.
    """
    history = player_gw[player_gw["gameweek"] < target_gw]
    pos_info = player_gw.drop_duplicates("player_id").set_index("player_id")[["position", "web_name", "team"]]

    form = _player_form(history, target_gw)
    pos_ppa = _position_means(history)
    pos_pts_gw = history.groupby("position")["total_points"].mean()

    df = fixtures[["player_id", "opponent", "was_home"]].drop_duplicates("player_id").copy()
    df = df.join(pos_info, on="player_id").join(form, on="player_id")
    df["pos_ppa"] = df["position"].map(pos_ppa)
    df["pos_pts_gw"] = df["position"].map(pos_pts_gw)
    df["has_history"] = df["w_rows"].notna() & (df["w_rows"] > 0)

    # rung 0: position average per gameweek
    df["m0_position_mean"] = df["pos_pts_gw"]
    # rung 1: last gameweek's points (what "form" often means to a casual manager)
    df["m1_last_gw"] = df["last_pts"].fillna(df["pos_pts_gw"])
    # rung 2: recency-weighted points per gameweek, zeros included
    df["m2_decayed_form"] = (df["w_pts"] / df["w_rows"]).where(df["has_history"], df["pos_pts_gw"])
    # rung 3: P(appears) x shrunk points per appearance
    p_app = (df["w_apps"] / df["w_rows"]).where(df["has_history"], 0.0)
    ppa = (df["w_pts_app"] + PPA_SHRINK_K * df["pos_ppa"]) / (df["w_apps"] + PPA_SHRINK_K)
    df["p_appear"] = p_app
    df["pts_per_app"] = ppa
    df["m3_availability_x_ppa"] = p_app * ppa
    # rung 4: x fixture factor
    if use_fixture:
        opp = _opponent_factor(history)
        home = _home_factor(history)
        keys = pd.MultiIndex.from_frame(df[["opponent", "position"]])
        df["opp_factor"] = opp.reindex(keys).to_numpy()
        df["opp_factor"] = df["opp_factor"].fillna(1.0)
        df["home_factor"] = [home.get((p, bool(h)), 1.0) for p, h in zip(df["position"], df["was_home"])]
        df["m4_with_fixture"] = df["m3_availability_x_ppa"] * df["opp_factor"] * df["home_factor"]
    return df.reset_index(drop=True)


MODELS = ["m0_position_mean", "m1_last_gw", "m2_decayed_form", "m3_availability_x_ppa", "m4_with_fixture"]
