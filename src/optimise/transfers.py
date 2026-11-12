"""
transfers.py - the transfer optimiser. Reads an FPL team, then picks transfers (or "hold"),
the starting XI, captain and bench, by planning several gameweeks ahead.

WHAT IT MAXIMISES (over the next HORIZON gameweeks, default 2)
    sum over weeks k of  DECAY^k x [ XI expected points + captain's points again
                                     + BENCH_WEIGHT x outfield bench points ]
    - 4 per hit
    - MARGIN per transfer        (noise margin: a free transfer must gain > MARGIN,
                                  a hit must gain > 4 + MARGIN, or the model holds)
    + FT_END_VALUE per free transfer still banked after the last week, counting at most FT_CAP of them
      (FT_CAP=2: having 2 free transfers is valued, a 3rd-5th adds nothing)
  Only week 1 is acted on; weeks 2+ are the plan that makes week 1 sensible (re-run every week).

HARD RULES
    15-man squad 2 GKP / 5 DEF / 5 MID / 3 FWD, max 3 per club, XI = 1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD.
    Budget = bank + SELLING prices (you keep half of any rise, rounded down to £0.1m).
    Free transfers roll over up to 5; each extra transfer costs 4 points. Chips are ignored.

EXPECTED POINTS
    m5 (default) = m3 (P(appears) x points per appearance) x Poisson team-model factor per fixture,
    so fixture swings over the horizon count. --model m3 ignores fixtures (doubles/blanks still count).
    FPL's chance-of-playing applies in full to week 1 and recovers towards fit in later weeks
    (half the gap closes each week). Status u/n (left the club etc.) = 0 for every week.

USAGE (run fpl.py, fpl_fixtures.py and ingest.py first, as for predict_next.py):
    python src/optimise/transfers.py                 # asks for a team ID (offers ones used before)
    python src/optimise/transfers.py --team 123456 --horizon 2
    python src/optimise/transfers.py --team 123456 --ft 2 --margin 3

OUTPUT: data/recommendations/<team_id>_<team-name>/gw<N>_<date>_<HHMM>.csv  (15-man squad: role, xpts per GW)
        data/recommendations/<team_id>_<team-name>/gw<N>_<date>_<HHMM>.json (transfers, plan, settings used)
        data/recommendations/teams.csv                               (teams used before, for the prompt)

INSTALL: pip install pulp highspy requests pandas pyarrow scipy
"""

import argparse
import json
import math
import os
import re
import sys
from datetime import date, datetime

import numpy as np
import pandas as pd
import pulp

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)
from models.baseline import _team_model_factor, predict  # noqa: E402
from models import team_strength as ts  # noqa: E402
from agents.news_agent import apply_news, load_latest  # noqa: E402

ROOT = os.path.dirname(SRC)
PROC = os.path.join(ROOT, "data", "processed")
REC_DIR = os.path.join(ROOT, "data", "recommendations")
TEAMS_CSV = os.path.join(REC_DIR, "teams.csv")

API = "https://fantasy.premierleague.com/api/"
HEADERS = {"User-Agent": "Mozilla/5.0 (fpl-agents project)"}

# Defaults. MARGIN and FT_END_VALUE were picked by the season replay (src/evaluate/backtest_transfers.py).
DEFAULTS = dict(
    horizon=2,          # gameweeks planned (replay: 1-2 beat 3-5)
    decay=0.9,          # weight per week further out: 1, 0.9, 0.81, ...
    bench_weight=0.1,   # value of an outfield bench player's points (auto-sub chance)
    margin=2.0,         # noise margin per transfer (replay: 2-3 is a flat region, fewer hits)
    ft_end_value=1.0,   # value of each free transfer still banked after the horizon
    ft_cap=2,           # only this many banked free transfers count towards ft_end_value (5 = all)
    hit=4,
    max_ft=5,
    model="m5",
)
SQUAD = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_MIN = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX = {"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3}
POOL_TOP = {"GKP": 12, "DEF": 40, "MID": 40, "FWD": 25}   # best by expected points
POOL_CHEAP = 4                                          # plus the cheapest few per position
RECOVERY = 0.5   # share of an injured player's missing availability that returns each week


# ---------------------------------------------------------------------------------------------
# Expected points for every player over the next gameweeks
# ---------------------------------------------------------------------------------------------
def project_points(player_gw, roster, team_fixtures, team_preds, start_gw, horizon, model="m5", avail=None):
    """
    roster      : player_id, position, team (current club). Players with no history get 0.
    team_preds  : ts.predict_fixtures output for gameweeks start_gw .. start_gw+horizon-1
    avail       : optional DataFrame player_id, avail (week-1 chance of playing 0-1), status
    Returns a wide DataFrame: index player_id, one column per gameweek, expected points.
    """
    scheduled = set(team_fixtures.gameweek)
    gws = [g for g in range(start_gw, start_gw + horizon) if g in scheduled]   # stops at season end
    stub = roster[["player_id"]].assign(opponent=None, was_home=False)
    base = predict(player_gw, start_gw, stub, use_fixture=False)
    base = base.set_index("player_id")["m3_availability_x_ppa"].fillna(0.0)

    fx = team_fixtures[team_fixtures.gameweek.isin(gws)][["gameweek", "team", "opponent", "was_home"]]
    df = roster[["player_id", "position", "team"]].merge(fx, on="team", how="inner")
    if model == "m5":
        history = player_gw[player_gw.gameweek < start_gw]
        f, _, _ = _team_model_factor(df, history, team_preds)
    else:
        f = 1.0
    df["xpts"] = df["player_id"].map(base).fillna(0.0).to_numpy() * f

    if avail is not None:
        a = avail.set_index("player_id")
        df = df.join(a[["avail", "status"]], on="player_id")
        k = df.gameweek - start_gw
        a0 = df["avail"].fillna(1.0)
        df["a"] = a0 + (1 - a0) * (1 - RECOVERY ** k)
        df.loc[df["status"].isin(["u", "n"]), "a"] = 0.0
        df["xpts"] *= df["a"]

    wide = df.pivot_table(index="player_id", columns="gameweek", values="xpts", aggfunc="sum")
    wide = wide.reindex(index=roster.player_id, columns=gws).fillna(0.0)
    wide.columns = [int(c) for c in wide.columns]
    return wide


def build_pool(pts, meta, squad, decay):
    """Best players by discounted horizon points per position, the cheapest few, and the current squad."""
    w = np.array([decay ** k for k in range(pts.shape[1])])
    score = pd.Series(pts.to_numpy() @ w, index=pts.index)
    m = meta.loc[meta.index.intersection(pts.index)].assign(score=score)
    keep = set(squad)
    for pos, n in POOL_TOP.items():
        mp = m[m.position == pos]
        keep |= set(mp.nlargest(n, "score").index)
        keep |= set(mp[mp.score > 0].nsmallest(POOL_CHEAP, "buy").index)
    return sorted(keep)


# ---------------------------------------------------------------------------------------------
# The optimisation model
# ---------------------------------------------------------------------------------------------
def build_watchlist(best, state, pts, pts_free, meta, players, cfg, n_alt=2, n_capt=3, n_flag=5):
    """
    Players worth checking team news on: the ones a news item could change the plan for.
      squad     your current 15 (one may have been ruled out / returned)
      target    anyone the plan buys, in any planned week
      captain   the planned captain / vice, plus the next best few by expected points
      alt       the best few non-squad players per position (the runners-up a target might be swapped for)
      flagged   top scorers FPL currently flags as doubtful/injured. FPL's flag already pushed them down the
                ranking, so without this a player who is actually fit again could never be found
    pts is week-by-week expected points WITH FPL's availability; pts_free is the same without it.
    Returns a list of dicts: player_id, web_name, team, reasons (comma-separated).
    """
    w = np.array([cfg["decay"] ** k for k in range(pts.shape[1])])
    score = pd.Series(pts.to_numpy() @ w, index=pts.index)
    free = pd.Series(pts_free.to_numpy() @ w, index=pts_free.index)
    why = {}

    def add(p, reason):
        why.setdefault(int(p), set()).add(reason)

    squad = set(state["squad"])
    for p in squad:
        add(p, "squad")
    for step in best["plan"]:
        for p in set(step["squad"]) - squad:
            add(p, "target")
        add(step["captain"], "captain")
        if step.get("vice"):
            add(step["vice"], "captain")
    for p in score.drop(index=list(squad), errors="ignore").nlargest(n_capt).index:
        add(p, "captain")
    outside = score.drop(index=list(squad), errors="ignore")
    for pos in SQUAD:
        ids = [p for p in outside.index if p in meta.index and meta.at[p, "position"] == pos]
        for p in outside.loc[ids].nlargest(n_alt).index:
            add(p, "alt")
    info = players.set_index("player_id")
    risky = info[((info["status"] != "a") | (info["chance_of_playing_next_round"].fillna(100) < 100))
                 & ~info["status"].isin(["u", "n"])].index
    for p in free.loc[free.index.intersection(risky)].nlargest(n_flag).index:
        add(p, "flagged")
    return [dict(player_id=p, web_name=meta.at[p, "web_name"], team=meta.at[p, "team"],
                 reasons=",".join(sorted(r))) for p, r in why.items() if p in meta.index]


def get_solver(time_limit=20):
    for name in ("HiGHS", "PULP_CBC_CMD", "HiGHS_CMD"):
        try:
            s = pulp.getSolver(name, msg=False, timeLimit=time_limit, gapRel=0.001)
            if s.available():
                return s
        except Exception:
            continue
    raise SystemExit("No MILP solver found. Run: pip install highspy   (or: pip install 'pulp[cbc]')")


def optimise(pts, meta, squad, bank, ft, cfg, allow_transfers=True, time_limit=20):
    """
    pts   : DataFrame index player_id, columns = gameweeks (in order), expected points
    meta  : DataFrame index player_id with position, team, buy (£m), sell (£m; = buy if not owned)
    squad : current 15 player_ids, or [] to build a squad from scratch (week-1 transfers free)
    bank  : money in the bank (£m); ft: free transfers available in week 1
    Returns a dict with the week-by-week plan; week 1 is the one to act on.
    """
    P = [p for p in pts.index if p in meta.index]
    W = list(pts.columns)
    owned = set(squad)
    fresh = len(squad) == 0
    pos = meta["position"].to_dict()
    team = meta["team"].to_dict()
    buy = {p: int(round(meta.at[p, "buy"] * 10)) for p in P}           # tenths of £m
    sell = {p: int(round(meta.at[p, "sell"] * 10)) if p in owned else buy[p] for p in P}
    xp = {(p, w): float(pts.at[p, w]) for p in P for w in W}
    disc = {w: cfg["decay"] ** k for k, w in enumerate(W)}

    prob = pulp.LpProblem("fpl_transfers", pulp.LpMaximize)
    B = pulp.LpBinary
    s = pulp.LpVariable.dicts("squad", (P, W), cat=B)
    x = pulp.LpVariable.dicts("xi", (P, W), cat=B)
    c = pulp.LpVariable.dicts("capt", (P, W), cat=B)
    tin = pulp.LpVariable.dicts("in", (P, W), cat=B)
    tout = pulp.LpVariable.dicts("out", (P, W), cat=B)
    money = pulp.LpVariable.dicts("bank", W, lowBound=0)
    hits = pulp.LpVariable.dicts("hits", W, lowBound=0, cat=pulp.LpInteger)
    ftv = pulp.LpVariable.dicts("ft", W + ["end"], lowBound=0, upBound=cfg["max_ft"], cat=pulp.LpInteger)
    ft_counted = pulp.LpVariable("ft_counted", lowBound=0, upBound=cfg.get("ft_cap", cfg["max_ft"]))
    prob += ft_counted <= ftv["end"]

    n_tr = {}
    for k, w in enumerate(W):
        prev_w = W[k - 1] if k else None
        n_tr[w] = pulp.lpSum(tin[p][w] for p in P)
        for p in P:
            prev = s[p][prev_w] if prev_w is not None else (1 if p in owned else 0)
            prob += s[p][w] == prev + tin[p][w] - tout[p][w]
            prob += tin[p][w] + tout[p][w] <= 1
            prob += x[p][w] <= s[p][w]
            prob += c[p][w] <= x[p][w]
        prev_money = money[prev_w] if prev_w is not None else int(round(bank * 10))
        prob += money[w] == prev_money + pulp.lpSum(sell[p] * tout[p][w] - buy[p] * tin[p][w] for p in P)
        for ps, n in SQUAD.items():
            prob += pulp.lpSum(s[p][w] for p in P if pos[p] == ps) == n
            prob += pulp.lpSum(x[p][w] for p in P if pos[p] == ps) >= XI_MIN[ps]
            prob += pulp.lpSum(x[p][w] for p in P if pos[p] == ps) <= XI_MAX[ps]
        prob += pulp.lpSum(x[p][w] for p in P) == 11
        prob += pulp.lpSum(c[p][w] for p in P) == 1
        for t in set(team[p] for p in P):
            # 3 per club; more is allowed only if a player you own moved clubs (FPL lets you keep him)
            limit = max(3, sum(1 for p in owned if team.get(p) == t))
            prob += pulp.lpSum(s[p][w] for p in P if team[p] == t) <= limit

        # free transfers: week-1 count is given; later weeks follow the rules
        if k == 0 and not fresh:
            prob += ftv[w] == min(ft, cfg["max_ft"])
        if not allow_transfers and not (fresh and k == 0):
            prob += n_tr[w] == 0
        if fresh and k == 0:
            prob += hits[w] == 0
        else:
            prob += hits[w] >= n_tr[w] - ftv[w]
            prob += hits[w] <= n_tr[w]
        nxt = W[k + 1] if k + 1 < len(W) else "end"
        carry = 0 if (fresh and k == 0) else ftv[w] - (n_tr[w] - hits[w])
        prob += ftv[nxt] <= carry + 1
        prob += ftv[nxt] >= 1

    pts_term = pulp.lpSum(
        disc[w] * pulp.lpSum(
            xp[p, w] * (x[p][w] + c[p][w]) + (cfg["bench_weight"] * xp[p, w] * (s[p][w] - x[p][w]) if pos[p] != "GKP" else 0)
            for p in P)
        for w in W)
    paid = [w for k, w in enumerate(W) if not (fresh and k == 0)]
    prob += (pts_term
             - cfg["hit"] * pulp.lpSum(hits[w] for w in W)
             - cfg["margin"] * pulp.lpSum(n_tr[w] for w in paid)
             + cfg["ft_end_value"] * ft_counted)

    status = prob.solve(get_solver(time_limit))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"Solver status: {pulp.LpStatus[status]}")

    val = lambda v: (v.varValue or 0) if hasattr(v, "varValue") else v  # noqa: E731
    plan = []
    for w in W:
        sq = [p for p in P if val(s[p][w]) > 0.5]
        xi = [p for p in sq if val(x[p][w]) > 0.5]
        cap = next(p for p in xi if val(c[p][w]) > 0.5)
        rest = sorted([p for p in xi if p != cap], key=lambda p: -xp[p, w])
        bench_gk = [p for p in sq if p not in xi and pos[p] == "GKP"]
        bench_out = sorted([p for p in sq if p not in xi and pos[p] != "GKP"], key=lambda p: -xp[p, w])
        plan.append(dict(
            gameweek=int(w),
            transfers_in=[p for p in P if val(tin[p][w]) > 0.5],
            transfers_out=[p for p in P if val(tout[p][w]) > 0.5],
            hits=int(round(val(hits[w]))), free_transfers=int(round(val(ftv[w]))),
            bank=round(val(money[w]) / 10, 1),
            squad=sq, xi=xi, captain=cap, vice=rest[0] if rest else None, bench=bench_gk + bench_out,
            xi_points=round(sum(xp[p, w] for p in xi) + xp[cap, w], 2),
        ))
    return dict(objective=pulp.value(prob.objective), plan=plan, ft_end=int(round(val(ftv["end"]))))


def selling_price(purchase, now):
    """FPL rule: you keep half of any price rise, rounded down to £0.1m; falls are passed on in full."""
    p, n = int(round(purchase * 10)), int(round(now * 10))
    return (p + (n - p) // 2) / 10 if n > p else n / 10


# ---------------------------------------------------------------------------------------------
# Reading a team from the FPL API
# ---------------------------------------------------------------------------------------------
def fetch(path):
    import requests
    r = requests.get(API + path, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def next_gameweek(get=fetch):
    events = get("bootstrap-static/")["events"]
    nxt = [e["id"] for e in events if e.get("is_next")]
    if not nxt:
        raise SystemExit("FPL shows no next gameweek (season over?).")
    return nxt[0]


def team_state(team_id, next_gw, prices, gw_prices, get=fetch):
    """
    Squad, bank, purchase/selling prices and free transfers for the deadline of next_gw.
    prices    : Series player_id -> current price (£m)
    gw_prices : DataFrame player_id, gameweek, price (to price a squad picked before any transfer)
    """
    entry = get(f"entry/{team_id}/")
    history = get(f"entry/{team_id}/history/")
    transfers = get(f"entry/{team_id}/transfers/")
    chips = {c["event"]: c["name"] for c in history.get("chips", [])}
    played = sorted(history.get("current", []), key=lambda e: e["event"])
    if not played:
        raise SystemExit("This team has not played a gameweek yet.")
    last = played[-1]["event"]

    # squad at the last deadline (a free hit reverts, so use the week before it)
    pick_gw = last - 1 if chips.get(last) == "freehit" and last > 1 else last
    picks = get(f"entry/{team_id}/event/{pick_gw}/picks/")
    squad = [p["element"] for p in picks["picks"]]
    bank = picks["entry_history"]["bank"] / 10

    # free transfers, replayed from the season history (max 5; a wildcard/free-hit week keeps the
    # banked FTs but adds none - checked against a real team on 2026-11-12)
    started = played[0]["event"]
    ft = 0
    for e in played:
        g = e["event"]
        if g == started:
            ft = 1
        elif chips.get(g) in ("wildcard", "freehit"):
            pass
        else:
            ft = min(5, max(ft - e["event_transfers"], 0) + 1)

    # transfers already made for the coming deadline
    fh_events = {g for g, n in chips.items() if n == "freehit"}
    pending = [t for t in transfers if t["event"] > last]
    for t in pending:
        squad = [t["element_in"] if p == t["element_out"] else p for p in squad]
        bank += (t["element_out_cost"] - t["element_in_cost"]) / 10
    ft = max(ft - len(pending), 0)

    # purchase price = cost of the latest (non-free-hit) transfer in, else price when the squad was picked
    purchase = {}
    for t in sorted(transfers, key=lambda t: t["time"]):
        if t["event"] not in fh_events and t["element_in"] in squad:
            purchase[t["element_in"]] = t["element_in_cost"] / 10
    first = gw_prices[gw_prices.gameweek == started].drop_duplicates("player_id").set_index("player_id")["price"]
    rows = []
    for p in squad:
        now = float(prices.get(p, np.nan))
        bought = purchase.get(p, first.get(p, now))
        rows.append(dict(player_id=p, purchase=bought, price=now, sell=selling_price(bought, now)))
    return dict(team_id=team_id, team_name=entry.get("name", str(team_id)),
                manager=f"{entry.get('player_first_name', '')} {entry.get('player_last_name', '')}".strip(),
                squad=squad, bank=round(bank, 1), ft=ft, prices=pd.DataFrame(rows),
                last_gw=last, pending=len(pending))


# ---------------------------------------------------------------------------------------------
# Team-ID prompt and saving
# ---------------------------------------------------------------------------------------------
def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "team"


def known_teams():
    return pd.read_csv(TEAMS_CSV) if os.path.exists(TEAMS_CSV) else pd.DataFrame(columns=["team_id", "team_name", "last_used"])


def ask_team_id():
    teams = known_teams()
    if len(teams):
        print("Teams used before:")
        for i, r in enumerate(teams.itertuples(), 1):
            print(f"  {i}. {r.team_name} ({r.team_id})")
        ans = input("Pick a number, or type a team ID: ").strip()
        if ans.isdigit() and 1 <= int(ans) <= len(teams):
            return int(teams.iloc[int(ans) - 1].team_id)
    else:
        ans = input("FPL team ID (the number in your team's points-page URL): ").strip()
    if not ans.isdigit():
        raise SystemExit("A team ID is a number.")
    return int(ans)


def remember_team(team_id, name):
    os.makedirs(REC_DIR, exist_ok=True)
    t = known_teams()
    t = t[t.team_id.astype(int) != team_id]
    t = pd.concat([pd.DataFrame([dict(team_id=team_id, team_name=name, last_used=date.today().isoformat())]), t])
    t.to_csv(TEAMS_CSV, index=False)


def save(result, state, pts, meta, cfg, gw, hold_obj, watch=None):
    out_dir = os.path.join(REC_DIR, f"{state['team_id']}_{slug(state['team_name'])}")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"gw{gw}_{datetime.now():%Y-%m-%d_%H%M}")
    w1 = result["plan"][0]
    roles = {p: "XI" for p in w1["xi"]} | {p: f"bench {i + 1}" for i, p in enumerate(w1["bench"])}
    roles[w1["captain"]] = "captain"
    if w1["vice"]:
        roles[w1["vice"]] = "vice"
    sq = meta.loc[w1["squad"], ["web_name", "team", "position", "buy"]].rename(columns={"buy": "price"})
    sq.insert(0, "role", sq.index.map(roles))
    sq["new"] = sq.index.isin(w1["transfers_in"])
    for g in pts.columns:
        sq[f"xpts_gw{g}"] = pts.loc[sq.index, g].round(2)
    order = {"captain": 0, "vice": 1, "XI": 2}
    sq = sq.sort_values(by=["role"], key=lambda r: r.map(lambda v: order.get(v, 3) if not str(v).startswith("bench") else 3 + int(v[-1])))
    sq.reset_index().to_csv(stem + ".csv", index=False, encoding="utf-8")

    name = lambda p: meta.at[p, "web_name"]  # noqa: E731
    js = dict(
        team_id=state["team_id"], team_name=state["team_name"], gameweek=gw, made=datetime.now().isoformat(timespec="minutes"),
        settings=cfg, free_transfers=state["ft"], bank=state["bank"],
        recommendation="hold" if not w1["transfers_in"] else "transfer",
        transfers_out=[name(p) for p in w1["transfers_out"]], transfers_in=[name(p) for p in w1["transfers_in"]],
        hits=w1["hits"], gain_vs_hold=round(result["objective"] - hold_obj, 2),
        captain=name(w1["captain"]), vice=name(w1["vice"]) if w1["vice"] else None,
        plan=[dict(gameweek=s["gameweek"], out=[name(p) for p in s["transfers_out"]], into=[name(p) for p in s["transfers_in"]],
                   hits=s["hits"], captain=name(s["captain"]), xi_points=s["xi_points"]) for s in result["plan"]],
    )
    js["watchlist"] = watch or []     # who the news agent should check (see build_watchlist)
    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump(js, f, indent=2, ensure_ascii=False)
    return stem, js


# ---------------------------------------------------------------------------------------------
def main(argv=None, get=fetch, ask=ask_team_id):
    ap = argparse.ArgumentParser(description="Recommend FPL transfers for a team.")
    ap.add_argument("--team", type=int, help="FPL team ID (asked for if left out)")
    ap.add_argument("--horizon", type=int, default=DEFAULTS["horizon"])
    ap.add_argument("--decay", type=float, default=DEFAULTS["decay"])
    ap.add_argument("--margin", type=float, default=DEFAULTS["margin"])
    ap.add_argument("--ft-value", type=float, default=DEFAULTS["ft_end_value"])
    ap.add_argument("--ft-cap", type=int, default=DEFAULTS["ft_cap"])
    ap.add_argument("--model", choices=["m3", "m5"], default=DEFAULTS["model"])
    ap.add_argument("--ft", type=int, help="override the free transfers worked out from your history")
    ap.add_argument("--bank", type=float, help="override money in the bank (£m)")
    ap.add_argument("--no-news", action="store_true", help="ignore the news agent, use FPL flags only")
    a = ap.parse_args(argv)
    cfg = dict(DEFAULTS, horizon=a.horizon, decay=a.decay, margin=a.margin, ft_end_value=a.ft_value, ft_cap=a.ft_cap, model=a.model)

    for f, hint in (("player_gw.parquet", "ingest.py"), ("fpl_players.parquet", "fpl.py"), ("fpl_team_fixtures.parquet", "fpl_fixtures.py")):
        if not os.path.exists(os.path.join(PROC, f)):
            raise SystemExit(f"Missing data/processed/{f} - run {hint} first.")
    player_gw = pd.read_parquet(os.path.join(PROC, "player_gw.parquet"))
    players = pd.read_parquet(os.path.join(PROC, "fpl_players.parquet")).rename(columns={"id": "player_id"})
    tf = pd.read_parquet(os.path.join(PROC, "fpl_team_fixtures.parquet"))
    snap = datetime.fromtimestamp(os.path.getmtime(os.path.join(PROC, "fpl_players.parquet"))).date()
    if snap != date.today():
        print(f"[warn] fpl_players.parquet is from {snap}: prices and injury flags may be stale. Run fpl.py first.")

    team_id = a.team or ask()
    gw = next_gameweek(get)
    if int(player_gw.gameweek.max()) < gw - 1:
        print(f"[warn] player_gw.parquet ends at GW{int(player_gw.gameweek.max())}; run ingest.py for GW{gw - 1}.")
    prices = players.set_index("player_id")["price"]
    state = team_state(team_id, gw, prices, player_gw[["player_id", "gameweek", "price"]], get)
    print(f"\n{state['team_name']} ({state['manager']}, ID {team_id}) - planning GW{gw}-{gw + cfg['horizon'] - 1}")
    if a.ft is not None:
        state["ft"] = a.ft
    if a.bank is not None:
        state["bank"] = a.bank
    print(f"Bank £{state['bank']:.1f}m, free transfers {state['ft']} (worked out from your history; use --ft to override)"
          + (f", {state['pending']} transfer(s) already made for GW{gw}" if state["pending"] else ""))
    remember_team(team_id, state["team_name"])

    # expected points
    model = ts.fit(tf, gw, prior=ts.last_season_prior(), carry=ts.CARRY)
    up = tf[tf.gameweek.between(gw, gw + cfg["horizon"] - 1)]
    team_preds = ts.predict_fixtures(model, up)
    roster = players[["player_id", "position", "team"]]
    avail = players[["player_id", "status"]].assign(
        avail=(players["chance_of_playing_next_round"].fillna(100) / 100).clip(0, 1))
    avail = apply_news(avail, None if a.no_news else load_latest(gw))   # news agent output, if any
    print(f"News overrides applied to {int((avail['avail_source'] == 'news').sum())} player(s).")
    pts = project_points(player_gw, roster, up, team_preds, gw, cfg["horizon"], cfg["model"], avail)

    meta = players.set_index("player_id")[["web_name", "team", "position", "price"]].rename(columns={"price": "buy"})
    meta["sell"] = meta["buy"]
    sp = state["prices"].set_index("player_id")["sell"]
    meta.loc[sp.index, "sell"] = sp
    pool = build_pool(pts, meta, state["squad"], cfg["decay"])
    best = optimise(pts.loc[pool], meta, state["squad"], state["bank"], state["ft"], cfg)
    hold = optimise(pts.loc[pool], meta, state["squad"], state["bank"], state["ft"], cfg, allow_transfers=False)
    pts_free = project_points(player_gw, roster, up, team_preds, gw, cfg["horizon"], cfg["model"], None)
    watch = build_watchlist(best, state, pts, pts_free, meta, players, cfg)
    stem, js = save(best, state, pts, meta, cfg, gw, hold["objective"], watch)

    w1 = best["plan"][0]
    print("\nRECOMMENDATION:", "HOLD (no transfer beats the margin)" if not w1["transfers_in"] else "")
    for o, i in zip(js["transfers_out"], js["transfers_in"]):
        print(f"  OUT {o:<18} IN {i}")
    if w1["hits"]:
        print(f"  Hits: {w1['hits']} (-{4 * w1['hits']} pts)")
    if w1["transfers_in"]:
        print(f"  Gain over holding: {js['gain_vs_hold']:+.1f} (weighted pts over {cfg['horizon']} GWs, after hits and margin)")
    print(f"  Captain {js['captain']}, vice {js['vice']}")
    sq = pd.read_csv(stem + ".csv")
    cols = ["role", "web_name", "team", "position", "price"] + [c for c in sq.columns if c.startswith("xpts")]
    print("\n" + sq[cols].to_string(index=False))
    print("\nPlan for later weeks (re-run each week; only this week is acted on):")
    for s in js["plan"][1:]:
        moves = ", ".join(f"{o} -> {i}" for o, i in zip(s["out"], s["into"])) or "hold"
        print(f"  GW{s['gameweek']}: {moves}" + (f" ({s['hits']} hit)" if s["hits"] else ""))
    print(f"\nSaved to {stem}.csv / .json")
    return stem, js


if __name__ == "__main__":
    main()
