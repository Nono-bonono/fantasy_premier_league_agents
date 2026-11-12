"""
tuesday.py - the Tuesday command: refresh the data, then review the gameweek that just finished.

    python src/tuesday.py                  # last finished gameweek, team used last
    python src/tuesday.py --team 123456
    python src/tuesday.py --gw 10          # review an earlier gameweek
    python src/tuesday.py --skip-refresh   # reuse the data on disk
    python src/tuesday.py --no-summary     # skip the LLM paragraph (template used instead)

WHAT YOU GET
    How your team did   points vs the gameweek average, captain, bench, the best XI you could have picked,
                        and what your transfers did compared with Friday's recommendation.
    How predictions     our model vs FPL's ep_next on every player (error, rank correlation, top-15 picks),
    panned out          your XI predicted vs actual, the biggest misses, and the season so far.

NEEDS (run these first or let it refresh): the gameweek must be finished in FPL and in player_gw.parquet.
  Predictions come from data/predictions/baseline_gw<N>_<date>.parquet, written by friday.py (predict_next.py);
  only files made on or before the day of the first kickoff count, so ep_next never "knows" the result.
  Friday's recommendation comes from data/reports/friday_gw<N>_*.json if friday.py was run.

OUTPUT (timestamped):
    data/reports/tuesday_gw<N>_<date>_<HHMM>.md / .json
    data/reports/ledger.csv   one row per team per gameweek, updated each run (your season record)

CODE computes every number. The LLM only writes the short summary paragraph from those facts.
"""

import argparse
import contextlib
import glob
import io
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pipeline  # noqa: E402
from agents import news_agent  # noqa: E402
from evaluate import compare_ep_next as cmp  # noqa: E402

PROC = pipeline.PROC
PRED_DIR = os.path.join(pipeline.ROOT, "data", "predictions")
MODEL_KEY, FPL_KEY = "xpts (ours)", "ep_next (FPL)"
N_MISSES = 5


def num(x, nd=1):
    return None if x is None or pd.isna(x) else round(float(x), nd)


# ---------------------------------------------------------------------------------------------
# Your team
# ---------------------------------------------------------------------------------------------
def lineup(picks_json, act, pl):
    """Per pick: points, minutes and the multiplier that really counted (auto-subs and vice-captain applied)."""
    picks = picks_json["picks"]
    subs = picks_json.get("automatic_subs") or []
    sub_in, sub_out = {s["element_in"] for s in subs}, {s["element_out"] for s in subs}
    cap_pick = next(p for p in picks if p["is_captain"])
    vice_pick = next((p for p in picks if p["is_vice_captain"]), None)
    cap_played = act["minutes"].get(cap_pick["element"], 0) > 0
    rows = []
    for p in picks:
        e = p["element"]
        mult = p["multiplier"]
        if e in sub_out:
            mult = 0
        if e in sub_in:
            mult = 1
        if vice_pick and e == vice_pick["element"] and not cap_played:
            mult = max(mult, cap_pick["multiplier"])
        info = pl.loc[e] if e in pl.index else None
        rows.append(dict(player_id=e, name=info["web_name"] if info is not None else str(e),
                         club=info["team"] if info is not None else "", pos=info["position"] if info is not None else "",
                         pts=int(act["total_points"].get(e, 0)), minutes=int(act["minutes"].get(e, 0)),
                         slot=p["position"], mult=mult, captain=bool(p["is_captain"]), vice=bool(p["is_vice_captain"]),
                         sub_in=e in sub_in, sub_out=e in sub_out))
    for r in rows:
        r["counted"] = r["mult"] * r["pts"]
    return rows


def best_xi(rows):
    """Best valid XI (and captain) from the 15, in hindsight. Returns (points, [names]) with captain counted twice."""
    by = {pos: sorted([r for r in rows if r["pos"] == pos], key=lambda r: -r["pts"]) for pos in ("GKP", "DEF", "MID", "FWD")}
    best = (-1, None)
    for d in range(3, 6):
        for m in range(2, 6):
            for f in range(1, 4):
                if d + m + f != 10:
                    continue
                xi = by["GKP"][:1] + by["DEF"][:d] + by["MID"][:m] + by["FWD"][:f]
                if len(xi) != 11:
                    continue
                total = sum(r["pts"] for r in xi) + max(r["pts"] for r in xi)
                if total > best[0]:
                    best = (total, [r["name"] for r in xi])
    return (best[0], best[1]) if best[1] else (None, None)


def team_review(team_id, gw, act, pl, gw_event, get):
    picks_json = get(f"entry/{team_id}/event/{gw}/picks/")
    moves = [t for t in get(f"entry/{team_id}/transfers/") if t["event"] == gw]
    name = get(f"entry/{team_id}/").get("name", str(team_id))
    rows = lineup(picks_json, act, pl)
    eh = picks_json["entry_history"]
    chip = picks_json.get("active_chip")
    counted = sum(r["counted"] for r in rows)
    cost = eh.get("event_transfers_cost", 0)
    xi = [r for r in rows if r["mult"] > 0]
    cap = next(r for r in rows if r["captain"])
    # who actually carried the armband: the vice if the captain didn't play
    armband = next((r for r in rows if r["vice"]), cap) if cap["minutes"] == 0 and any(r["vice"] and r["minutes"] > 0 for r in rows) else cap
    starters = [r for r in rows if r["slot"] <= 11]
    best_pick = max(starters, key=lambda r: r["pts"])
    mult = max(armband["mult"], 1)
    bench = [r for r in rows if r["slot"] > 11 and r["mult"] == 0]
    mp = best_xi(rows) if not chip else (None, None)
    t_in = [t["element_in"] for t in moves]
    t_out = [t["element_out"] for t in moves]
    pts_of = lambda e: int(act["total_points"].get(e, 0))  # noqa: E731
    nm = lambda e: pl.at[e, "web_name"] if e in pl.index else str(e)  # noqa: E731
    return dict(
        team_id=team_id, team_name=name, chip=chip, rows=rows,
        points_gross=eh["points"], cost=cost, points_net=eh["points"] - cost,
        counted_check=counted, check_ok=counted == eh["points"],
        average=gw_event.get("average_entry_score"), highest=gw_event.get("highest_score"),
        gw_rank=eh.get("rank"), overall_rank=eh.get("overall_rank"), total_points=eh.get("total_points"),
        captain=dict(name=armband["name"], pts=armband["pts"], mult=mult, minutes=armband["minutes"], named=cap["name"]),
        best_starter=dict(name=best_pick["name"], pts=best_pick["pts"]),
        captain_cost=(best_pick["pts"] - armband["pts"]) * (mult - 1) if best_pick["pts"] > armband["pts"] else 0,
        bench_points=eh.get("points_on_bench"), bench=[dict(name=r["name"], pts=r["pts"]) for r in bench],
        max_possible=mp[0], max_possible_xi=mp[1],
        made=dict(ins=[nm(e) for e in t_in], outs=[nm(e) for e in t_out], in_ids=t_in, out_ids=t_out,
                  in_pts=sum(pts_of(e) for e in t_in), out_pts=sum(pts_of(e) for e in t_out)),
    )


# ---------------------------------------------------------------------------------------------
# Friday's recommendation vs what happened
# ---------------------------------------------------------------------------------------------
def latest_friday(gw, team_id, reports_dir):
    for f in sorted(glob.glob(os.path.join(reports_dir, f"friday_gw{gw}_*.json")), reverse=True):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if int(d.get("team_id", -1)) == int(team_id):
            return d
    return None


def friday_review(fri, review, act, pl):
    if not fri:
        return None
    pts_of = lambda e: int(act["total_points"].get(e, 0))  # noqa: E731
    xi = fri["xi"]
    cap = fri["captain"]
    actual_xi = sum(pts_of(p["player_id"]) for p in xi) + pts_of(cap["player_id"])
    rec_in = [p["name"] for p in fri.get("ins", [])]
    rec_out = [m["out"]["name"] for m in fri.get("moves", [])]
    made = review["made"]
    out_ids = [m["out"].get("id") for m in fri.get("moves", [])]
    in_ids = [p["player_id"] for p in fri.get("ins", [])]
    rec_gain = None
    if out_ids and all(i is not None for i in out_ids):
        rec_gain = sum(pts_of(i) for i in in_ids) - sum(pts_of(i) for i in out_ids)
    return dict(
        recommendation=fri["recommendation"], rec_in=rec_in, rec_out=rec_out, rec_captain=cap["name"], rec_captain_pts=pts_of(cap["player_id"]),
        followed=sorted(rec_in) == sorted(made["ins"]), n_followed=len(set(rec_in) & set(made["ins"])), followed_captain=cap["name"] == review["captain"]["named"],
        predicted_xi_points=fri["xi_points"], actual_xi_points=actual_xi, rec_gain_one_gw=rec_gain,
        confidence=fri.get("confidence"), generated=fri.get("made"))


# ---------------------------------------------------------------------------------------------
# Predictions vs actual
# ---------------------------------------------------------------------------------------------
def league_scores(gw, gw_table, pred_dir):
    """(scores dict, merged frame, prediction file) or (None, None, reason) when no valid predictions exist."""
    with contextlib.redirect_stdout(io.StringIO()):
        files = cmp.find_prediction_files(pred_dir, gw_table, include_late=False)
    if gw not in files:
        return None, None, (f"no predictions logged on or before the day GW{gw} kicked off "
                            f"(friday.py logs them; late files are ignored because their ep_next may know the result)")
    path, made = files[gw]
    scores, df = cmp.score_gameweek(pd.read_parquet(path), gw_table, gw)
    s = {r["model"]: {k: num(r[k], 3) for k in ("mae", "rmse", "spearman", "top_n_actual")} | {"n": int(r["n_players"])}
         for r in scores.to_dict("records")}
    mins = gw_table[gw_table["gameweek"] == gw].set_index("player_id")["minutes"]
    df = df.assign(minutes=df["player_id"].map(mins))
    return s, df, os.path.basename(path)


def misses(df):
    cols = ["web_name", "team", "position", "xpts", "ep_next", "total_points", "minutes"]
    top = df.nlargest(40, "xpts")
    over = top.assign(err=top["xpts"] - top["total_points"]).nlargest(N_MISSES, "err")
    under = df.assign(err=df["total_points"] - df["xpts"]).nlargest(N_MISSES, "err")
    fmt = lambda d: [dict(name=r.web_name, club=r.team, pos=r.position, predicted=num(r.xpts), ep_next=num(r.ep_next),  # noqa: E731
                          actual=int(r.total_points), minutes=None if pd.isna(r.minutes) else int(r.minutes)) for r in d[cols].itertuples()]
    return fmt(over), fmt(under)


def xi_predicted(review, df):
    """Your XI: model prediction vs actual, using the logged (pre-deadline) predictions."""
    if df is None:
        return None
    px = df.set_index("player_id")["xpts"]
    rows = [r for r in review["rows"] if r["mult"] > 0]
    pred = [(r, px.get(r["player_id"])) for r in rows]
    tot_pred = sum((p or 0) * r["mult"] for r, p in pred)
    return dict(predicted=round(float(tot_pred), 1), actual=sum(r["counted"] for r in rows),
                players=[dict(name=r["name"], predicted=num(p), actual=r["pts"], mult=r["mult"]) for r, p in pred])


# ---------------------------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------------------------
LEDGER_COLS = ["gameweek", "team_id", "team_name", "points_gross", "hit_cost", "points_net", "gw_average", "captain", "captain_pts",
               "captain_cost", "bench_points", "max_possible", "followed_friday", "model_mae", "ep_mae", "model_spearman",
               "ep_spearman", "model_top15", "ep_top15", "generated"]


def update_ledger(path, f, now):
    t, s, fr = f["team"], f["scores"], f["friday"]
    row = dict(gameweek=f["gameweek"], team_id=t["team_id"], team_name=t["team_name"], points_gross=t["points_gross"], hit_cost=t["cost"],
               points_net=t["points_net"], gw_average=t["average"], captain=t["captain"]["name"], captain_pts=t["captain"]["pts"],
               captain_cost=t["captain_cost"], bench_points=t["bench_points"], max_possible=t["max_possible"],
               followed_friday=None if fr is None else bool(fr["followed"]),
               model_mae=(s or {}).get(MODEL_KEY, {}).get("mae"), ep_mae=(s or {}).get(FPL_KEY, {}).get("mae"),
               model_spearman=(s or {}).get(MODEL_KEY, {}).get("spearman"), ep_spearman=(s or {}).get(FPL_KEY, {}).get("spearman"),
               model_top15=(s or {}).get(MODEL_KEY, {}).get("top_n_actual"), ep_top15=(s or {}).get(FPL_KEY, {}).get("top_n_actual"),
               generated=now.isoformat(timespec="minutes"))
    old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=LEDGER_COLS)
    old = old[~((old["gameweek"] == row["gameweek"]) & (old["team_id"] == row["team_id"]))]
    new = pd.concat([old, pd.DataFrame([row])] if len(old) else [pd.DataFrame([row])], ignore_index=True).sort_values(["team_id", "gameweek"])
    new[LEDGER_COLS].to_csv(path, index=False, encoding="utf-8")
    return new[new["team_id"] == row["team_id"]]


def season_so_far(led):
    n = len(led)
    led = led.assign(**{c: pd.to_numeric(led[c], errors="coerce") for c in ("points_net", "gw_average", "model_top15", "ep_top15", "model_mae", "ep_mae")})
    d = led.dropna(subset=["points_net", "gw_average"])
    out = dict(gameweeks=n, vs_average=None if d.empty else round(float((d["points_net"] - d["gw_average"]).sum()), 1),
               gws_above_average=int((d["points_net"] > d["gw_average"]).sum()), gws_with_average=len(d))
    p = led.dropna(subset=["model_top15", "ep_top15"])
    out["pred_gws"] = len(p)
    if len(p):
        out.update(model_top15=round(float(p["model_top15"].mean()), 2), ep_top15=round(float(p["ep_top15"].mean()), 2),
                   model_mae=round(float(p["model_mae"].mean()), 3), ep_mae=round(float(p["ep_mae"].mean()), 3),
                   model_wins_top15=int((p["model_top15"] > p["ep_top15"]).sum()))
    return out


# ---------------------------------------------------------------------------------------------
# Summary and report
# ---------------------------------------------------------------------------------------------
SUMMARY_SYSTEM = """You write the summary paragraph of a weekly Fantasy Premier League review.
Use ONLY the facts in the JSON. Quote numbers exactly as given; do not compute new ones.
- 110 words at most, plain prose, no bullets, no headings.
- Cover: how the team did against the gameweek average, the captain call, what the transfers did (and whether Friday's advice was followed, if given), and how the model's predictions compared with FPL's ep_next.
- One gameweek is mostly noise: say so if you compare the model with ep_next, and do not claim either is better from one week.
- If a fact is missing, leave it out. Do not give advice for next week."""


def summary_facts(f):
    t = f["team"]
    return dict(gameweek=f["gameweek"], points=t["points_gross"], hit_cost=t["cost"], net=t["points_net"], gw_average=t["average"], gw_highest=t["highest"],
                captain=t["captain"], best_starter=t["best_starter"], captain_cost=t["captain_cost"], bench_points=t["bench_points"],
                max_possible=t["max_possible"], transfers_made=dict(ins=t["made"]["ins"], outs=t["made"]["outs"], in_pts=t["made"]["in_pts"], out_pts=t["made"]["out_pts"]),
                friday=f["friday"], scores=f["scores"], no_predictions_reason=f["scores_note"], season=f["season"])


def template_summary(f):
    t = f["team"]
    s = f"GW{f['gameweek']}: {t['points_gross']} points" + (f" (-{t['cost']} hit = {t['points_net']})" if t["cost"] else "")
    if t["average"] is not None:
        s += f" against a gameweek average of {t['average']} ({t['points_net'] - t['average']:+d} net)"
    s += f". Captain {t['captain']['name']} scored {t['captain']['pts']}"
    s += f" (best starter was {t['best_starter']['name']} with {t['best_starter']['pts']})." if t["captain_cost"] else "."
    if f["scores"]:
        a, b = f["scores"].get(MODEL_KEY), f["scores"].get(FPL_KEY)
        if a and b:
            s += f" Our top 15 picks averaged {a['top_n_actual']} points vs {b['top_n_actual']} for FPL's ep_next (one gameweek is mostly noise)."
    return s


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    return out + ["| " + " | ".join("" if c is None else str(c) for c in r) + " |" for r in rows]


def render(f, summary, now):
    t, gw = f["team"], f["gameweek"]
    out = [f"# GW{gw} review - {t['team_name']}", "", f"Generated {now:%a %d %b %H:%M}" + (f" · chip played: {t['chip']}" if t["chip"] else "")]
    if f["stale"]:
        out += ["", "> **WARNINGS**"] + [f"> - {s}" for s in f["stale"]]
    out += ["", "## Summary", "", summary, "", "## How your team did", ""]
    pts = f"**{t['points_gross']} points**" + (f", -{t['cost']} for transfers = **{t['points_net']} net**" if t["cost"] else "")
    if t["average"] is not None:
        pts += f". Gameweek average {t['average']} ({t['points_net'] - t['average']:+d}), highest {t['highest']}"
    out.append(f"- Score: {pts}. Overall: {t['total_points']} points, rank {t['overall_rank']:,}." if t["overall_rank"] else f"- Score: {pts}.")
    c = t["captain"]
    cap_line = f"- Captain: {c['name']} {c['pts']} pts" + (f" (x{c['mult']} = {c['pts'] * c['mult']})" if c["mult"] > 1 else "")
    if c["name"] != c["named"]:
        cap_line += f" - your captain {c['named']} did not play, so the vice took the armband"
    cap_line += f". Best starter: {t['best_starter']['name']} {t['best_starter']['pts']}" + (f" - the captain call cost {t['captain_cost']} pts." if t["captain_cost"] else " - best call available.")
    out.append(cap_line)
    if t["bench"]:
        out.append(f"- Bench: {t['bench_points']} points left there (" + ", ".join(f"{b['name']} {b['pts']}" for b in t["bench"]) + ").")
    if t["max_possible"] is not None:
        out.append(f"- Best XI from your 15 in hindsight: {t['max_possible']} points (you scored {t['points_gross']}).")
    if not t["check_ok"]:
        out.append(f"- Note: player-level points add up to {t['counted_check']} vs FPL's {t['points_gross']}; FPL's number is used above.")
    m = t["made"]
    if m["ins"]:
        net = m["in_pts"] - m["out_pts"] - t["cost"]
        out.append(f"- Transfers made: " + ", ".join(f"{o} -> {i}" for o, i in zip(m["outs"], m["ins"])) + f". The players in scored {m['in_pts']}, the players out {m['out_pts']}"
                   f" ({net:+d} after the hit) - one gameweek of the plan only.")
    else:
        out.append("- Transfers made: none.")
    fr = f["friday"]
    if fr:
        line = f"- Friday said: {'hold' if fr['recommendation'] == 'hold' else ', '.join(f'{o} -> {i}' for o, i in zip(fr['rec_out'], fr['rec_in']))}; captain {fr['rec_captain']} ({fr['rec_captain_pts']} pts). "
        if fr["followed"]:
            line += "You followed the transfer advice. "
        elif fr["recommendation"] == "hold":
            line += "You made transfers instead of holding. "
        elif fr["n_followed"]:
            line += f"You made {fr['n_followed']} of the {len(fr['rec_in'])} recommended signings. "
        else:
            line += "You did not make the recommended transfers. "
        if fr["rec_gain_one_gw"] is not None and fr["recommendation"] != "hold":
            line += f"The recommended swaps scored {fr['rec_gain_one_gw']:+d} this gameweek."
        out.append(line)
    out += ["", "| Player | Pos | Pts | Counted | Notes |", "| --- | --- | --- | --- | --- |"]
    for r in sorted(t["rows"], key=lambda r: (r["slot"] > 11, r["slot"])):
        tags = [x for x, on in (("C", r["captain"]), ("VC", r["vice"]), ("auto-sub in", r["sub_in"]), ("auto-sub out", r["sub_out"]), ("bench", r["slot"] > 11 and r["mult"] == 0),
                                ("did not play", r["minutes"] == 0)) if on]
        out.append(f"| {r['name']} ({r['club']}) | {r['pos']} | {r['pts']} | {r['counted']} | {', '.join(tags)} |")

    out += ["", "## How the predictions panned out", ""]
    if not f["scores"]:
        out.append(f"Nothing to compare: {f['scores_note']}.")
    else:
        s = f["scores"]
        out += [f"All {s[MODEL_KEY]['n']} players with a fixture, against real points (logged file: {f['pred_file']}):", ""]
        out += table(["", "MAE", "RMSE", "Rank corr.", "Top-15 picks, avg actual pts"],
                     [[k, s[k]["mae"], s[k]["rmse"], s[k]["spearman"], s[k]["top_n_actual"]] for k in (MODEL_KEY, FPL_KEY) if k in s])
        out.append("\nOne gameweek is mostly noise; the season record below is what counts.")
        if f["xi"]:
            x = f["xi"]
            out += ["", f"Your XI: predicted {x['predicted']} points, scored {x['actual']}."]
        fr = f["friday"]
        if fr:
            out.append(f"Friday's recommended XI: predicted {fr['predicted_xi_points']}, would have scored {fr['actual_xi_points']} (before auto-subs).")
        for title, items in (("Over-predicted (model's top 40)", f["over"]), ("Under-predicted", f["under"])):
            out += ["", f"**{title}**", ""]
            out += table(["Player", "Pos", "Predicted", "ep_next", "Actual", "Minutes"],
                         [[f"{i['name']} ({i['club']})", i["pos"], i["predicted"], i["ep_next"], i["actual"], i["minutes"]] for i in items])
    se = f["season"]
    out += ["", "## Season so far", ""]
    line = f"- {se['gameweeks']} gameweek(s) logged"
    if se["vs_average"] is not None:
        line += f"; net points {se['vs_average']:+.1f} vs the gameweek average in total, above average in {se['gws_above_average']} of {se['gws_with_average']}"
    out.append(line + ".")
    if se["pred_gws"]:
        out.append(f"- Predictions over {se['pred_gws']} gameweek(s): top-15 average {se['model_top15']} (ours) vs {se['ep_top15']} (ep_next), ours higher in {se['model_wins_top15']}; "
                   f"MAE {se['model_mae']} vs {se['ep_mae']}.")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------------------------
def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Tuesday: review the gameweek that just finished.")
    ap.add_argument("--team", type=int, help="FPL team ID (default: the one used last)")
    ap.add_argument("--gw", type=int, help="gameweek to review (default: last finished)")
    ap.add_argument("--skip-refresh", action="store_true")
    ap.add_argument("--no-summary", action="store_true")
    ap.add_argument("--model", default=news_agent.DEFAULT_MODEL, help="OpenRouter model for the summary")
    ap.add_argument("--pred-dir", default=PRED_DIR)
    ap.add_argument("--out-dir", default=pipeline.REPORTS_DIR)
    a = ap.parse_args(argv)

    started = datetime.now()
    team_id = a.team
    if not team_id:
        d = pipeline.default_team()
        if not d:
            raise SystemExit("No team ID known yet: run  python src/tuesday.py --team <your FPL team ID>")
        team_id = d[0]
        print(f"Using {d[1]} ({team_id}) - the team used last. Pass --team to change.")
    os.makedirs(a.out_dir, exist_ok=True)

    steps = []
    if a.skip_refresh:
        print("Skipping data refresh.")
    else:
        print("Refreshing data ...")
        steps = pipeline.refresh()
    stale = pipeline.data_warnings(steps)

    get = pipeline.api_get
    try:
        events = get("bootstrap-static/")["events"]
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Could not reach the FPL API ({type(e).__name__}: {e}). Tuesday's review needs it for your picks.")
    done = [e["id"] for e in events if e.get("finished")]
    gw = a.gw or (max(done) if done else None)
    if gw is None:
        raise SystemExit("No gameweek has finished yet.")
    ev = next(e for e in events if e["id"] == gw)
    if not ev.get("finished"):
        stale.append(f"GW{gw} is not marked finished by FPL yet: points and bonus may still change")
    gw_table = pd.read_parquet(os.path.join(PROC, "player_gw.parquet"))
    if int(gw_table["gameweek"].max()) < gw:
        raise SystemExit(f"player_gw.parquet only goes up to GW{int(gw_table['gameweek'].max())}: the refresh did not pick up GW{gw}. "
                         f"{'; '.join(stale)}")
    print(f"Reviewing GW{gw} ...")
    pl = pd.read_parquet(os.path.join(PROC, "fpl_players.parquet")).set_index("id")
    act = gw_table[gw_table["gameweek"] == gw].set_index("player_id")[["minutes", "total_points"]]

    team = team_review(team_id, gw, act, pl, ev, get)
    fri_json = latest_friday(gw, team_id, a.out_dir)
    friday = friday_review(fri_json, team, act, pl)
    if not fri_json:
        stale.append(f"no friday report found for GW{gw}, so there is no recommendation to compare with")
    scores, df, note = league_scores(gw, gw_table, a.pred_dir)
    over = under = []
    xi = None
    if scores:
        over, under = misses(df)
        xi = xi_predicted(team, df)
    facts = dict(gameweek=gw, team=team, friday=friday, scores=scores, scores_note=None if scores else note,
                 pred_file=note if scores else None, xi=xi, over=over, under=under, stale=stale, steps=steps)
    facts["season"] = season_so_far(update_ledger(os.path.join(a.out_dir, "ledger.csv"), facts, started))

    summary, cost = None, 0.0
    if not a.no_summary:
        summary, res = pipeline.llm_text(SUMMARY_SYSTEM, json.dumps(summary_facts(facts), ensure_ascii=False, default=str), a.model, max_tokens=350)
        if summary is None:
            facts["stale"].append(f"the summary is a plain template because the LLM call failed ({res})")
        else:
            cost = res
    summary = summary or template_summary(facts)
    facts["summary_cost"] = cost

    report = render(facts, summary, started)
    base = os.path.join(a.out_dir, f"tuesday_gw{gw}_{started:%Y-%m-%d_%H%M}")
    open(base + ".md", "w", encoding="utf-8").write(report)
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump(dict(facts, summary=summary), fh, indent=2, ensure_ascii=False, default=str)
    print("\n" + "=" * 70 + "\n" + report + "=" * 70)
    print(f"Saved to {base}.md  (ledger: {os.path.join(a.out_dir, 'ledger.csv')})")
    return base


if __name__ == "__main__":
    main()
