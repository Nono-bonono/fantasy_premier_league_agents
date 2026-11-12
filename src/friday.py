"""
friday.py - the Friday command: refresh the data, optimise, check team news, write the report.

    python src/friday.py                  # uses the team you used last
    python src/friday.py --team 123456
    python src/friday.py --skip-refresh   # reuse the data on disk (quick re-run)
    python src/friday.py --no-summary     # skip the LLM paragraph; the report is then fully deterministic
    python src/friday.py --verbose        # show the optimiser / news agent output live
Anything not recognised here is passed on to runner.py / transfers.py (--horizon, --ft, --bank, --margin, ...).

STEPS
    1. refresh: fpl.py, fpl_fixtures.py, fpl_history.py, ingest.py     (a failed step = STALE warning, run continues)
    2. runner.py: optimise on FPL flags -> news agent on the players that matter -> optimise again
    3. predict_next.py: logs this week's player predictions (with the news applied) for Tuesday's comparison
    4. report: recommendation, why, news that matters, risk and confidence

OUTPUT (timestamped):
    data/reports/friday_gw<N>_<date>_<HHMM>.md     the report (also printed)
    data/reports/friday_gw<N>_<date>_<HHMM>.json   the facts behind it, for tuesday.py
    data/reports/logs/friday_gw<N>_<date>_<HHMM>.log   everything runner.py printed

WHO WRITES WHAT: code computes every number, the news selection, and the risk / confidence rating. The LLM
only writes the short "Why" paragraph from those facts (a plain template is used if it is unavailable).
"""

import argparse
import contextlib
import glob
import io
import json
import os
import sys
import time
from datetime import datetime

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pipeline  # noqa: E402
from agents import news_agent, runner  # noqa: E402
from optimise import transfers  # noqa: E402

PROC = pipeline.PROC
POS_ORDER = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
CORE = {"squad", "target", "captain"}
CONF_RANK = news_agent.CONF_RANK


# ---------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------
class Tee(io.TextIOBase):
    """Write to a log file and (optionally) the console."""
    def __init__(self, f, console=None):
        self.f, self.console = f, console

    def write(self, s):
        self.f.write(s)
        if self.console:
            self.console.write(s)
            self.console.flush()
        return len(s)

    def flush(self):
        self.f.flush()


def num(x, nd=1):
    return None if x is None or pd.isna(x) else round(float(x), nd)


def latest_stem(team_id, gw):
    """Newest recommendation transfers.py saved for this team and gameweek (path without extension)."""
    files = sorted(glob.glob(os.path.join(transfers.REC_DIR, f"{team_id}_*", f"gw{gw}_*.json")))
    return files[-1][:-5] if files else None


def fixture_str(tf, team, gw):
    d = tf[(tf["gameweek"] == gw) & (tf["team"] == team)].sort_values("kickoff_time")
    if d.empty:
        return "blank"
    return " + ".join(f"{o} ({'H' if h else 'A'})" for o, h in zip(d["opponent"], d["was_home"]))


def effective_avail(pl, gw):
    """Chance of playing (0-1) per player as transfers.py saw it: FPL's flag, replaced by usable news."""
    av = pl[["id", "status"]].rename(columns={"id": "player_id"})
    av = av.assign(avail=(pl["chance_of_playing_next_round"].fillna(100) / 100).clip(0, 1).to_numpy())
    av = news_agent.apply_news(av, news_agent.load_latest(gw))
    av.loc[av["status"].isin(["u", "n"]), "avail"] = 0.0
    return av.set_index("player_id")


def resolve_name(pl, name, squad_ids):
    """web_name -> player row. Names repeat across clubs, so use the current squad to break ties."""
    c = pl[pl["web_name"] == name]
    if len(c) > 1 and squad_ids:
        c = c[c["id"].isin(squad_ids)]
    return c.iloc[0] if len(c) == 1 else None


# ---------------------------------------------------------------------------------------------
# Facts: everything the report says, as plain data
# ---------------------------------------------------------------------------------------------
def build_facts(stem, run, team_id, steps, stale, notes, squad_fn=None):
    rec = json.load(open(stem + ".json", encoding="utf-8"))
    sq = pd.read_csv(stem + ".csv")
    gw = rec["gameweek"]
    pl = pd.read_parquet(os.path.join(PROC, "fpl_players.parquet"))
    tf = pd.read_parquet(os.path.join(PROC, "fpl_team_fixtures.parquet"))
    av = effective_avail(pl, gw)
    xcols = [c for c in sq.columns if c.startswith("xpts_gw")]
    x0 = f"xpts_gw{gw}"
    by_id = pl.set_index("id")

    sq["bench"] = sq["role"].str.startswith("bench")
    sq["avail"] = sq["player_id"].map(av["avail"])
    sq["avail_src"] = sq["player_id"].map(av["avail_source"])
    sq["fixture"] = [fixture_str(tf, t, gw) for t in sq["team"]]
    players = []
    for r in sq.itertuples():
        players.append(dict(player_id=int(r.player_id), role=r.role, name=r.web_name, club=r.team, pos=r.position,
                            price=num(r.price), new=bool(r.new), xpts=num(getattr(r, x0)), avail=num(r.avail, 2),
                            avail_src=r.avail_src, fixture=r.fixture,
                            xpts_by_gw={c[len("xpts_gw"):]: num(getattr(r, c), 2) for c in xcols},
                            ep_next=num(by_id.at[r.player_id, "ep_next"]) if r.player_id in by_id.index else None))
    xi = [p for p in players if not p["role"].startswith("bench")]
    cap = next(p for p in xi if p["role"] == "captain")
    vice = next((p for p in xi if p["role"] == "vice"), None)

    # transfers: who goes out (details from the FPL table), who comes in
    squad_ids = None
    if rec["transfers_out"] and squad_fn:
        try:
            squad_ids = set(squad_fn(team_id, gw, pl))
        except Exception as e:  # noqa: BLE001
            notes.append(f"could not re-read your squad from FPL ({type(e).__name__}); outgoing player details may be missing")
    moves = []
    ins = [p for p in players if p["new"]]
    for name in rec["transfers_out"]:
        o = resolve_name(pl, name, squad_ids)
        moves.append(dict(out=dict(name=name, id=None if o is None else int(o["id"]), pos=None if o is None else o["position"], club=None if o is None else o["team"], price=None if o is None else num(o["price"]),
                                   status=None if o is None else o["status"],
                                   chance=None if o is None else num(o["chance_of_playing_next_round"], 0),
                                   ep_next=None if o is None else num(o["ep_next"]),
                                   fpl_news=None if o is None else (str(o["news"]) if pd.notna(o["news"]) else "")[:120],
                                   avail=None if o is None else num(av.at[o["id"], "avail"], 2),
                                   fixture=None if o is None else fixture_str(tf, o["team"], gw))))
    in_names = rec["transfers_in"]
    ins_sorted = sorted(ins, key=lambda p: in_names.index(p["name"]) if p["name"] in in_names else 99)
    # the optimiser does not pair players, so pair each outgoing player with an incoming one in the same position
    free_in = list(ins_sorted)
    for m in moves:
        j = next((k for k, p in enumerate(free_in) if p["pos"] == m["out"]["pos"]), 0 if free_in else None)
        m["into"] = free_in.pop(j)["name"] if j is not None else None
    ep_out = sum(m["out"]["ep_next"] or 0 for m in moves)
    ep_in = sum(p["ep_next"] or 0 for p in ins_sorted)
    all_known = all(m["out"]["ep_next"] is not None for m in moves) and all(p["ep_next"] is not None for p in ins_sorted)

    # news: only items that touch your squad, a target or a captain option
    watch = {(w["web_name"], w["team"]): set(w["reasons"].split(",")) for w in rec.get("watchlist", [])}
    news = []
    for n in run.get("news_used", []):
        reasons = watch.get((n["web_name"], n["team"]), set())
        n = dict(n, reasons=sorted(reasons), core=bool(reasons & CORE),
                 drop=(n["fpl_chance"] if n["fpl_chance"] == n["fpl_chance"] else 100) - n["chance_of_playing"])
        news.append(n)
    news.sort(key=lambda n: (not n["core"], -abs(n["drop"]), -CONF_RANK[n["confidence"]]))

    f = dict(
        team_id=team_id, team_name=rec["team_name"], gameweek=gw, made=rec["made"],
        recommendation=rec["recommendation"], transfers_out=rec["transfers_out"], transfers_in=rec["transfers_in"],
        hits=rec["hits"], gain_vs_hold=rec["gain_vs_hold"], free_transfers=rec["free_transfers"], bank=rec["bank"],
        horizon=rec["settings"]["horizon"], margin=rec["settings"]["margin"], model=rec["settings"]["model"],
        captain=cap, vice=vice, xi=xi, bench=[p for p in players if p["role"].startswith("bench")],
        xi_points=rec["plan"][0]["xi_points"], later_plan=rec["plan"][1:],
        moves=moves, ins=ins_sorted, ep_next_out=round(ep_out, 1), ep_next_in=round(ep_in, 1), ep_known=all_known,
        plan_changed_by_news=bool(run.get("plan_changed")), plan_before=run.get("plan_before"),
        news_used=news, news_shown=[n for n in news if n["core"]][:2],
        failed_clubs=sum(r.get("failed_clubs", 0) for r in run.get("rounds", [])),
        players_checked=run.get("players_checked", 0), news_cost=run.get("news_cost", 0.0),
        stale=stale, notes=notes, steps=steps, stem=stem,
    )
    f.update(assess(f))
    return f


# ---------------------------------------------------------------------------------------------
# Risk and confidence (all code; thresholds are judgment calls, not calibrated probabilities)
# ---------------------------------------------------------------------------------------------
def assess(f):
    """flags = (severity 1|2, text). Total severity 0 -> High, 1-2 -> Medium, 3+ -> Low."""
    flags = []
    if f["recommendation"] == "transfer":
        g = f["gain_vs_hold"]
        if g < 1.5:
            flags.append((2, f"Gain over holding is only {g:+.1f} weighted pts (after the {f['margin']:.0f}-pt noise margin per transfer"
                             f"{' and hits' if f['hits'] else ''}): barely clear of noise."))
        elif g < 3:
            flags.append((1, f"Gain over holding is modest ({g:+.1f} weighted pts)."))
        if f["hits"]:
            flags.append((1, f"Costs {f['hits']} hit(s) (-{4 * f['hits']} pts)."))
        if f["moves"] and f["ep_known"] and f["ep_next_in"] < f["ep_next_out"]:
            flags.append((1, f"FPL's own ep_next rates the incoming players lower for GW{f['gameweek']} "
                             f"({f['ep_next_in']} vs {f['ep_next_out']} outgoing); the case rests on later weeks / our model."))
    if f["plan_changed_by_news"]:
        flags.append((1, "The plan changed once team news was applied, so it depends on those reports being right."))
    for p in f["xi"]:
        if p["avail"] is not None and p["avail"] < 1:
            sev = 2 if (p["role"] == "captain" or p["avail"] <= 0.5) else 1
            flags.append((sev, f"{p['name']} ({'captain, ' if p['role'] == 'captain' else ''}in your XI) is a doubt: "
                               f"{p['avail'] * 100:.0f}% to play ({'team news' if p['avail_src'] == 'news' else 'FPL flag'})."))
    if f["vice"] and f["captain"]["xpts"] is not None and f["vice"]["xpts"] is not None:
        gap = f["captain"]["xpts"] - f["vice"]["xpts"]
        if gap < 0.3:
            flags.append((1, f"Captain is close to a toss-up: {f['captain']['name']} {f['captain']['xpts']} vs {f['vice']['name']} {f['vice']['xpts']} xPts."))
    if f["failed_clubs"]:
        flags.append((1, f"The news search failed for {f['failed_clubs']} club(s), so some availability news may be missing."))
    if f["players_checked"] == 0:
        flags.append((1, "No team news was checked, so only FPL's injury flags were used."))
    if f["stale"]:
        flags.append((2, "Some data may be out of date (see the warning at the top)."))
    total = sum(s for s, _ in flags)
    return dict(flags=[t for _, t in sorted(flags, key=lambda x: -x[0])],
                confidence="High" if total == 0 else "Medium" if total <= 2 else "Low", risk_score=total)


# ---------------------------------------------------------------------------------------------
# The written "Why"
# ---------------------------------------------------------------------------------------------
WHY_SYSTEM = """You write the "Why" paragraph of a weekly Fantasy Premier League report.
Use ONLY the facts in the JSON. Quote numbers exactly as given; do not compute new ones.
- 70 words at most, plain prose, no bullets, no headings.
- Start with the recommendation (the exact swaps, or hold).
- Explain it: what the incoming players offer (xpts = model expected points for that gameweek), why the outgoing players are worse (flags, ep_next, fixtures), and why the captain.
- 'gain' is weighted expected points over the horizon, after hits and a noise margin.
- If a fact is missing, leave it out. Do not predict results or promise outcomes. Do not list risks."""


def why_facts(f):
    slim = lambda p: {k: p[k] for k in ("name", "club", "pos", "price", "xpts", "ep_next", "fixture", "xpts_by_gw") if k in p}  # noqa: E731
    return dict(recommendation=f["recommendation"], gameweek=f["gameweek"], horizon_gws=f["horizon"], gain=f["gain_vs_hold"],
                hits=f["hits"], noise_margin_per_transfer=f["margin"], free_transfers=f["free_transfers"],
                out=[m["out"] for m in f["moves"]], into=[slim(p) for p in f["ins"]],
                captain=slim(f["captain"]), vice=slim(f["vice"]) if f["vice"] else None,
                doubts_in_xi=[dict(name=p["name"], chance=p["avail"]) for p in f["xi"] if p["avail"] is not None and p["avail"] < 1],
                news=[dict(name=n["web_name"], fpl=n["fpl_chance"], news=n["chance_of_playing"], evidence=n["evidence"]) for n in f["news_shown"]])


def template_why(f):
    c = f["captain"]
    if f["recommendation"] == "transfer":
        moves = " and ".join(f"{m['out']['name']} for {m['into']}" for m in f["moves"])
        s = (f"Swap {moves}. Over the next {f['horizon']} gameweeks this is worth {f['gain_vs_hold']:+.1f} weighted points more than holding, "
             f"after {'a ' + str(4 * f['hits']) + '-point hit and ' if f['hits'] else ''}the {f['margin']:.0f}-point noise margin per transfer.")
    else:
        s = (f"Hold. No transfer beats the {f['margin']:.0f}-point noise margin"
             f"{' plus the 4-point hit' if f['free_transfers'] < 2 else ''} over the next {f['horizon']} gameweeks.")
    return s + f" Captain {c['name']} ({c['xpts']} xPts)."


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------
def row(p):
    tags = [t for t, on in (("C", p["role"] == "captain"), ("VC", p["role"] == "vice"), ("NEW", p["new"])) if on]
    if p["avail"] is not None and p["avail"] < 1:
        tags.append(f"doubt {p['avail'] * 100:.0f}%")
    return f"| {p['pos']} | {p['name']} | {p['club']} | {p['fixture']} | {p['xpts']} | {', '.join(tags)} |"


def render(f, why, deadline, now):
    gw = f["gameweek"]
    out = [f"# GW{gw} recommendations - {f['team_name']}", ""]
    when = f"Generated {now:%a %d %b %H:%M}"
    if deadline:
        left = deadline - datetime.now(deadline.tzinfo)
        d, h = left.days, left.seconds // 3600
        when += f" · Deadline {deadline.astimezone():%a %d %b %H:%M} ({d}d {h}h away)" if left.total_seconds() > 0 else " · Deadline has passed"
    out.append(f"{when} · {f['free_transfers']} free transfer(s), bank £{f['bank']:.1f}m")
    if f["stale"]:
        out += ["", "> **STALE DATA / WARNINGS**"] + [f"> - {s}" for s in f["stale"]]

    if f["notes"]:
        out += ["", "Notes: " + "; ".join(f["notes"]) + "."]
    out += ["", "## Recommendation", ""]
    if f["recommendation"] == "transfer":
        out += [f"**Transfers:** " + "; ".join(f"OUT {m['out']['name']} -> IN {m['into']}" for m in f["moves"])
                + (f" ({f['hits']} hit, -{4 * f['hits']} pts)" if f["hits"] else "")]
    else:
        out += ["**HOLD** - make no transfer."]
    c, v = f["captain"], f["vice"]
    out += [f"**Captain:** {c['name']}" + (f" (vice {v['name']})" if v else ""), ""]
    if f["recommendation"] == "transfer":
        out += [f"Gain over holding: {f['gain_vs_hold']:+.1f} weighted points over {f['horizon']} gameweeks (after hits and the noise margin).", ""]
    out += [f"Starting XI (expected {f['xi_points']} points for GW{gw}, captain counted twice):", "",
            "| Pos | Player | Club | Fixture | xPts | Notes |", "| --- | --- | --- | --- | --- | --- |"]
    out += [row(p) for p in sorted(f["xi"], key=lambda p: (POS_ORDER[p["pos"]], -p["xpts"]))]
    out += ["", "Bench (in order): " + ", ".join(f"{p['name']} ({p['pos']}, {p['xpts']})" + (f" doubt {p['avail'] * 100:.0f}%" if p["avail"] is not None and p["avail"] < 1 else "")
                                                  for p in f["bench"])]
    if f["later_plan"]:
        out += ["", "Later weeks (re-run each Friday; only this week is acted on): " + "; ".join(
            f"GW{s['gameweek']}: " + (", ".join(f"{o} -> {i}" for o, i in zip(s["out"], s["into"])) or "hold") for s in f["later_plan"])]

    out += ["", "## Why", "", why, "", "## News that matters", ""]
    if f["news_shown"]:
        for n in f["news_shown"]:
            fpl = "no flag" if n["fpl_chance"] != n["fpl_chance"] else f"{n['fpl_chance']:.0f}%"
            out.append(f"- **{n['web_name']}** ({n['team']}): FPL {fpl} -> news {n['chance_of_playing']}% [{n['confidence']}]. "
                       f"{n['evidence']} ([source]({n['source_url']}), {n['source_date']})")
        if f["plan_changed_by_news"]:
            out.append(f"- This news changed the plan (before news: {runner.describe(f['plan_before'])}).")
    else:
        extra = len(f["news_used"])
        out.append("No team news affecting your squad, targets or captain options." + (f" ({extra} other availability update(s) found elsewhere in the league.)" if extra else ""))
    out.append(f"\nChecked {f['players_checked']} players" + (f"; news search cost ${f['news_cost']:.2f}." if f["news_cost"] else "."))

    out += ["", "## Risk and confidence", "", f"**Confidence: {f['confidence']}**", ""]
    out += [f"- {t}" for t in f["flags"]] or ["- Nothing flagged: no doubts in the XI, a clear gain, fresh data."]
    out += ["", "_Confidence is a rule-of-thumb rating from the flags above, not a calibrated probability._"]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------------------------
def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Friday: refresh, optimise, check news, write the report.",
                                 epilog="Other options are passed to runner.py / transfers.py.")
    ap.add_argument("--team", type=int, help="FPL team ID (default: the one used last)")
    ap.add_argument("--skip-refresh", action="store_true", help="don't re-download data")
    ap.add_argument("--no-summary", action="store_true", help="no LLM paragraph")
    ap.add_argument("--verbose", action="store_true", help="show optimiser / news output live")
    ap.add_argument("--model", default=news_agent.DEFAULT_MODEL, help="OpenRouter model for the news agent and the summary")
    ap.add_argument("--max-rounds", type=int, default=2)
    ap.add_argument("--out-dir", default=pipeline.REPORTS_DIR)
    args, extra = ap.parse_known_args(argv)

    started = datetime.now()
    team_id = args.team
    if not team_id:
        d = pipeline.default_team()
        if not d:
            raise SystemExit("No team ID known yet: run  python src/friday.py --team <your FPL team ID>")
        team_id = d[0]
        print(f"Using {d[1]} ({team_id}) - the team used last. Pass --team to change.")
    stamp = f"{started:%Y-%m-%d_%H%M}"
    os.makedirs(os.path.join(args.out_dir, "logs"), exist_ok=True)

    steps, stale, notes = [], [], []
    if args.skip_refresh:
        print("Skipping data refresh.")
    else:
        print("Refreshing data ...")
        steps = pipeline.refresh()
    stale += pipeline.data_warnings(steps)

    print("Optimising and checking team news (this takes a few minutes) ...", flush=True)
    log_path = os.path.join(args.out_dir, "logs", f"friday_{stamp}.log")
    t0 = time.time()
    try:
        with open(log_path, "w", encoding="utf-8") as lf:
            sink = Tee(lf, sys.stdout if args.verbose else None)
            with contextlib.redirect_stdout(sink):
                run_path = runner.run(["--team", str(team_id), "--model", args.model,
                                       "--max-rounds", str(args.max_rounds)] + extra)
    except (SystemExit, Exception) as e:  # noqa: BLE001
        tail = "\n".join(open(log_path, encoding="utf-8").read().strip().splitlines()[-8:])
        print(f"\nFAILED - no recommendation produced: {e}\n--- end of log ({log_path}) ---\n{tail}")
        raise SystemExit(1)
    print(f"  done ({time.time() - t0:.0f}s)")
    run = json.load(open(run_path, encoding="utf-8"))
    gw = run["gameweek"]

    pr = pipeline.run_script("predict_next.py", ["--gw", gw])
    steps.append(dict(pr, label="Log predictions"))
    if not pr["ok"]:
        notes.append(f"predict_next.py failed, so this week's predictions were not logged for Tuesday ({pr['tail'].splitlines()[-1] if pr['tail'] else ''})")

    stem = latest_stem(team_id, gw)
    if not stem:
        raise SystemExit(f"runner finished but no recommendation file was found for team {team_id}, GW{gw}.")
    age = (datetime.now() - datetime.strptime(os.path.basename(stem).split("_", 1)[1], "%Y-%m-%d_%H%M")).total_seconds()
    if age > 2 * 3600:
        stale.append(f"newest recommendation file is {age / 3600:.1f}h old - this run may not have saved one")

    def squad_fn(tid, g, pl):
        pgw = pd.read_parquet(os.path.join(PROC, "player_gw.parquet"), columns=["player_id", "gameweek", "price"])
        return transfers.team_state(tid, g, pl.set_index("id")["price"], pgw)["squad"]

    facts = build_facts(stem, run, team_id, steps, stale, notes, squad_fn)

    why, cost = None, 0.0
    if not args.no_summary:
        why, res = pipeline.llm_text(WHY_SYSTEM, json.dumps(why_facts(facts), ensure_ascii=False, default=str), args.model)
        if why is None:
            facts["notes"].append(f"the written summary is a plain template because the LLM call failed ({res})")
        else:
            cost = res
    why = why or template_why(facts)
    facts["summary_cost"] = cost

    report = render(facts, why, pipeline.gw_deadline(gw), started)
    base = os.path.join(args.out_dir, f"friday_gw{gw}_{stamp}")
    with open(base + ".md", "w", encoding="utf-8") as fh:
        fh.write(report)
    with open(base + ".json", "w", encoding="utf-8") as fh:
        json.dump(dict(facts, why=why), fh, indent=2, ensure_ascii=False, default=str)
    print("\n" + "=" * 70 + "\n" + report + "=" * 70)
    print(f"Saved to {base}.md  (facts: .json, log: {log_path})")
    return base


if __name__ == "__main__":
    main()
