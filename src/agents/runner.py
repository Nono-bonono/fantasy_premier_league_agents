"""
runner.py - the weekly pipeline in one command. Plain Python, no LLM except inside the news agent.

    1. Optimise on FPL's own injury flags only.                      (transfers.py --no-news)
    2. Build a watchlist from that plan: your squad, transfer targets, captain options, the best
       runners-up and a few top scorers FPL flags as doubtful.       (watchlist in the recommendation .json)
    3. Search team news for just those players.                      (news_agent.py --players ...)
    4. Optimise again with the news applied and compare.
    5. If the new plan includes a squad / target / captain player nobody has checked yet, check them
       and optimise once more. Stops after --max-rounds news rounds, or as soon as nothing new is left.

Writes data/runs/run_gw<N>_<date>_<HHMM>.json: the plan before and after, which news mattered,
and what the news cost.

USAGE (run fpl.py, fpl_fixtures.py and ingest.py first):
    python src/agents/runner.py --team 123456
    python src/agents/runner.py --team 123456 --model moonshotai/kimi-k2.6 --max-rounds 3
Any other options (--horizon, --ft, --bank, --margin ...) are passed on to transfers.py.
"""

import argparse
import json
import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from agents import news_agent  # noqa: E402
from optimise import transfers  # noqa: E402

RUNS_DIR = os.path.join(ROOT, "data", "runs")
CORE = {"squad", "target", "captain"}      # reasons that mean "the plan depends on this player"
CONF_RANK = news_agent.CONF_RANK


def plan_key(js):
    return (tuple(sorted(js["transfers_out"])), tuple(sorted(js["transfers_in"])), js["captain"])


def summary(js):
    keys = ("recommendation", "transfers_out", "transfers_in", "hits", "captain", "vice", "gain_vs_hold")
    return {k: js[k] for k in keys}


def describe(js):
    if not js["transfers_in"]:
        move = "hold"
    else:
        move = ", ".join(f"OUT {o} -> IN {i}" for o, i in zip(js["transfers_out"], js["transfers_in"]))
        if js["hits"]:
            move += f" ({js['hits']} hit)"
    return f"{move}; captain {js['captain']}"


def news_that_mattered(tables, min_confidence="medium"):
    """News rows that would change a player's availability (same rules as apply_news)."""
    rows = []
    for t in tables:
        if t is None or len(t) == 0:
            continue
        for r in t.to_dict("records"):
            fpl = 100 if r["fpl_chance"] != r["fpl_chance"] else r["fpl_chance"]      # NaN -> fully fit
            ok = bool(r["verified"]) and CONF_RANK[r["confidence"]] >= CONF_RANK[min_confidence]
            if ok and r["chance_of_playing"] != fpl and (r["chance_of_playing"] < fpl or r["confidence"] == "high"):
                rows.append({k: r[k] for k in ("team", "web_name", "fpl_chance", "chance_of_playing", "confidence",
                                                "evidence", "source_url", "source_date")})
    return rows


def run(argv=None, optimise_main=transfers.main, news_main=news_agent.main):
    ap = argparse.ArgumentParser(description="Optimise, check team news for the players that matter, re-optimise.",
                                 epilog="Other options are passed to transfers.py.")
    ap.add_argument("--team", type=int, help="FPL team ID (asked for if left out)")
    ap.add_argument("--model", default=news_agent.DEFAULT_MODEL, help="OpenRouter model for the news agent")
    ap.add_argument("--max-rounds", type=int, default=2, help="most news rounds (each = search + re-optimise)")
    ap.add_argument("--max-results", type=int, default=3, help="search results per club")
    ap.add_argument("--timeout", type=int, default=120, help="seconds per club before the news agent gives up on it")
    ap.add_argument("--out-dir", default=RUNS_DIR)
    args, extra = ap.parse_known_args(argv)
    tr_args = (["--team", str(args.team)] if args.team else []) + extra

    print("=== Step 1: optimise on FPL's injury flags only ===")
    _, first = optimise_main(tr_args + ["--no-news"])
    gw = first["gameweek"]
    tr_args = ["--team", str(first["team_id"])] + extra          # don't ask for the team again
    cur, checked, tables, cost, rounds = first, set(), [], 0.0, []

    for rnd in range(1, args.max_rounds + 1):
        todo = {w["player_id"] for w in cur["watchlist"]
                if rnd == 1 or set(w["reasons"].split(",")) & CORE}
        todo -= checked
        if not todo:
            print(f"\nRound {rnd}: nobody left to check.")
            break
        print(f"\n=== Round {rnd}: team news for {len(todo)} player(s) ===")
        try:
            res = news_main(["--gw", str(gw), "--players", ",".join(str(p) for p in sorted(todo)),
                             "--model", args.model, "--max-results", str(args.max_results),
                             "--timeout", str(args.timeout)])
        except SystemExit as e:
            print(f"News agent stopped: {e}")
            break
        checked |= todo
        cost += res["cost"] or 0
        tables.append(res["table"])
        print(f"\n=== Round {rnd}: optimise again with the news applied ===")
        _, new = optimise_main(tr_args)
        changed = plan_key(new) != plan_key(cur)
        rounds.append(dict(round=rnd, players_checked=len(todo), clubs=res["clubs"], failed_clubs=res["failed"],
                           news_cost=res["cost"], plan_changed=changed, plan=summary(new)))
        cur = new

    mattered = news_that_mattered(tables)
    changed = plan_key(cur) != plan_key(first)
    print("\n" + "=" * 60)
    print(f"GW{gw} - {first['team_name']}")
    print(f"  FPL flags only : {describe(first)}")
    print(f"  With news      : {describe(cur)}")
    print("  -> " + ("PLAN CHANGED because of team news." if changed else "plan unchanged by team news."))
    if mattered:
        print("\nNews that changed a player's availability:")
        for m in mattered:
            print(f"  {m['web_name']} ({m['team']}): FPL {m['fpl_chance']:.0f}% -> {m['chance_of_playing']}% "
                  f"[{m['confidence']}] {m['evidence']} ({m['source_date']}) {m['source_url']}")
    print(f"\nPlayers checked: {len(checked)}; news cost ${cost:.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"run_gw{gw}_{datetime.now():%Y-%m-%d_%H%M}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(gameweek=gw, team=first["team_name"], news_model=args.model, plan_before=summary(first),
                       plan_after=summary(cur), plan_changed=changed, rounds=rounds, news_used=mattered,
                       players_checked=len(checked), news_cost=cost), f, indent=2, ensure_ascii=False, default=str)
    print(f"Saved to {out}")
    return out


if __name__ == "__main__":
    run()
